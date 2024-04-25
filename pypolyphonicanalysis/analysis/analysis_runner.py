import itertools
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd
import seaborn as sn
import librosa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import BaseModel
from scipy.cluster.hierarchy import dendrogram
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from pypolyphonicanalysis.analysis.pitch_drift_detrenders.base_pitch_drift_detrender import BasePitchDriftDetrender
from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.datamodel.features.features import Features
from pypolyphonicanalysis.datamodel.tracks.track import (
    track_is_saved,
    load_track,
    Track,
)
from pypolyphonicanalysis.analysis.f0_processing.base_f0_processor import BaseF0Processor
from pypolyphonicanalysis.models.multiple_f0_estimation.base_multiple_f0_estimation_model import (
    BaseMultipleF0EstimationModel,
)
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import (
    FloatArray,
    get_estimated_times_and_frequencies_from_salience_map,
    save_f0_trajectories_csv,
    check_output_path,
    save_reconstructed_audio,
    plot_predictions,
    get_random_state,
    F0TimesAndFrequencies,
)
from textwrap import wrap

logger = logging.getLogger(__name__)


def get_activation_cache_path(settings: Settings) -> Path:
    path = Path(settings.data_directory_path).joinpath("activation_cache")
    check_output_path(path)
    return path


def reconstruct_gmm_from_parameters(gmm_parameters: list[tuple[float, float, float]], settings: Settings) -> GaussianMixture:
    weights, means, vars = zip(*gmm_parameters)
    gmm = GaussianMixture(
        n_components=len(gmm_parameters),
        covariance_type="full",
        random_state=get_random_state(settings),
    )
    gmm.means_ = np.array(means).reshape(-1, 1)
    gmm.covariances_ = np.array(vars).reshape(-1, 1, 1)
    gmm.weights_ = np.array(weights)
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
    return gmm


def save_gmm_parameters(weights_means_vars: list[tuple[float, float, float]], filename_prefix: str, output_path: Path) -> None:
    check_output_path(output_path)
    distribution_table_str = "Component\tWeight\tMean\tStd.\n"
    for idx in range(len(weights_means_vars)):
        distribution_table_str += f"{idx + 1}\t\t{weights_means_vars[idx][0]:.4f}\t{weights_means_vars[idx][1]:.2f}\t{weights_means_vars[idx][2] ** 0.5:.4f}\n"
    open(
        os.path.join(
            output_path,
            f"{filename_prefix}_estimated_distribution_parameters.txt",
        ),
        "w",
    ).write(distribution_table_str)


def _get_cluster_recording_idxs_from_child(node: FloatArray, children: FloatArray, number_of_labels: int) -> list[FloatArray]:
    nodes = []
    a, b = node.tolist()
    if a < number_of_labels:
        nodes.append(a)
    else:
        nodes.extend(_get_cluster_recording_idxs_from_child(children[a - number_of_labels], children, number_of_labels))
    if b < number_of_labels:
        nodes.append(b)
    else:
        nodes.extend(_get_cluster_recording_idxs_from_child(children[b - number_of_labels], children, number_of_labels))
    return nodes


class AnalysisResults(BaseModel):
    # Base F0s
    recording_f0s_dict: dict[Recording, tuple[list[float], list[list[float]]]]

    # Harmonic interval analysis
    recording_harmonic_intervals_dict: dict[Recording, list[float]]
    recording_harmonic_interval_gaussian_mixture_weights_means_variances: dict[Recording, list[tuple[float, float, float]]]
    harmonic_interval_distribution_distance_matrix: list[list[float]]
    harmonic_interval_distribution_cluster_weights_means_variances: dict[str, list[tuple[float, float, float]]]


class AutomaticAnalysisRunner:
    def __init__(
        self,
        output_path: Path,
        multiple_f0_estimation_model: BaseMultipleF0EstimationModel,
        processors: Sequence[BaseF0Processor],
        settings: Settings,
        detrender: BasePitchDriftDetrender | None = None,
    ) -> None:
        check_output_path(output_path)
        self._output_path = output_path
        self._model = multiple_f0_estimation_model
        self._processors = processors
        self._settings = settings
        self._detrender = detrender

    def _estimate_recording_f0s(self, recording: Recording) -> FloatArray:
        return self._model.predict_on_file(recording.file_path)[Features.SALIENCE_MAP]

    def _save_f0s(
        self,
        times: FloatArray,
        freqs: FloatArray,
        recording: Recording,
        stage: str,
        correction_values: FloatArray | None = None,
    ) -> None:
        recording_output_path = self._output_path.joinpath(recording.name)
        check_output_path(recording_output_path)
        name_prefix = f"{recording.name}_{stage}"
        save_f0_trajectories_csv(recording_output_path.joinpath(f"{name_prefix}.csv"), times.tolist(), freqs)
        plot_predictions(times, freqs, name_prefix, recording_output_path, self._settings.default_figsize, correction_values)

    def _save_harmonic_intervals(self, harmonic_intervals: FloatArray, recording: Recording, name: str) -> None:
        recording_output_path = self._output_path.joinpath(recording.name)
        check_output_path(recording_output_path)
        json.dump(
            harmonic_intervals.tolist(),
            open(
                recording_output_path.joinpath(f"{name}_harmonic_intervals.json"),
                "w",
            ),
        )

    def _get_harmonic_intervals(self, freqs: FloatArray) -> FloatArray:
        cents_above_a1 = 1200 * np.log2(freqs / librosa.note_to_hz("A1"), out=-1 * np.inf * np.ones_like(freqs), where=freqs != 0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in subtract")
            diffs = np.diff(cents_above_a1)
        harmonic_intervals = np.reshape(diffs, -1)
        harmonic_intervals = harmonic_intervals[~np.isnan(harmonic_intervals)]
        harmonic_intervals = harmonic_intervals[~np.isinf(harmonic_intervals)]
        if self._settings.squeeze_harmonic_intervals_into_one_octave:
            harmonic_intervals %= 1200
            harmonic_intervals = harmonic_intervals[harmonic_intervals != 0]
        return harmonic_intervals

    def _export_recording_harmonic_interval_distribution_plots_and_files(
        self,
        harmonic_intervals: FloatArray,
        kde: KernelDensity,
        gmm_weights_means_vars: list[tuple[float, float, float]],
        ground_truth_gmm_params: list[tuple[float, float, float]] | None,
        name: str,
    ) -> None:
        recording_output_path = self._output_path.joinpath(name)
        check_output_path(recording_output_path)
        gmm = reconstruct_gmm_from_parameters(gmm_weights_means_vars, self._settings)
        plt.figure(figsize=self._settings.default_figsize)
        x = np.linspace(np.max([np.min(harmonic_intervals), 1]), np.max(harmonic_intervals), 1000).reshape(-1, 1)
        plt.hist(
            harmonic_intervals,
            bins=self._settings.histogram_bins,
            density=True,
            alpha=0.5,
            color="blue",
            label="Histogram",
        )
        plt.plot(x, np.exp(kde.score_samples(x)), color="red", label="Kernel Density Estimation")
        plt.plot(
            x,
            np.exp(gmm.score_samples(x)),
            color="green",
            linestyle="--",
            label="Gaussian Mixture Model",
        )
        if ground_truth_gmm_params is not None:
            ground_truth_gmm = reconstruct_gmm_from_parameters(ground_truth_gmm_params, self._settings)
            plt.plot(
                x,
                np.exp(ground_truth_gmm.score_samples(x)),
                color="blue",
                linestyle="dashed",
                label="Ground Truth GMM",
            )

        plt.legend()
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.title(f"Harmonic interval distribution of {name}")
        plt.savefig(recording_output_path.joinpath(f"{name}_harmonic_interval_distribution.jpg"))
        plt.close()

        weights_means_vars = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )
        save_gmm_parameters(weights_means_vars, name, recording_output_path)
        self._generate_gmm_derived_scale_example_file([mean for _, mean, _ in weights_means_vars], name, recording_output_path)

    def _export_cluster_harmonic_interval_distribution_plots_and_files(
        self,
        average_gmm_weights_means_vars: list[tuple[float, float, float]],
        individual_gmm_params: list[list[tuple[float, float, float]]],
        name: str,
        safe_filename_prefix: str,
    ) -> None:
        cluster_output_path = self._output_path.joinpath(safe_filename_prefix)
        check_output_path(cluster_output_path)
        save_gmm_parameters(average_gmm_weights_means_vars, safe_filename_prefix, cluster_output_path)
        x = np.linspace(1, 1300, 1000).reshape(-1, 1)
        plt.figure(figsize=self._settings.default_figsize)
        for single_gmm in [reconstruct_gmm_from_parameters(params, self._settings) for params in individual_gmm_params]:
            plt.plot(x, np.exp(single_gmm.score_samples(x.reshape(-1, 1))), color="black")
        gmm = reconstruct_gmm_from_parameters(average_gmm_weights_means_vars, self._settings)
        self._generate_gmm_derived_scale_example_file([mean for mean in gmm.means_.reshape(-1)], safe_filename_prefix, cluster_output_path)
        plt.plot(x, np.exp(gmm.score_samples(x)), color="red")
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.suptitle("Harmonic interval distribution (average of estimated Gaussian Mixtures)")
        plt.title(
            "\n".join(
                wrap(
                    f"{safe_filename_prefix}: {name if len(name) < 250 else f'{name[:250]}...]'}",
                    120,
                )
            ),
        )
        plt.savefig(cluster_output_path.joinpath(f"{safe_filename_prefix}_harmonic_interval_distribution.jpg"))
        plt.close()

    def _model_harmonic_interval_distribution(self, harmonic_intervals: FloatArray, name: str) -> tuple[list[tuple[float, float, float]], KernelDensity]:
        if len(harmonic_intervals) == 0:
            logger.warning(f"No harmonic intervals found in {name}")
            return [], KernelDensity()
        x = np.linspace(np.max([np.min(harmonic_intervals), 1]), np.max(harmonic_intervals), 1000).reshape(-1, 1)
        kde = KernelDensity(bandwidth=self._settings.density_estimation_bandwidth, kernel="gaussian")
        kde.fit(harmonic_intervals.reshape(-1, 1))
        peaks, _ = find_peaks(
            np.exp(kde.score_samples(x)),
            height=0,
            distance=self._settings.peak_finding_minimum_cent_distance,
        )
        peak_number = len(peaks)

        if peak_number == 0:
            logger.warning(f"Peak-finding returned 0 peaks for {name}")
            return [], KernelDensity()

        gmm = GaussianMixture(n_components=peak_number, random_state=get_random_state(self._settings))
        gmm.fit(harmonic_intervals.reshape(-1, 1))
        weights_means_vars = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )

        return weights_means_vars, kde

    def _generate_gmm_derived_scale_example_file(self, means: list[float], name: str, output_path: Path) -> None:
        check_output_path(output_path)
        sr = self._settings.sr
        arr = np.array([])
        for mean in sorted(means):
            freq = librosa.note_to_hz("C6") * 2 ** (mean / 1200)
            t = np.linspace(0.0, 0.5, int(sr * 0.5))
            y = np.sin(freq * t)
            arr = np.concatenate((arr, y))
        wavfile.write(output_path.joinpath(f"{name}_dervied_scale.wav"), sr, arr)

    def _get_harmonic_interval_ground_truth_gmm_parameters(self, recording: Recording) -> list[tuple[float, float, float]] | None:
        if recording.ground_truth_files is None:
            return None
        name = f"ground_truth_{recording.name}"
        tracks: list[Track] = []
        unsaved_tracks: list[Track] = []
        for ground_truth_file in recording.ground_truth_files:
            track_name = ground_truth_file.stem
            if track_is_saved(track_name, self._settings):
                track = load_track(track_name, self._settings)
            else:
                track = Track(track_name, ground_truth_file, self._settings)
                unsaved_tracks.append(track)
            tracks.append(track)
        freq_array = np.stack([track.f0_trajectory_annotation[1] for track in tracks]).transpose()
        freq_array = np.sort(freq_array, 1)
        harmonic_intervals = self._get_harmonic_intervals(freq_array)
        self._save_harmonic_intervals(harmonic_intervals, recording, "ground_truth")
        weights_means_vars, _ = self._model_harmonic_interval_distribution(harmonic_intervals, name)
        if self._settings.save_ground_truth_track_data:
            for track in unsaved_tracks:
                track.save()
        if len(weights_means_vars) == 0:
            return None
        return weights_means_vars

    def _apply_processing_to_times_and_freqs(self, recording: Recording, times: FloatArray, freqs: FloatArray) -> F0TimesAndFrequencies:
        if len(self._processors) == 0 and self._detrender is not None:
            correction_values = self._detrender.get_correction_values(times, freqs)
            self._save_f0s(times, freqs, recording, "initial", correction_values)
            freqs = self._detrender.detrend(freqs, correction_values)
            self._save_f0s(
                times,
                freqs,
                recording,
                "final_detrended",
            )
        elif len(self._processors) == 0:
            self._save_f0s(times, freqs, recording, "initial")
            self._save_f0s(times, freqs, recording, "final")
        else:
            self._save_f0s(times, freqs, recording, "initial")
            for idx, processor in enumerate(self._processors):
                times, freqs = processor.process(recording, times, freqs)
                if idx == len(self._processors) - 1:
                    if self._detrender is not None:
                        correction_values = self._detrender.get_correction_values(times, freqs)
                        self._save_f0s(times, freqs, recording, f"{processor.get_stage_name()}{'_final' if idx == len(self._processors) - 1 else ''}", correction_values)
                        freqs = self._detrender.detrend(freqs, correction_values)
                        self._save_f0s(times, freqs, recording, "final_detrended")
                    else:
                        self._save_f0s(times, freqs, recording, f"{processor.get_stage_name()}{'_final' if idx == len(self._processors) -1 else ''}")
                else:
                    self._save_f0s(times, freqs, recording, processor.get_stage_name())
        return times, freqs

    def _analyze_recording(self, recording: Recording) -> tuple[
        tuple[FloatArray, FloatArray],
        FloatArray,
        list[tuple[float, float, float]],
    ]:
        logger.info(f"Analyzing recording {recording.name}")
        if self._settings.use_activation_cache:
            activation_cache_path = get_activation_cache_path(self._settings)
            if os.path.isfile(activation_cache_path.joinpath(f"{recording.name}.npy")):
                initial_f0s = np.load(activation_cache_path.joinpath(f"{recording.name}.npy").absolute().as_posix())
            else:
                initial_f0s = self._estimate_recording_f0s(recording)
                np.save(
                    activation_cache_path.joinpath(f"{recording.name}.npy").absolute().as_posix(),
                    initial_f0s,
                )
        else:
            initial_f0s = self._estimate_recording_f0s(recording)
        times, freqs = get_estimated_times_and_frequencies_from_salience_map(initial_f0s, self._settings, True)
        if len(freqs[freqs != 0]) == 0:
            raise ValueError(f"No non-zero estimated frequencies in recording {recording.name}")
        recording_output_path = self._output_path.joinpath(recording.name)
        check_output_path(recording_output_path)
        save_reconstructed_audio(times, freqs, recording.name, recording_output_path, self._settings)
        times, freqs = self._apply_processing_to_times_and_freqs(recording, times, freqs)

        harmonic_intervals = self._get_harmonic_intervals(freqs)
        self._save_harmonic_intervals(harmonic_intervals, recording, recording.name)
        gaussian_mixture_weights_means_variances, kde = self._model_harmonic_interval_distribution(harmonic_intervals, recording.name)
        self._export_recording_harmonic_interval_distribution_plots_and_files(
            harmonic_intervals, kde, gaussian_mixture_weights_means_variances, self._get_harmonic_interval_ground_truth_gmm_parameters(recording), recording.name
        )

        return ((times, freqs), harmonic_intervals, gaussian_mixture_weights_means_variances)

    def _get_average_gmm_parameters(
        self,
        gmm_parameters: list[list[tuple[float, float, float]]],
    ) -> list[tuple[float, float, float]]:
        x = np.linspace(1, 1300, 1000).reshape(-1, 1)
        plt.figure(figsize=self._settings.default_figsize)
        combined_weights: list[float] = []
        combined_means: list[float] = []
        combined_vars: list[float] = []
        single_gmms: list[GaussianMixture] = []
        for gmm_params in gmm_parameters:
            if gmm_params == []:
                continue
            single_gmm = reconstruct_gmm_from_parameters(gmm_params, self._settings)
            combined_weights.extend((np.array(single_gmm.weights_).reshape(-1) / len(gmm_parameters)).tolist())
            combined_means.extend(single_gmm.means_.reshape(-1).tolist())
            combined_vars.extend(single_gmm.covariances_.reshape(-1).tolist())
            single_gmms.append(single_gmm)

        concatenated_gmm = reconstruct_gmm_from_parameters(list(zip(combined_weights, combined_means, combined_vars)), self._settings)
        kde = KernelDensity(bandwidth=self._settings.density_estimation_bandwidth, kernel="gaussian")
        gmm_samples = np.array(concatenated_gmm.sample(6000)[0]).reshape(-1, 1)
        kde.fit(gmm_samples)
        density_values = np.exp(kde.score_samples(x))

        peaks, _ = find_peaks(
            density_values,
            height=0,
            distance=self._settings.peak_finding_minimum_cent_distance,
        )
        peak_number = len(peaks)

        gmm = GaussianMixture(n_components=peak_number, random_state=get_random_state(self._settings))
        gmm.fit(kde.sample(6000))

        weights_means_vars = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )

        return weights_means_vars

    def _generate_distance_matrix(
        self,
        gmm_parameters: list[list[tuple[float, float, float]]],
        recordings: list[Recording],
    ) -> list[list[float]]:
        samples: list[list[float]] = []
        for idx, gmm_params in enumerate(gmm_parameters):
            if gmm_params == []:
                logger.info(f"Empty GMM found for {recordings[idx].name}, skipping sampling.")
                samples.append([0])
                continue
            weights, means, vars = zip(*gmm_params)
            cov = np.array(vars).reshape(-1, 1, 1)
            gmm = GaussianMixture(
                n_components=len(means),
                covariance_type="full",
                random_state=get_random_state(self._settings),
            )
            gmm.means_ = np.array(means).reshape(-1, 1)
            gmm.covariances_ = cov
            gmm.weights_ = weights
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
            samples.append(gmm.sample(6000)[0].reshape(-1).tolist())
        distance_matrix = np.zeros(shape=(len(samples), len(samples))).astype(np.float32)
        sample_idxs = list(range(len(samples)))
        for u, v in tqdm(
            list(itertools.product(sample_idxs, sample_idxs)),
            "Computing Wasserstein distances",
        ):
            distance_matrix[u, v] = wasserstein_distance(samples[u], samples[v])

        df_cm = pd.DataFrame(
            distance_matrix,
            index=[recording.name for recording in recordings],
            columns=[recording.name for recording in recordings],
        )
        plt.figure(figsize=self._settings.default_figsize)
        sn.heatmap(df_cm, annot=True, fmt="g")
        plt.title("Estimated interval distributions' Wasserstein distance matrix")
        plt.savefig(os.path.join(self._output_path, "distribution_distance_matrix.jpg"))
        plt.close()
        distance_matrix_list: list[list[float]] = distance_matrix.tolist()
        return distance_matrix_list

    def _cluster_gmms(
        self, distance_matrix: FloatArray, recordings: list[Recording], recording_gmm_param_dict: dict[Recording, list[tuple[float, float, float]]]
    ) -> list[list[Recording]]:
        clusters: list[list[Recording]] = []
        name_recording_dict: dict[str, Recording] = {recording.name: recording for recording in recordings}

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            distance_threshold=self._settings.clustering_distance_threshold,
            linkage="average",
        )
        clustering.fit(distance_matrix)

        counts = np.zeros(clustering.children_.shape[0])
        n_samples = len(clustering.labels_)
        for i, merge in enumerate(clustering.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        for child in clustering.children_:
            clusters.append([recordings[idx] for idx in _get_cluster_recording_idxs_from_child(child, clustering.children_, len(clustering.labels_))])

        linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)

        fig_width = 2 * max(len(recording.name) for recording in recordings)
        fig_height = int(distance_matrix.shape[0] * 1.5)

        axs: list[Axes]
        fig, axs = plt.subplots(1, 2, width_ratios=[0.1, 0.9], figsize=(fig_width, fig_height))
        distribution_ax = axs[0]
        dendrogram_ax = axs[1]

        dendrogram_ax.set_xlabel("Wasserstein distance")
        dendrogram_ax.set_title("Hierarchical Clustering Dendrogram of Estimated GMMs")

        dendrogram(
            linkage_matrix,
            ax=dendrogram_ax,
            labels=[recording.name for recording in recordings],
            orientation="right",
            leaf_font_size=6,
        )

        tick_labels = []
        tick_gmms = []
        for label_point in dendrogram_ax.get_ymajorticklabels():
            label_recording = name_recording_dict[label_point.get_text()]
            tick_gmms.append(reconstruct_gmm_from_parameters(recording_gmm_param_dict[label_recording], self._settings))
            lines = [label_point.get_text()]
            if label_recording.performers is not None:
                lines.append(label_recording.performers)
            if label_recording.recording_site is not None:
                lines.append(label_recording.recording_site)
            if label_recording.recording_date is not None:
                lines.append(label_recording.recording_date)
            tick_labels.append("\n".join(lines))

        dendrogram_ax.set_yticklabels(tick_labels)
        ticks = dendrogram_ax.get_yticks()
        tick_width = np.diff(ticks)[-1]
        y_lim = dendrogram_ax.get_ylim()

        distribution_ax.set_title("Interval distribution")
        distribution_ax.set_xlabel("Cents")
        distribution_ax.set_xlim(0, 1200)
        distribution_ax.set_ylim(y_lim)
        distribution_ax.set_yticks(ticks)
        distribution_ax.set_yticklabels(tick_labels, fontsize=6)
        distribution_ax.set_xticks(np.arange(0, 1200, 100))
        distribution_ax.tick_params(axis="x", labelsize=6)
        distribution_ax.grid(axis="x", color="red", linestyle="dashed")

        x = np.linspace(1, 1199, 1200).reshape(-1, 1)
        x_corners = np.arange(0, 1201, 1)
        y_corners = np.arange(0, len(recordings) + 1, 1) * tick_width
        exps = np.array([np.exp(gmm.score_samples(x)) for gmm in tick_gmms])
        color_data = exps / np.max(exps, axis=1).reshape(-1, 1)
        color_data = 1 / (1 + np.exp(-color_data))

        distribution_ax.pcolormesh(x_corners, y_corners, color_data, cmap="binary", norm=None)

        fig.savefig(self._output_path.joinpath("hierarchical_clustering_dendrogram_of_estimated_gmms.jpg"))
        plt.close(fig)

        return clusters

    def _analyze_cluster(
        self,
        cluster: list[Recording],
        recording_gaussian_mixture_weights_means_variances: dict[Recording, list[tuple[float, float, float]]],
        cluster_name: str,
        cluster_idx: int,
    ) -> list[tuple[float, float, float]]:
        logger.info(f"Analyzing cluster {cluster_name}")
        safe_filename_prefix = f"cluster_{cluster_idx}"
        cluster_gaussian_mixture_weights_means_variances = [recording_gaussian_mixture_weights_means_variances[recording] for recording in cluster]
        weights_means_variances = self._get_average_gmm_parameters(
            cluster_gaussian_mixture_weights_means_variances,
        )
        self._export_cluster_harmonic_interval_distribution_plots_and_files(
            weights_means_variances, cluster_gaussian_mixture_weights_means_variances, cluster_name, safe_filename_prefix
        )
        return weights_means_variances

    def generate_analysis_results(self, recordings: list[Recording]) -> AnalysisResults:
        recording_f0s_dict: dict[Recording, tuple[list[float], list[list[float]]]] = {}
        recording_harmonic_intervals_dict: dict[Recording, list[float]] = {}
        recording_harmonic_interval_gaussian_mixture_weights_means_variances: dict[Recording, list[tuple[float, float, float]]] = {}
        for recording in tqdm(recordings, desc="Analyzing recordings"):
            (
                recording_f0s,
                harmonic_intervals,
                gaussian_mixture_weights_means_variances,
            ) = self._analyze_recording(recording)
            recording_f0s_dict[recording] = (recording_f0s[0].tolist(), [[val for val in row if val > 0] for row in recording_f0s[1].tolist()])
            recording_harmonic_intervals_dict[recording] = harmonic_intervals.tolist()
            recording_harmonic_interval_gaussian_mixture_weights_means_variances[recording] = gaussian_mixture_weights_means_variances
        harmonic_gmm_parameters = list(recording_harmonic_interval_gaussian_mixture_weights_means_variances.values())
        harmonic_interval_distribution_distance_matrix: list[list[float]] = self._generate_distance_matrix(harmonic_gmm_parameters, recordings)

        harmonic_interval_distribution_clusters = (
            self._cluster_gmms(np.array(harmonic_interval_distribution_distance_matrix), recordings, recording_harmonic_interval_gaussian_mixture_weights_means_variances)
            if len(recordings) > 1
            else []
        )
        harmonic_interval_distribution_cluster_weights_means_variances: dict[str, list[tuple[float, float, float]]] = {}
        cluster_idx_name_dict: dict[int, str] = {}
        for cluster_idx, cluster in enumerate(tqdm(harmonic_interval_distribution_clusters, desc="Analyzing harmonic_interval_distribution_clusters")):
            cluster_name = json.dumps(sorted(recording.name for recording in cluster))
            cluster_idx_name_dict[cluster_idx] = cluster_name
            harmonic_interval_distribution_cluster_weights_means_variances[cluster_name] = self._analyze_cluster(
                cluster,
                recording_harmonic_interval_gaussian_mixture_weights_means_variances,
                cluster_name,
                cluster_idx,
            )
        json.dump(
            cluster_idx_name_dict,
            open(os.path.join(self._output_path, "cluster_idx_names.json"), "w"),
            indent=4,
        )

        results = AnalysisResults(
            recording_f0s_dict=recording_f0s_dict,
            recording_harmonic_intervals_dict=recording_harmonic_intervals_dict,
            recording_harmonic_interval_gaussian_mixture_weights_means_variances=recording_harmonic_interval_gaussian_mixture_weights_means_variances,
            harmonic_interval_distribution_distance_matrix=harmonic_interval_distribution_distance_matrix,
            harmonic_interval_distribution_cluster_weights_means_variances=harmonic_interval_distribution_cluster_weights_means_variances,
        )
        open(os.path.join(self._output_path, "analysis_results_serialization.json"), "w").write(results.model_dump_json())
        return results
