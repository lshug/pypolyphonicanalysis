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
from pydantic import BaseModel
from scipy.cluster.hierarchy import dendrogram
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from pypolyphonicanalysis.datamodel.features.features import Features
from pypolyphonicanalysis.datamodel.tracks.track import (
    track_is_saved,
    load_track,
    Track,
)
from pypolyphonicanalysis.processing.f0.base_f0_processor import BaseF0Processor
from pypolyphonicanalysis.models.base_multiple_f0_estimation_model import (
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
)
from textwrap import wrap

logger = logging.getLogger(__name__)


class Recording(BaseModel, frozen=True):
    name: str
    file_path: Path
    metadata_json: str = "{}"
    ground_truth_files: tuple[Path, ...] | None = None


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
    recording_f0s_dict: dict[Recording, tuple[list[float], list[list[float]]]]
    recording_harmonic_intervals_dict: dict[Recording, list[float]]
    recording_gaussian_mixture_weights_means_variances: dict[Recording, list[tuple[float, float, float]]]
    distance_matrix: list[list[float]]
    cluster_weights_means_variances: dict[str, list[tuple[float, float, float]]]


class AutomaticAnalysisRunner:
    def __init__(
        self,
        output_path: Path,
        model: BaseMultipleF0EstimationModel,
        processors: Sequence[BaseF0Processor],
        settings: Settings,
        activation_cache: Path | None = None,
    ) -> None:
        check_output_path(output_path)
        self._output_path = output_path
        self._model = model
        self._processors = processors
        self._cache_activations = activation_cache
        self._settings = settings

    def set_output_path(self, path: Path) -> None:
        check_output_path(path)
        self._output_path = path

    def _estimate_recording_f0s(self, recording: Recording) -> FloatArray:
        return self._model.predict_on_file(recording.file_path)[Features.SALIENCE_MAP]

    def _save_f0s(
        self,
        times: FloatArray,
        freqs: FloatArray,
        recording: Recording,
        stage: str,
    ) -> None:
        name_prefix = f"{recording.name}_{stage}"
        save_f0_trajectories_csv(self._output_path.joinpath(f"{name_prefix}.csv"), times.tolist(), freqs)
        plot_predictions(times, freqs, name_prefix, self._output_path, self._settings.default_figsize)

    def _get_harmonic_intervals(self, freqs: FloatArray, name: str) -> FloatArray:
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
        json.dump(
            harmonic_intervals.tolist(),
            open(
                self._output_path.joinpath(f"{name}_harmonic_intervals.json"),
                "w",
            ),
        )
        return harmonic_intervals

    def _export_harmonic_interval_distribution_plots_and_files(
        self,
        harmonic_intervals: FloatArray,
        kde: KernelDensity,
        gmm: GaussianMixture,
        ground_truth_gmm: GaussianMixture | None,
        name: str,
    ) -> None:
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
        if ground_truth_gmm is not None:
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
        plt.savefig(self._output_path.joinpath(f"{name}_harmonic_interval_distribution.jpg"))
        plt.close()

        weights_means_and_vars = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )

        self._save_gmm_parameters(weights_means_and_vars, name)

    def _model_harmonic_interval_distribution(self, harmonic_intervals: FloatArray, name: str) -> tuple[GaussianMixture, KernelDensity, list[tuple[float, float, float]]]:
        if len(harmonic_intervals) == 0:
            logger.warning(f"No harmonic intervals found in {name}")
            return GaussianMixture(), KernelDensity(), [(0, 0, 0)]
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
            return GaussianMixture(), KernelDensity(), [(0, 0, 0)]

        gmm = GaussianMixture(n_components=peak_number, random_state=get_random_state(self._settings))
        gmm.fit(harmonic_intervals.reshape(-1, 1))
        weights_means_and_vars = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )

        return gmm, kde, weights_means_and_vars

    def _generate_gmm_derived_scale_example_file(self, means: list[float], name: str) -> None:
        sr = self._settings.sr
        arr = np.array([])
        for mean in sorted(means):
            freq = librosa.note_to_hz("C6") * 2 ** (mean / 1200)
            t = np.linspace(0.0, 0.5, int(sr * 0.5))
            y = np.sin(freq * t)
            arr = np.concatenate((arr, y))
        wavfile.write(self._output_path.joinpath(f"{name}_dervied_scale.wav"), sr, arr)

    def _save_gmm_parameters(self, weights_means_and_vars: list[tuple[float, float, float]], filename_prefix: str) -> None:
        distribution_table_str = "Component\tWeight\tMean\tStd.\n"

        for idx in range(len(weights_means_and_vars)):
            distribution_table_str += f"{idx + 1}\t\t{weights_means_and_vars[idx][0]:.4f}\t{weights_means_and_vars[idx][1]:.2f}\t{weights_means_and_vars[idx][2] ** 0.5:.4f}\n"
        open(
            os.path.join(
                self._output_path,
                f"{filename_prefix}_estimated_distribution_parameters.txt",
            ),
            "w",
        ).write(distribution_table_str)

    def _get_ground_truth_gmm(self, recording: Recording) -> GaussianMixture | None:
        gmm: GaussianMixture | None
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
        harmonic_intervals = self._get_harmonic_intervals(freq_array, name)
        gmm, _, _ = self._model_harmonic_interval_distribution(harmonic_intervals, name)
        try:
            check_is_fitted(gmm)
        except NotFittedError:
            gmm = None
        if self._settings.save_raw_training_data:
            for track in unsaved_tracks:
                track.save()
        return gmm

    def _analyze_recording(self, recording: Recording) -> tuple[
        tuple[FloatArray, FloatArray],
        FloatArray,
        list[tuple[float, float, float]],
    ]:
        logger.info(f"Analyzing recording {recording.name}")
        if self._cache_activations is not None:
            self._cache_activations.mkdir(parents=True, exist_ok=True)
            if os.path.isfile(os.path.join(self._cache_activations, f"{recording.name}.npy")):
                initial_f0s = np.load(os.path.join(self._cache_activations, f"{recording.name}.npy"))
            else:
                initial_f0s = self._estimate_recording_f0s(recording)
                np.save(
                    os.path.join(self._cache_activations, f"{recording.name}.npy"),
                    initial_f0s,
                )
        else:
            initial_f0s = self._estimate_recording_f0s(recording)
        times, freqs = get_estimated_times_and_frequencies_from_salience_map(initial_f0s, self._settings, True)
        json.dump(
            json.loads(recording.metadata_json),
            open(
                os.path.join(self._output_path, f"{recording.name}.json"),
                "w",
                encoding="utf8",
            ),
        )
        save_reconstructed_audio(times, freqs, recording.name, self._output_path, self._settings)
        self._save_f0s(times, freqs, recording, "initial")
        for processor in self._processors:
            times, freqs = processor.process(times, freqs)
            self._save_f0s(times, freqs, recording, processor.get_stage_name())
        self._save_f0s(times, freqs, recording, "final")
        harmonic_intervals = self._get_harmonic_intervals(freqs, recording.name)
        gmm, kde, gaussian_mixture_weights_means_variances = self._model_harmonic_interval_distribution(harmonic_intervals, recording.name)
        self._export_harmonic_interval_distribution_plots_and_files(
            harmonic_intervals,
            kde,
            gmm,
            self._get_ground_truth_gmm(recording),
            recording.name,
        )
        self._generate_gmm_derived_scale_example_file(
            [mean for _, mean, _ in gaussian_mixture_weights_means_variances],
            recording.name,
        )
        return (
            (times, freqs),
            harmonic_intervals,
            gaussian_mixture_weights_means_variances,
        )

    def _analyze_gmm_collection(
        self,
        gmm_parameters: list[list[tuple[float, float, float]]],
        name: str,
        safe_filename_prefix: str,
    ) -> tuple[GaussianMixture, KernelDensity, list[tuple[float, float, float]]]:
        x = np.linspace(1, 1300, 1000).reshape(-1, 1)
        plt.figure(figsize=self._settings.default_figsize)
        combined_weights: list[float] = []
        combined_means: list[float] = []
        combined_vars: list[float] = []
        for gmm_params in gmm_parameters:
            if gmm_params == [(0, 0, 0)]:
                continue
            weights, means, vars = zip(*gmm_params)
            combined_weights.extend((np.array(weights) / len(gmm_parameters)).tolist())
            combined_means.extend(means)
            combined_vars.extend(vars)
            cov = np.array(vars).reshape(-1, 1, 1)
            single_gmm = GaussianMixture(
                n_components=len(means),
                covariance_type="full",
                random_state=get_random_state(self._settings),
            )
            single_gmm.means_ = np.array(means).reshape(-1, 1)
            single_gmm.covariances_ = cov
            single_gmm.weights_ = weights
            single_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
            plt.plot(x, np.exp(single_gmm.score_samples(x.reshape(-1, 1))))
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.suptitle("Harmonic interval distributions (estimated Gaussian Mixtures)")
        plt.title(
            "\n".join(
                wrap(
                    f"{safe_filename_prefix}: {name if len(name) < 250 else f'{name[:250]}...]'}",
                    120,
                )
            ),
            fontsize=5,
        )
        plt.savefig(self._output_path.joinpath(f"{safe_filename_prefix}_harmonic_interval_distributions.jpg"))
        plt.close()

        cov = np.array(combined_vars).reshape(-1, 1, 1)
        concatenated_gmm = GaussianMixture(
            n_components=len(combined_means),
            covariance_type="full",
            random_state=get_random_state(self._settings),
        )
        concatenated_gmm.means_ = np.array(combined_means).reshape(-1, 1)
        concatenated_gmm.covariances_ = cov
        concatenated_gmm.weights_ = np.array(combined_weights)
        concatenated_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))

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

        self._generate_gmm_derived_scale_example_file([mean for mean in gmm.means_.reshape(-1)], safe_filename_prefix)

        plt.figure(figsize=self._settings.default_figsize)
        plt.plot(x, np.exp(gmm.score_samples(x)))
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
        plt.savefig(self._output_path.joinpath(f"{safe_filename_prefix}_harmonic_interval_distribution.jpg"))
        plt.close()

        weights_means_and_vars = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )
        self._save_gmm_parameters(weights_means_and_vars, safe_filename_prefix)
        return gmm, kde, weights_means_and_vars

    def _generate_distance_matrix(
        self,
        gmm_parameters: list[list[tuple[float, float, float]]],
        recordings: list[Recording],
    ) -> tuple[list[list[float]], list[list[float]]]:
        samples: list[list[float]] = []
        for idx, gmm_params in enumerate(gmm_parameters):
            if gmm_params == [(0, 0, 0)]:
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
        sn.heatmap(df_cm, annot=True)
        plt.title("Estimated harmonic interval distributions' Wasserstein distance distance matrix")
        plt.savefig(os.path.join(self._output_path, "distribution_distance_matrix.jpg"))
        plt.close()
        return samples, distance_matrix.tolist()

    def _cluster_gmms(self, distance_matrix: FloatArray, recordings: list[Recording]) -> list[list[Recording]]:
        clusters: list[list[Recording]] = []

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

        plt.figure(figsize=(fig_width, fig_height))
        plt.xlabel("Wasserstein distance")
        plt.title("Hierarchical Clustering Dendrogram of Estimated GMMs")
        dendrogram(
            linkage_matrix,
            labels=[recording.name for recording in recordings],
            orientation="right",
            leaf_font_size=6,
        )
        plt.savefig(self._output_path.joinpath("hierarchical_clustering_dendrogram_of_estimated_gmms.jpg"))
        plt.close()

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
        _, _, weights_means_variances = self._analyze_gmm_collection(
            cluster_gaussian_mixture_weights_means_variances,
            cluster_name,
            safe_filename_prefix,
        )
        return weights_means_variances

    def generate_analysis_results(self, recordings: list[Recording]) -> AnalysisResults:
        recording_f0s_dict: dict[Recording, tuple[list[float], list[list[float]]]] = {}
        recording_harmonic_intervals_dict: dict[Recording, list[float]] = {}
        recording_gaussian_mixture_weights_means_variances: dict[Recording, list[tuple[float, float, float]]] = {}
        for recording in tqdm(recordings, desc="Analyzing recordings"):
            (
                recording_f0s,
                harmonic_intervals,
                gaussian_mixture_weights_means_variances,
            ) = self._analyze_recording(recording)
            recording_f0s_dict[recording] = (recording_f0s[0].tolist(), [[val for val in row if val > 0] for row in recording_f0s[1].tolist()])
            recording_harmonic_intervals_dict[recording] = harmonic_intervals.tolist()
            recording_gaussian_mixture_weights_means_variances[recording] = gaussian_mixture_weights_means_variances
        gmm_parameters = list(recording_gaussian_mixture_weights_means_variances.values())
        samples: list[list[float]]
        distance_matrix: list[list[float]]
        samples, distance_matrix = self._generate_distance_matrix(gmm_parameters, recordings)
        clusters = self._cluster_gmms(np.array(distance_matrix), recordings)
        cluster_weights_means_variances: dict[str, list[tuple[float, float, float]]] = {}
        cluster_idx_name_dict: dict[int, str] = {}
        for cluster_idx, cluster in enumerate(tqdm(clusters, desc="Analyzing clusters")):
            cluster_name = json.dumps(sorted(recording.name for recording in cluster))
            cluster_idx_name_dict[cluster_idx] = cluster_name
            cluster_weights_means_variances[cluster_name] = self._analyze_cluster(
                cluster,
                recording_gaussian_mixture_weights_means_variances,
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
            recording_gaussian_mixture_weights_means_variances=recording_gaussian_mixture_weights_means_variances,
            distance_matrix=distance_matrix,
            cluster_weights_means_variances=cluster_weights_means_variances,
        )
        open(os.path.join(self._output_path, "analysis_results_serialization.json"), "w").write(results.model_dump_json())
        return results
