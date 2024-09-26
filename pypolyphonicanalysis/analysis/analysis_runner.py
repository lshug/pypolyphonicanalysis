import enum
import itertools
import json
import logging
import os
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import Sequence, cast

import matplotlib
import pandas
import pandas as pd
import seaborn as sn
import librosa
import numpy as np
import umap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pydantic import BaseModel
from scipy.cluster.hierarchy import dendrogram
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance
from sklearn.base import TransformerMixin
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
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

matplotlib.rcParams["figure.dpi"] = 600
matplotlib.rcParams["axes.titlesize"] = "large"
matplotlib.rcParams["axes.labelsize"] = "large"


def imscatter(x: float, y: float, image: FloatArray, ax: Axes, zoom: float = 1) -> None:
    im = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
    ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def get_activation_cache_path(settings: Settings) -> Path:
    path = Path(settings.data_directory_path).joinpath("activation_cache")
    check_output_path(path)
    return path


def reconstruct_gmm_from_parameters(gmm_parameters: list[tuple[float, float, float]], settings: Settings) -> GaussianMixture:
    if len(gmm_parameters) == 0:
        return reconstruct_gmm_from_parameters([(0, 1, 1)], settings)
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


def save_gmm_parameters(parameters: list[tuple[float, float, float]], filename_prefix: str, output_path: Path) -> None:
    check_output_path(output_path)
    weights, means, vars = zip(*parameters)
    std = [x**0.5 for x in vars]
    df = pandas.DataFrame({"Weight": weights, "Mean": means, "Std.": std})
    df.to_csv(output_path.joinpath(f"{filename_prefix}_estimated_distribution_parameters.csv"), index_label="Component")

    distribution_table_str = "Component\tWeight\tMean\tStd.\n"
    for idx in range(len(parameters)):
        distribution_table_str += f"{idx + 1}\t\t{parameters[idx][0]:.4f}\t{parameters[idx][1]:.2f}\t{parameters[idx][2] ** 0.5:.4f}\n"
    open(output_path.joinpath(f"{filename_prefix}_estimated_distribution_parameters.txt"), "w").write(distribution_table_str)


def predict_label_with_embedding(name: str, embedding: FloatArray, y_true_labels: list[str], output_path: Path) -> None:
    class_idx_dict = {label: idx for idx, label in enumerate(sorted(set(y_true_labels)))}
    classes = list(class_idx_dict.keys())
    y_true = np.array([class_idx_dict[label] for label in y_true_labels])
    X_train, X_test, y_train, y_test = train_test_split(embedding, y_true, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f"{name} embedding classification")
    report = classification_report(y_test, y_pred, labels=np.arange(0, len(classes), 1), target_names=classes)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=np.arange(0, len(classes), 1), display_labels=classes, ax=ax, xticks_rotation="vertical").plot()
    fig.savefig(output_path.joinpath(f"{name}_embedding_classification_confusion_matrix.png"))
    open(output_path.joinpath(f"{name}_embedding_classification_report.txt"), "w").write(report)


def predict_label_with_cluster(name: str, recordings: list[Recording], recording_cluster_dict: dict[Recording, int], y_true_labels: list[str], output_path: Path) -> None:
    class_idx_dict = {label: idx for idx, label in enumerate(sorted(set(y_true_labels)))}
    classes = list(class_idx_dict.keys())
    y_true = [class_idx_dict[label] for label in y_true_labels]
    cluster_class_idx_counter: dict[int, Counter[int]] = defaultdict(Counter)
    for idx, recording in enumerate(recordings):
        cluster_idx = recording_cluster_dict[recording]
        recording_label_idx = class_idx_dict[y_true_labels[idx]]
        cluster_class_idx_counter[cluster_idx][recording_label_idx] += 1
    cluster_predicted_class = {cluster: cluster_class_idx_counter[cluster].most_common(1)[0][0] for cluster in set(recording_cluster_dict.values())}
    y_pred = [cluster_predicted_class[recording_cluster_dict[recording]] for recording in recordings]
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f"{name} cluster classification")
    report = classification_report(y_true, y_pred, labels=np.arange(0, len(classes), 1), target_names=classes)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=np.arange(0, len(classes), 1), display_labels=classes, ax=ax, xticks_rotation="vertical").plot()
    fig.savefig(output_path.joinpath(f"{name}_cluster_classification_confusion_matrix.png"))
    open(output_path.joinpath(f"{name}_cluster_classification_report.txt"), "w").write(report)


def run_classification_experiments(name: str, clusters: dict[int, list[Recording]], recordings: list[Recording], embedding: FloatArray, output_path: Path) -> None:
    recording_cluster_dict: dict[Recording, int] = {}
    for cluster_idx, cluster in clusters.items():
        for recording in cluster:
            recording_cluster_dict[recording] = cluster_idx
    regions = [recording.recording_region for recording in recordings if recording.recording_region is not None]
    assert len(regions) == len(recordings)
    origin_tapes = [recording.name.split("-")[0] for recording in recordings]
    for task_name, y_true_labels in [("regions", regions), ("origin_tapes", origin_tapes)]:
        predict_label_with_cluster(f"{name}_{task_name}", recordings, recording_cluster_dict, y_true_labels, output_path)
        predict_label_with_embedding(f"{name}_{task_name}", embedding, y_true_labels, output_path)
    pass


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


class EmbeddingMethod(enum.Enum):
    PCA = 0
    UMAP = 1


class AnalysisResults(BaseModel):
    recording_name_f0s_dict: dict[str, tuple[list[float], list[list[float]]]]
    recording_name_harmonic_intervals_dict: dict[str, list[float]]
    recording_name_harmonic_interval_gaussian_mixture_parameters: dict[str, list[tuple[float, float, float]]]


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
        gmm_parameters: list[tuple[float, float, float]],
        ground_truth_gmm_params: list[tuple[float, float, float]] | None,
        name: str,
    ) -> None:
        if len(harmonic_intervals) == 0:
            return
        recording_output_path = self._output_path.joinpath(name)
        check_output_path(recording_output_path)
        gmm = reconstruct_gmm_from_parameters(gmm_parameters, self._settings)
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

        plt.xlim(left=0, right=1200)
        plt.xticks(np.arange(0, 1200, 100))
        plt.grid(True, "major", "x")

        plt.legend()
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.title(f"Harmonic interval distribution of {name}")
        plt.savefig(recording_output_path.joinpath(f"{name}_harmonic_interval_distribution.jpg"))

        plt.close()

        parameters = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )
        save_gmm_parameters(parameters, name, recording_output_path)
        self._generate_gmm_derived_scale_example_file([mean for _, mean, _ in parameters], name, recording_output_path)

    def _export_cluster_harmonic_interval_distribution_plots_and_files(
        self,
        average_gmm_parameters: list[tuple[float, float, float]],
        individual_gmm_params: list[list[tuple[float, float, float]]],
        name: str,
        safe_filename_prefix: str,
        output_parent_directory: Path,
    ) -> None:
        cluster_output_path = output_parent_directory.joinpath(safe_filename_prefix)
        check_output_path(cluster_output_path)
        save_gmm_parameters(average_gmm_parameters, safe_filename_prefix, cluster_output_path)
        x = np.linspace(1, 1300, 1000).reshape(-1, 1)
        plt.figure(figsize=self._settings.default_figsize)
        for single_gmm in [reconstruct_gmm_from_parameters(params, self._settings) for params in individual_gmm_params]:
            plt.plot(x, np.exp(single_gmm.score_samples(x.reshape(-1, 1))), color="black")
        gmm = reconstruct_gmm_from_parameters(average_gmm_parameters, self._settings)
        self._generate_gmm_derived_scale_example_file([mean for mean in gmm.means_.reshape(-1)], safe_filename_prefix, cluster_output_path)
        plt.plot(x, np.exp(gmm.score_samples(x)), color="red")
        plt.xlim(left=0, right=1200)
        plt.xticks(np.arange(0, 1200, 100))
        plt.grid(True, "major", "x")
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.suptitle("Harmonic interval distribution (average of estimated Gaussian Mixtures)")
        plt.title(
            "\n".join(
                wrap(
                    f"{safe_filename_prefix} ({len(individual_gmm_params)}): {name if len(name) < 250 else f'{name[:250]}...]'}",
                    120,
                )
            ),
        )
        plt.savefig(cluster_output_path.joinpath(f"{safe_filename_prefix}_harmonic_interval_distribution.jpg"))
        plt.close()

    def _export_recording_metadata(self, recording: Recording) -> None:
        recording_output_path = self._output_path.joinpath(recording.name)
        check_output_path(recording_output_path)
        with open(recording_output_path.joinpath("metadata.json"), "w", encoding="utf8") as f:
            f.write(recording.model_dump_json(indent=4))

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
        parameters = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )

        return parameters, kde

    def _generate_gmm_derived_scale_example_file(self, means: list[float], name: str, output_path: Path) -> None:
        check_output_path(output_path)
        sr = self._settings.sr
        arr = np.array([])
        for mean in sorted(means):
            freq = librosa.note_to_hz("C6") * 2 ** (mean / 1200)
            t = np.linspace(0.0, 0.5, int(sr * 0.5))
            y = np.sin(freq * t)
            arr = np.concatenate((arr, y))
        wavfile.write(output_path.joinpath(f"{name}_derived_scale.wav"), sr, arr)

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
        parameters, _ = self._model_harmonic_interval_distribution(harmonic_intervals, name)
        if self._settings.save_ground_truth_track_data:
            for track in unsaved_tracks:
                track.save()
        if len(parameters) == 0:
            return None
        return parameters

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
        tuple[list[float], list[list[float]]],
        list[float],
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
        gaussian_mixture_parameters, kde = self._model_harmonic_interval_distribution(harmonic_intervals, recording.name)
        if len(gaussian_mixture_parameters) != 0:
            self._export_recording_harmonic_interval_distribution_plots_and_files(
                harmonic_intervals, kde, gaussian_mixture_parameters, self._get_harmonic_interval_ground_truth_gmm_parameters(recording), recording.name
            )
        self._export_recording_metadata(recording)

        return (times.tolist(), freqs.tolist()), harmonic_intervals.tolist(), gaussian_mixture_parameters

    def _get_average_gmm_parameters(
        self,
        gmm_parameters: list[list[tuple[float, float, float]]],
    ) -> list[tuple[float, float, float]]:
        x = np.linspace(1, 1300, 1000).reshape(-1, 1)
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

        parameters = sorted(
            list(
                zip(
                    gmm.weights_.reshape(-1).tolist(),
                    gmm.means_.reshape(-1).tolist(),
                    gmm.covariances_.reshape(-1).tolist(),
                )
            ),
            key=lambda weight_mean_and_var: weight_mean_and_var[1],
        )

        return parameters

    def _generate_distance_matrix(
        self,
        gmm_parameters: list[list[tuple[float, float, float]]],
        recordings: dict[str, Recording],
    ) -> list[list[float]]:
        recordings_names_list = list(recordings.keys())
        samples: list[list[float]] = []
        for idx, gmm_params in enumerate(gmm_parameters):
            if gmm_params == []:
                logger.info(f"Empty GMM found for {recordings_names_list[idx]}, skipping sampling.")
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
            index=[recording_name for recording_name in recordings],
            columns=[recording_name for recording_name in recordings],
        )
        df_cm.to_csv(os.path.join(self._output_path, "distribution_distance_matrix.csv"))
        plt.figure(figsize=(1.7 * len(recordings) / 2, 1.7 * len(recordings) / 2), dpi=100)
        sn.heatmap(df_cm, annot=True, fmt=".2f")
        plt.title("Estimated interval distributions' Wasserstein distance matrix")
        plt.savefig(os.path.join(self._output_path, "distribution_distance_matrix.jpg"))

        plt.close()
        distance_matrix_list: list[list[float]] = distance_matrix.tolist()
        return distance_matrix_list

    def _cluster_gmms_using_agglomerative_clustering(
        self, distance_matrix: FloatArray, recordings: dict[str, Recording], recording_gmm_param_dict: dict[str, list[tuple[float, float, float]]]
    ) -> list[list[Recording]]:
        clusters: list[list[Recording]] = []
        recordings_list = list(recordings.values())

        clustering = AgglomerativeClustering(
            n_clusters=len(recordings),
            metric="precomputed",
            linkage="average",
            compute_distances=True,
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

        linkage_index_to_recording_name_set_dict: dict[int, set[str]] = {}
        for linkage_idx, child in enumerate(clustering.children_):
            clusters.append([recordings_list[idx] for idx in _get_cluster_recording_idxs_from_child(child, clustering.children_, len(clustering.labels_))])
            linkage_index_to_recording_name_set_dict[linkage_idx] = set(recording.name for recording in clusters[-1])

        linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)

        fig_width = 2 * max(len(recording_name) for recording_name in recordings.keys())
        fig_height = int(distance_matrix.shape[0] * 1.5)

        axs: list[Axes]
        fig, axs = plt.subplots(1, 2, width_ratios=[0.1, 0.9], figsize=(fig_width, fig_height), dpi=100)
        distribution_ax = axs[0]
        dendrogram_ax = axs[1]

        dendrogram_ax.set_xlabel("Wasserstein distance")
        dendrogram_ax.set_title("Hierarchical Clustering Dendrogram of Estimated GMMs")

        link_order: list[int] = []

        def _link_order_tracker(link_index: int) -> str:
            link_order.append(link_index - len(recordings))
            return "C0"

        dendrogram_data = dendrogram(
            linkage_matrix,
            ax=dendrogram_ax,
            labels=[recording_name for recording_name in recordings.keys()],
            orientation="right",
            leaf_font_size=6,
            color_threshold=float(0),
            link_color_func=_link_order_tracker,
        )

        tick_labels = []
        tick_gmms = []
        for label_point in dendrogram_ax.get_ymajorticklabels():
            label_recording = recordings[label_point.get_text()]
            tick_gmms.append(reconstruct_gmm_from_parameters(recording_gmm_param_dict[label_point.get_text()], self._settings))
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

        icoord = dendrogram_data["icoord"]
        dcoord = dendrogram_data["dcoord"]
        for idx, (ys, xs) in enumerate(zip(icoord, dcoord)):
            x = xs[1]
            y = (ys[1] + ys[2]) / 2
            point = (x, y)
            dendrogram_ax.annotate(f"cluster_{link_order[idx]}", point)
            idx += 1

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
        fig.savefig(self._output_path.joinpath("hierarchical_clustering_dendrogram_of_estimated_gmms.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        return clusters

    def _analyze_cluster(
        self,
        cluster: list[Recording],
        recording_gaussian_mixture_parameters: dict[str, list[tuple[float, float, float]]],
        cluster_name: str,
        cluster_idx: int,
        output_parent_directory: Path,
    ) -> list[tuple[float, float, float]]:
        logger.info(f"Analyzing cluster {cluster_name}")
        safe_filename_prefix = f"cluster_{cluster_idx}"
        cluster_gaussian_mixture_parameters = [recording_gaussian_mixture_parameters[recording.name] for recording in cluster]
        parameters = self._get_average_gmm_parameters(
            cluster_gaussian_mixture_parameters,
        )
        self._export_cluster_harmonic_interval_distribution_plots_and_files(
            parameters, cluster_gaussian_mixture_parameters, cluster_name, safe_filename_prefix, output_parent_directory
        )
        return parameters

    def _analyze_clusters(
        self, clusters: dict[int, list[Recording]], recording_harmonic_interval_gaussian_mixture_parameters: dict[str, list[tuple[float, float, float]]], clustering_directory: Path
    ) -> dict[str, list[tuple[float, float, float]]]:
        harmonic_interval_distribution_cluster_gaussian_mixture_parameters: dict[str, list[tuple[float, float, float]]] = {}
        recording_name_cluster_idxs_dict: dict[str, list[int]] = defaultdict(list)
        cluster_idx_name_dict: dict[int, dict[str, str | int | None]] = {}
        cluster_idxs: list[int] = []
        num_recordings: list[int] = []
        modal_regions: list[str | None] = []
        modal_tape_idxs: list[str | None] = []
        synoptic_scales: list[str] = []
        for cluster_idx, cluster in tqdm(sorted(clusters.items(), key=lambda c: len(c[1]), reverse=True), desc="Analyzing clusters"):
            cluster_name = json.dumps(sorted(recording.name for recording in cluster))
            for recording in cluster:
                recording_name_cluster_idxs_dict[recording.name].append(cluster_idx)
            harmonic_interval_distribution_cluster_gaussian_mixture_parameters[cluster_name] = self._analyze_cluster(
                cluster, recording_harmonic_interval_gaussian_mixture_parameters, cluster_name, cluster_idx, clustering_directory
            )
            modal_region = Counter([recording.recording_region for recording in cluster]).most_common(1)[0][0]
            modal_tape_index = Counter([recording.name.split("-")[0] for recording in cluster]).most_common(1)[0][0]
            synoptic_scale = ", ".join([str(params[1]) for params in harmonic_interval_distribution_cluster_gaussian_mixture_parameters[cluster_name]])
            cluster_idx_name_dict[cluster_idx] = {
                "num_recordings": len(cluster),
                "cluster_name": cluster_name,
                "modal_region": modal_region,
                "modal_tape_index": modal_tape_index,
                "synoptic_scale": synoptic_scale,
            }
            cluster_idxs.append(cluster_idx)
            num_recordings.append(len(cluster))
            modal_regions.append(modal_region)
            modal_tape_idxs.append(modal_tape_index)
            synoptic_scales.append(synoptic_scale)
        df = pd.DataFrame(
            {
                "Cluster idx.": cluster_idxs,
                "Number of recordings": num_recordings,
                "Modal region": modal_regions,
                "Modal tape index": modal_tape_idxs,
                "Synoptic scale": synoptic_scales,
            }
        )
        df.to_csv(clustering_directory.joinpath("clusters.csv"))

        json.dump(
            cluster_idx_name_dict,
            open(clustering_directory.joinpath("cluster_idx_names.json"), "w"),
            indent=4,
        )
        json.dump(
            recording_name_cluster_idxs_dict,
            open(clustering_directory.joinpath("recording_name_cluster_idxs.json"), "w"),
            indent=2,
        )
        return harmonic_interval_distribution_cluster_gaussian_mixture_parameters

    def perform_hierarchical_clustering(
        self,
        recordings: dict[str, Recording],
        recording_harmonic_interval_gaussian_mixture_parameters: dict[str, list[tuple[float, float, float]]],
    ) -> tuple[list[list[float]], dict[str, list[tuple[float, float, float]]]]:
        clustering_directory = self._output_path.joinpath("hierarchical_clustering")
        check_output_path(clustering_directory)
        recording_harmonic_interval_gaussian_mixture_parameters = {k: v for k, v in recording_harmonic_interval_gaussian_mixture_parameters.items() if k in recordings}
        harmonic_gmm_parameters = list(recording_harmonic_interval_gaussian_mixture_parameters.values())
        harmonic_interval_distribution_distance_matrix = self._generate_distance_matrix(harmonic_gmm_parameters, recordings)
        clusters = dict(
            enumerate(
                self._cluster_gmms_using_agglomerative_clustering(
                    np.array(harmonic_interval_distribution_distance_matrix), recordings, recording_harmonic_interval_gaussian_mixture_parameters
                )
                if len(recordings) > 1
                else []
            )
        )
        harmonic_interval_distribution_cluster_gaussian_mixture_parameters = self._analyze_clusters(
            clusters, recording_harmonic_interval_gaussian_mixture_parameters, clustering_directory
        )
        return harmonic_interval_distribution_distance_matrix, harmonic_interval_distribution_cluster_gaussian_mixture_parameters

    def perform_embedding_clustering(
        self,
        recordings: dict[str, Recording],
        recording_harmonic_interval_gaussian_mixture_parameters: dict[str, list[tuple[float, float, float]]],
        embedding_method: EmbeddingMethod = EmbeddingMethod.PCA,
        classification_experiments: bool = True,
        n_components: int = 2,
        umap_n_neighbors: int = 10,
        umap_min_dist: float = 0.1,
        pdf_vector_number_of_bins: int = 200,
        pdf_vector_cents_lower_bound: float = 1,
        pdf_vector_cents_upper_bound: float = 1200,
        pdf_vector_multiplier: float = 100,
    ) -> tuple[dict[int, list[Recording]], list[Recording], FloatArray, dict[str, list[tuple[float, float, float]]]]:
        embedding_dir_name = f"embedding_clustering_components_{n_components}_method_{embedding_method.name}"
        if embedding_method == EmbeddingMethod.UMAP:
            embedding_dir_name += f"_n_neighbors_{umap_n_neighbors}_min_distance_{umap_min_dist}"
        clustering_directory = self._output_path.joinpath(embedding_dir_name)
        check_output_path(clustering_directory)
        recording_harmonic_interval_gaussian_mixture_parameters = {k: v for k, v in recording_harmonic_interval_gaussian_mixture_parameters.items() if k in recordings}
        skipped_recordings: set[Recording] = set()
        recording_vector_dict: dict[Recording, FloatArray] = {}
        recording_image_dict: dict[Recording, FloatArray] = {}
        for recording_name, harmonic_interval_gaussian_mixture_parameters in tqdm(
            recording_harmonic_interval_gaussian_mixture_parameters.items(), desc="Retrieving density vectors from recording GMMs"
        ):
            recording = recordings[recording_name]
            if len(harmonic_interval_gaussian_mixture_parameters) < 2:
                skipped_recordings.add(recording)
                continue
            gmm = reconstruct_gmm_from_parameters(harmonic_interval_gaussian_mixture_parameters, self._settings)
            bins = np.linspace(pdf_vector_cents_lower_bound, pdf_vector_cents_upper_bound, pdf_vector_number_of_bins)
            scaled_pdf_vector = pdf_vector_multiplier * np.exp(gmm.score_samples(bins.reshape(-1, 1)))
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot(bins, scaled_pdf_vector)
            ax.margins(0)
            canvas: FigureCanvasAgg = cast(FigureCanvasAgg, fig.canvas)
            canvas.draw()
            image_from_plot = np.asarray(canvas.buffer_rgba())[:, :, :3]
            recording_image_dict[recording] = image_from_plot
            recording_vector_dict[recording] = scaled_pdf_vector
        recordings_list = list(recording_vector_dict.keys())
        recording_pdf_vectors_array = np.array(list(recording_vector_dict.values()))
        reducer: TransformerMixin
        match embedding_method:
            case EmbeddingMethod.UMAP:
                reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, n_components=n_components)
            case EmbeddingMethod.PCA | _:
                reducer = PCA(n_components=n_components)
        recording_pdf_vectors_array = StandardScaler().fit_transform(recording_pdf_vectors_array)

        embedding = reducer.fit_transform(recording_pdf_vectors_array)
        hdb = HDBSCAN()
        hdb.fit(embedding)
        colors = [plt.get_cmap("Spectral")(each) for each in np.linspace(0, 1, len(set(hdb.labels_)))]

        recording_list = list(recording_image_dict.keys())
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim(np.min(embedding[:, 0] - 10), np.max(embedding[:, 0]) + 10)
        ax.set_ylim(np.min(embedding[:, 1]) - 10, np.max(embedding[:, 1]) + 10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.suptitle("Embedding clustering", fontsize=30)
        ax.set_title(f"Method={embedding_method}, n_components={n_components} \nFound {len(set(hdb.labels_))} clusters", fontsize=20)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))

        clusters: dict[int, list[Recording]] = defaultdict(list)
        for idx, (x, y) in enumerate(zip(embedding[:, 0], embedding[:, 1])):
            recording = recording_list[idx]
            recording_cluster = int(hdb.labels_[idx])
            clusters[recording_cluster].append(recording)
            imscatter(x, y, recording_image_dict[recording], ax, zoom=0.01)
            lines = [recording.name]
            if recording.performers is not None:
                lines.append(recording.performers)
            if recording.recording_site is not None:
                lines.append(recording.recording_site)
            if recording.recording_date is not None:
                lines.append(recording.recording_date)
            recording_label = "\n".join(lines)
            ax.annotate(recording_label, (x, y + 0.01), fontsize=1, color=colors[recording_cluster])

        fig.savefig(clustering_directory.joinpath("embedding_clustering_results.png"))
        harmonic_interval_distribution_cluster_gaussian_mixture_parameters = self._analyze_clusters(
            clusters, recording_harmonic_interval_gaussian_mixture_parameters, clustering_directory
        )

        if classification_experiments:
            run_classification_experiments(embedding_dir_name, clusters, recordings_list, embedding, clustering_directory)

        return clusters, recordings_list, embedding, harmonic_interval_distribution_cluster_gaussian_mixture_parameters

    def generate_analysis_results(self, recordings: dict[str, Recording]) -> AnalysisResults:
        recording_f0s_dict: dict[str, tuple[list[float], list[list[float]]]] = {}
        recording_harmonic_intervals_dict: dict[str, list[float]] = {}
        recording_harmonic_interval_gaussian_mixture_parameters: dict[str, list[tuple[float, float, float]]] = {}
        for recording_name, recording in tqdm(list(recordings.items()), desc="Analyzing recordings"):
            (
                recording_f0s,
                harmonic_intervals,
                gaussian_mixture_parameters,
            ) = self._analyze_recording(recording)
            recording_f0s_dict[recording_name] = (recording_f0s[0], [[val for val in row if val > 0] for row in recording_f0s[1]])
            recording_harmonic_intervals_dict[recording_name] = harmonic_intervals
            recording_harmonic_interval_gaussian_mixture_parameters[recording_name] = gaussian_mixture_parameters

        results = AnalysisResults(
            recording_name_f0s_dict=recording_f0s_dict,
            recording_name_harmonic_intervals_dict=recording_harmonic_intervals_dict,
            recording_name_harmonic_interval_gaussian_mixture_parameters=recording_harmonic_interval_gaussian_mixture_parameters,
        )
        return results
