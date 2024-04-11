import json
from contextlib import nullcontext
from pathlib import Path

import mir_eval
import pandas as pd
from tqdm import tqdm

from pypolyphonicanalysis.datamodel.features.feature_store import FeatureStore
from pypolyphonicanalysis.datamodel.features.features import Features
from pypolyphonicanalysis.datamodel.tracks.sum_track import load_sum_track
from pypolyphonicanalysis.models.base_multiple_f0_estimation_model import (
    BaseMultipleF0EstimationModel,
)
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import (
    get_estimated_times_and_frequencies_from_salience_map,
    save_f0_trajectories_csv,
    get_random_number_generator,
    save_reconstructed_audio,
    plot_predictions,
)

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)


def get_evaluations_path(settings: Settings) -> Path:
    evaluations_path = Path(settings.data_directory_path).joinpath("evaluations")
    evaluations_path.mkdir(parents=True, exist_ok=True)
    return evaluations_path


class TestSetEvaluator:
    def __init__(self, settings: Settings, max_count: int = -1) -> None:
        self._settings = settings
        self._feature_store = FeatureStore(self._settings)
        training_metadata_path = Path(settings.data_directory_path).joinpath("training_metadata")
        train_test_validation_split = json.load(open(training_metadata_path.joinpath("train_test_validation_split.json"), "r"))
        self._test_sum_tracks = [load_sum_track(sum_track_name, self._settings) for sum_track_name in train_test_validation_split["test"]]
        get_random_number_generator(settings).shuffle(self._test_sum_tracks)
        if max_count != -1:
            self._test_sum_tracks = self._test_sum_tracks[:max_count]

    def evaluate_model(self, model: BaseMultipleF0EstimationModel) -> pd.DataFrame:
        evaluation_path = get_evaluations_path(self._settings)
        all_scores: list[dict[str, float]] = []
        for idx, sum_track in enumerate(tqdm(self._test_sum_tracks)):
            with nullcontext():  # tf.profiler.experimental.Trace("eval", step_num=idx, _r=1):
                ground_truth_salience_map = self._feature_store.generate_or_load_feature_for_sum_track(sum_track, Features.SALIENCE_MAP)
                predicted_salience_map = model.predict_on_file(sum_track.audio_source_path)
                gt_times, gt_freqs = get_estimated_times_and_frequencies_from_salience_map(
                    ground_truth_salience_map,
                    self._settings.threshold,
                    settings=self._settings,
                )
                pred_times, pred_freqs = get_estimated_times_and_frequencies_from_salience_map(
                    predicted_salience_map,
                    self._settings.threshold,
                    settings=self._settings,
                )
                save_reconstructed_audio(gt_times, gt_freqs, f"ground_truth_{sum_track.name}", evaluation_path, settings=self._settings)
                save_reconstructed_audio(pred_times, pred_freqs, f"prediction_{sum_track.name}", evaluation_path, settings=self._settings)
                plot_predictions(gt_times, gt_freqs, f"ground_truth_{sum_track.name}", evaluation_path, self._settings.default_figsize)
                plot_predictions(pred_times, pred_freqs, f"prediction_{sum_track.name}", evaluation_path, self._settings.default_figsize)
                save_f0_trajectories_csv(
                    evaluation_path.joinpath(f"ground_truth_{sum_track.name}.csv"),
                    gt_times.tolist(),
                    gt_freqs,
                )
                save_f0_trajectories_csv(
                    evaluation_path.joinpath(f"prediction_{sum_track.name}.csv"),
                    pred_times.tolist(),
                    pred_freqs,
                )
                scores = mir_eval.multipitch.evaluate(gt_times, gt_freqs, pred_times, pred_freqs)
                scores["track"] = sum_track.name
                all_scores.append(scores)
                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    print("\n")
                    print(pd.DataFrame(all_scores).describe())
        return pd.DataFrame(all_scores)
