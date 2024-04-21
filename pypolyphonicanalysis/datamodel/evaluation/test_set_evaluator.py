from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd
from tqdm import tqdm

from pypolyphonicanalysis.datamodel.tracks.splits import SumTrackSplitType
from pypolyphonicanalysis.datamodel.tracks.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.features.features import Features
from pypolyphonicanalysis.models.multiple_f0_estimation.base_multiple_f0_estimation_model import (
    BaseMultipleF0EstimationModel,
)
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import (
    get_estimated_times_and_frequencies_from_salience_map,
    save_f0_trajectories_csv,
    get_random_number_generator,
    save_reconstructed_audio,
    plot_predictions,
    check_output_path,
)
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)


def get_evaluations_path(settings: Settings) -> Path:
    evaluations_path = Path(settings.data_directory_path).joinpath("evaluations")
    check_output_path(evaluations_path)
    return evaluations_path


class TestSetEvaluator:
    def __init__(self, sum_track_provider: SumTrackProvider, settings: Settings, max_count: int = -1) -> None:
        self._settings = settings
        self._feature_store = get_feature_store(self._settings)
        self._test_sum_tracks = [sum_track for sum_track, split in sum_track_provider.get_sum_tracks() if split is SumTrackSplitType.TEST]
        get_random_number_generator(settings).shuffle(self._test_sum_tracks)
        if max_count != -1:
            self._test_sum_tracks = self._test_sum_tracks[:max_count]

    def evaluate_model(self, model: BaseMultipleF0EstimationModel) -> pd.DataFrame:
        if Features.SALIENCE_MAP not in model.model_label_features:
            raise ValueError("TestSetEvaluator can only evaluate models that output SALIENCE_MAP")
        evaluation_path = get_evaluations_path(self._settings).joinpath(model.name)
        check_output_path(evaluation_path)
        all_scores: list[dict[str, float]] = []
        for idx, sum_track in enumerate(tqdm(self._test_sum_tracks)):
            ground_truth_salience_map = self._feature_store.generate_or_load_feature_for_sum_track(sum_track, Features.SALIENCE_MAP)
            predicted_salience_map = model.predict_on_sum_track(sum_track)[Features.SALIENCE_MAP]
            gt_times, gt_freqs = get_estimated_times_and_frequencies_from_salience_map(
                ground_truth_salience_map,
                settings=self._settings,
            )
            pred_times, pred_freqs = get_estimated_times_and_frequencies_from_salience_map(
                predicted_salience_map,
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
            gt_freqs_list = [np.array([elem for elem in row if elem != 0]) for row in gt_freqs]
            pred_freqs_list = [np.array([elem for elem in row if elem != 0]) for row in pred_freqs]
            scores = mir_eval.multipitch.evaluate(gt_times, gt_freqs_list, pred_times, pred_freqs_list)
            scores["track"] = sum_track.name
            all_scores.append(scores)
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print("\n")
                print(pd.DataFrame(all_scores).describe())
        return pd.DataFrame(all_scores)
