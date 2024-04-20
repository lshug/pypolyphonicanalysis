import numpy as np

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature
from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray, get_random_state
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store


class FeatureStream:
    def __init__(self, sum_track: SumTrack, input_features: list[InputFeature], label_features: list[LabelFeature], settings: Settings) -> None:
        self._sum_track = sum_track
        self._settings = settings
        self._input_features = input_features
        self._label_features = label_features
        self._batch_size = settings.training_batch_size
        self._number_of_slices = settings.training_input_number_of_slices
        self._feature_store = get_feature_store(settings)
        self._rng = get_random_state(settings)
        self._input_feature_arrs = [self._feature_store.generate_or_load_feature_for_sum_track(self._sum_track, feature) for feature in self._input_features]
        self._label_feature_arrs = [self._feature_store.generate_or_load_feature_for_sum_track(self._sum_track, feature) for feature in self._label_features]
        self._n_t = self._input_feature_arrs[0].shape[-1]

    def __iter__(self) -> "FeatureStream":
        return self

    def __next__(self) -> tuple[list[FloatArray], list[FloatArray]]:
        input_slices_list = []
        label_slices_batch = []
        for idx in range(self._batch_size):
            slice_idx = self._rng.randint(0, self._n_t - self._number_of_slices + 1)
            input_slices = [arr[..., slice_idx : slice_idx + self._number_of_slices] for arr in self._input_feature_arrs]
            label_slices = [arr[..., slice_idx : slice_idx + self._number_of_slices] for arr in self._label_feature_arrs]
            input_slices_list.append(input_slices)
            label_slices_batch.append(label_slices)
        input_batches = []
        for idx in range(len(self._input_feature_arrs)):
            input_batches.append(np.stack([slice[idx] for slice in input_slices_list]))
        label_batches = []
        for idx in range(len(self._label_feature_arrs)):
            label_batches.append(np.stack([slice[idx] for slice in label_slices_batch]))
        return input_batches, label_batches

    def __hash__(self) -> int:
        return hash(f"feature_stream_{self._sum_track.name}")
