import hashlib
import os
from functools import cache
from pathlib import Path

import numpy as np

from pypolyphonicanalysis.datamodel.features.feature_generators import (
    FeatureGenerator,
    HCQTMagPhaseDiffGenerator,
    SalienceMapGenerator,
    InputFeatureGenerator,
)
from pypolyphonicanalysis.datamodel.features.features import Features, InputFeature
from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray


def get_features_path(settings: Settings) -> Path:
    return Path(settings.data_directory_path).joinpath("features")


def feature_is_generated_for_sum_track(sum_track: SumTrack, feature: Features, settings: Settings) -> bool:
    sum_track_feature_store_path = get_features_path(settings).joinpath(sum_track.name)
    return sum_track_feature_store_path.joinpath(f"{feature.name}.npy").is_file() and sum_track_feature_store_path.joinpath(f"{feature.name}.saved").is_file()


def feature_is_generated_for_file(file: Path, feature: Features, settings: Settings) -> bool:
    if not settings.save_prediction_file_features:
        return False
    file_feature_store_path = get_features_path(settings).joinpath(f"file_{hashlib.file_digest(open(file, 'rb'), 'md5').hexdigest()}")
    return file_feature_store_path.joinpath(f"{feature.name}.npy").is_file() and file_feature_store_path.joinpath(f"{feature.name}.saved").is_file()


def load_feature_for_sum_track(sum_track: SumTrack, feature: Features, settings: Settings) -> FloatArray:
    sum_track_feature_store_path = get_features_path(settings).joinpath(sum_track.name)
    loaded_array = np.load(sum_track_feature_store_path.joinpath(f"{feature.name}.npy"))
    assert isinstance(loaded_array, np.ndarray)
    return loaded_array


def load_feature_for_file(file: Path, feature: Features, settings: Settings) -> FloatArray:
    file_feature_store_path = get_features_path(settings).joinpath(f"file_{hashlib.file_digest(open(file, 'rb'), 'md5').hexdigest()}")
    loaded_array = np.load(file_feature_store_path.joinpath(f"{feature.name}.npy"))
    assert isinstance(loaded_array, np.ndarray)
    return loaded_array


class FeatureStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._feature_generator_and_index_dict: dict[Features, tuple[FeatureGenerator, int]] = self._get_feature_generator_and_index_dict()
        self._validate_feature_generator_and_index_dict(self._feature_generator_and_index_dict)

    def _validate_feature_generator_and_index_dict(
        self,
        feature_generator_and_index_dict: dict[Features, tuple[FeatureGenerator, int]],
    ) -> None:
        for feature in Features:
            if feature not in feature_generator_and_index_dict:
                raise NotImplementedError(f"Missing generator implementation for feature {feature}")
            generator, index = feature_generator_and_index_dict[feature]
            if index >= generator.number_of_features:
                raise ValueError(f"Feature index {index} is out of bounds for generator {generator} with {generator.number_of_features} features.")

    def _get_feature_generator_and_index_dict(
        self,
    ) -> dict[Features, tuple[FeatureGenerator, int]]:
        hcqt_mag_and_phase_diff_generator = HCQTMagPhaseDiffGenerator(self._settings)
        return {
            Features.HCQT_MAG: (hcqt_mag_and_phase_diff_generator, 0),
            Features.HCQT_PHASE_DIFF: (hcqt_mag_and_phase_diff_generator, 1),
            Features.SALIENCE_MAP: (SalienceMapGenerator(self._settings), 0),
        }

    def generate_or_load_feature_for_sum_track(self, sum_track: SumTrack, feature: Features) -> FloatArray:
        if feature_is_generated_for_sum_track(sum_track, feature, self._settings):
            return load_feature_for_sum_track(sum_track, feature, self._settings)
        return self._generate_feature_for_sum_track(sum_track, feature)

    def generate_all_features_for_sum_track(self, sum_track: SumTrack) -> None:
        for feature in Features:
            self.generate_or_load_feature_for_sum_track(sum_track, feature)

    def generate_or_load_feature_for_file(self, file: Path, feature: InputFeature) -> FloatArray:
        if feature_is_generated_for_file(file, feature, self._settings):
            return load_feature_for_file(file, feature, self._settings)
        return self._generate_feature_for_file(file, feature)

    def _generate_feature_for_file(self, file: Path, input_feature: InputFeature) -> FloatArray:
        generator, index = self._feature_generator_and_index_dict[input_feature]
        assert isinstance(generator, InputFeatureGenerator)
        generated_features = generator.generate_features_for_file(file)
        if self._settings.save_prediction_file_features:
            for feature, (
                feature_generator,
                feature_index,
            ) in self._feature_generator_and_index_dict.items():
                if feature_generator is generator:
                    self._save_array_for_file(generated_features[feature_index], file, feature)
        return generated_features[index]

    def _generate_feature_for_sum_track(self, sum_track: SumTrack, feature: Features) -> FloatArray:
        generator, index = self._feature_generator_and_index_dict[feature]
        generated_features = generator.generate_features_for_sum_track(sum_track)
        for feature, (
            feature_generator,
            feature_index,
        ) in self._feature_generator_and_index_dict.items():
            if feature_generator is generator and self._settings.save_training_features:
                self._save_array_for_sum_track(generated_features[feature_index], sum_track, feature)
        return generated_features[index]

    def _save_array_for_sum_track(self, array: FloatArray, sum_track: SumTrack, feature: Features) -> None:
        features_path = get_features_path(self._settings)
        features_path.mkdir(parents=True, exist_ok=True)
        sum_track_features_path = features_path.joinpath(sum_track.name)
        sum_track_features_path.mkdir(parents=True, exist_ok=True)
        np.save(sum_track_features_path.joinpath(f"{feature.name}.npy"), array)
        with open(sum_track_features_path.joinpath(f"{feature.name}.saved"), "a"):
            os.utime(sum_track_features_path.joinpath(f"{feature.name}.saved"), None)

    def _save_array_for_file(self, array: FloatArray, file: Path, feature: Features) -> None:
        features_path = get_features_path(self._settings)
        features_path.mkdir(parents=True, exist_ok=True)
        file_features_path = features_path.joinpath(f"file_{hashlib.file_digest(open(file, 'rb'), 'md5')}")
        file_features_path.mkdir(parents=True, exist_ok=True)
        np.save(file_features_path.joinpath(f"{feature.name}.npy"), array)
        with open(file_features_path.joinpath(f"{feature.name}.saved"), "a"):
            os.utime(file_features_path.joinpath(f"{feature.name}.saved"), None)


@cache
def get_feature_store(settings: Settings) -> FeatureStore:
    return FeatureStore(settings)
