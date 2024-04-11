from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pypolyphonicanalysis.datamodel.features.feature_store import (
    FeatureStore,
)
from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray


def get_models_path(settings: Settings) -> Path:
    models_path = Path(settings.data_directory_path).joinpath("models")
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


class BaseMultipleF0EstimationModel:
    def __init__(self, settings: Settings, inference_mode: bool = False) -> None:
        self._settings = settings
        self._feature_store = FeatureStore(self._settings)
        self._inference_mode = inference_mode
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cuda:0")

    @property
    @abstractmethod
    def model_input_features(self) -> list[InputFeature]:
        pass

    @property
    @abstractmethod
    def model_label_features(self) -> list[LabelFeature]:
        pass

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    def predict_on_slice_batch(self, feature_arrays: list[FloatArray]) -> FloatArray:
        if self._inference_mode:
            self.model.eval()
        feature_array_tensors = [torch.from_numpy(np.transpose(arr, (0, 3, 1, 2))).float() for arr in feature_arrays]
        feature_array_tensors = [feature_array_tensor.to(self._device) for feature_array_tensor in feature_array_tensors]
        manager = torch.inference_mode if self._inference_mode else nullcontext
        with manager():
            prediction_tensor: torch.Tensor = self.model(*feature_array_tensors)
        prediction: FloatArray = prediction_tensor.detach().cpu().numpy()
        return prediction

    def predict_on_feature_arrays(self, input_feature_arrays: list[FloatArray]) -> FloatArray:
        batch_size = self._settings.inference_batch_size if self._inference_mode else self._settings.training_batch_size
        reshaped_feature_arrays: list[FloatArray] = []
        n_t = input_feature_arrays[0].shape[0]
        for feature_array in input_feature_arrays:
            reshaped_feature_array = feature_array.transpose(1, 2, 0)[np.newaxis, :, :, :]
            reshaped_feature_array_batch_list = [
                reshaped_feature_array[:, :, :, t : t + self._settings.input_number_of_slices] for t in range(0, n_t, self._settings.input_number_of_slices)
            ]
            zeros = np.zeros(
                (1, reshaped_feature_array.shape[1], reshaped_feature_array.shape[2], self._settings.input_number_of_slices - reshaped_feature_array_batch_list[-1].shape[3])
            ).astype(np.float32)
            reshaped_feature_array_batch_list[-1] = np.concatenate([reshaped_feature_array_batch_list[-1], zeros], -1)
            reshaped_feature_batch = np.vstack(reshaped_feature_array_batch_list)
            reshaped_feature_batch = np.transpose(reshaped_feature_batch, (0, 1, 3, 2))
            reshaped_feature_arrays.append(reshaped_feature_batch)
        predictions: list[FloatArray] = []
        for start_idx in range(0, reshaped_feature_arrays[0].shape[0], batch_size):
            end_idx = start_idx + batch_size
            features = [feature_arr[start_idx:end_idx] for feature_arr in reshaped_feature_arrays]
            prediction = self.predict_on_slice_batch(features)
            predictions.append(prediction)
        concatenated_predictions = np.concatenate(np.concatenate(predictions, 0), 1).astype(np.float32)[:, :n_t]
        return concatenated_predictions

    def predict_on_file(self, file: Path) -> FloatArray:
        input_feature_arrays = [self._feature_store.generate_or_load_feature_for_file(file, feature) for feature in self.model_input_features]
        return self.predict_on_feature_arrays(input_feature_arrays)
