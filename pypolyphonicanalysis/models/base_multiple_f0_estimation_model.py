import logging
import math
from abc import abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature
from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.models.evaluation_metrics import get_evaluation_metrics_for_model_outputs
from pypolyphonicanalysis.models.losses import get_train_loss_function, get_eval_loss_function
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store

logger = logging.getLogger(__name__)


def get_models_path(settings: Settings) -> Path:
    models_path = Path(settings.data_directory_path).joinpath("models")
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


class BaseMultipleF0EstimationModel:
    def __init__(self, settings: Settings, name: str | None = None) -> None:
        self._settings = settings
        self._feature_store = get_feature_store(self._settings)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        if name is not None:
            self.model.load_state_dict(torch.load(get_models_path(self._settings).joinpath(f"{name}.pth").absolute().as_posix()))

    @property
    @abstractmethod
    def model_input_features(self) -> list[InputFeature]:
        pass

    @property
    @abstractmethod
    def model_label_features(self) -> list[LabelFeature]:
        pass

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._create_model().to(self._device)
        return self._model

    @abstractmethod
    def _create_model(self) -> nn.Module:
        pass

    def predict_on_slice_batch(self, feature_arrays: list[FloatArray]) -> dict[LabelFeature, FloatArray]:
        feature_array_tensors = [torch.from_numpy(arr).float().to(self._device) for arr in feature_arrays]
        self.model.eval()
        with torch.inference_mode():
            prediction_tensors_dict: dict[LabelFeature, torch.Tensor] = self.model(*feature_array_tensors)
        prediction: dict[LabelFeature, FloatArray] = {feature: prediction_tensor.detach().cpu().numpy() for feature, prediction_tensor in prediction_tensors_dict.items()}
        return prediction

    def predict_on_feature_arrays(self, input_feature_arrays: list[FloatArray]) -> dict[LabelFeature, FloatArray]:
        batch_size = self._settings.inference_batch_size
        reshaped_feature_arrays: list[FloatArray] = []
        n_t = input_feature_arrays[0].shape[-1]
        n_bins = self._settings.bins_per_octave * self._settings.n_octaves
        n_harmonics = len(self._settings.harmonics)
        for feature_array in input_feature_arrays:
            reshaped_feature_array = feature_array[np.newaxis, :, :, :]  # batch, harmonics, bins, t
            reshaped_feature_array_batch_list = [
                reshaped_feature_array[:, :, :, t : t + self._settings.inference_input_number_of_slices] for t in range(0, n_t, self._settings.inference_input_number_of_slices)
            ]
            zeros = np.zeros(
                (
                    1,
                    n_harmonics,
                    n_bins,
                    self._settings.inference_input_number_of_slices - reshaped_feature_array_batch_list[-1].shape[3],
                )
            ).astype(np.float32)
            reshaped_feature_array_batch_list[-1] = np.concatenate([reshaped_feature_array_batch_list[-1], zeros], 3)
            reshaped_feature_batch = np.vstack(reshaped_feature_array_batch_list)
            reshaped_feature_arrays.append(reshaped_feature_batch)
        predictions: dict[LabelFeature, list[FloatArray]] = {feature: [] for feature in self.model_label_features}
        for start_idx in range(0, reshaped_feature_arrays[0].shape[0], batch_size):
            end_idx = start_idx + batch_size
            features = [feature_arr[start_idx:end_idx] for feature_arr in reshaped_feature_arrays]
            prediction = self.predict_on_slice_batch(features)
            for feature, arr in prediction.items():
                predictions[feature].append(arr)
        concatenated_predictions: dict[LabelFeature, FloatArray] = {
            feature: np.concatenate(np.concatenate(feature_pred_list, 0), 1).astype(np.float32)[:, :n_t] for feature, feature_pred_list in predictions.items()
        }
        return concatenated_predictions

    def predict_on_file(self, file: Path) -> dict[LabelFeature, FloatArray]:
        input_feature_arrays = [self._feature_store.generate_or_load_feature_for_file(file, feature) for feature in self.model_input_features]
        return self.predict_on_feature_arrays(input_feature_arrays)

    def predict_on_sum_track(self, sum_track: SumTrack) -> dict[LabelFeature, FloatArray]:
        input_feature_arrays = [self._feature_store.generate_or_load_feature_for_sum_track(sum_track, feature) for feature in self.model_input_features]
        return self.predict_on_feature_arrays(input_feature_arrays)

    def train_on_feature_iterable(
        self, train_iterator: Iterator[tuple[list[FloatArray], list[FloatArray]]], optimizer: torch.optim.Optimizer, expected_iter_length: int | None = None
    ) -> None:
        try:
            torch.cuda.is_available()
        except AssertionError:
            raise SystemError("CUDA must be available for training")
        scaler = torch.cuda.amp.GradScaler()
        total_loss: float = 0
        batch_count = 0
        self.model.train()
        for input_features, label_features in tqdm(train_iterator, desc="Training", total=expected_iter_length):
            input_feature_tensors = [torch.from_numpy(arr).float().to(self._device) for arr in input_features]
            label_feature_tensors = [torch.from_numpy(arr).float().to(self._device) for arr in label_features]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                model_output: dict[LabelFeature, torch.Tensor] = self.model(*input_feature_tensors)
                loss = torch.sum(
                    torch.stack([get_train_loss_function(feature)(model_output[feature], label_feature_tensors[idx]) for idx, (feature, output) in enumerate(model_output.items())])
                )
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
            last_loss = loss.item()
            total_loss += last_loss
            batch_count += 1
            logger.info(f"Batch {batch_count}: average loss={total_loss / batch_count}, last loss={last_loss}")

    def validate_on_feature_iterable(self, validation_iterator: Iterator[tuple[list[FloatArray], list[FloatArray]]], expected_iter_length: int | None = None) -> float:
        total_loss: float = 0
        batch_count = 0
        self.model.eval()
        with torch.no_grad():
            for input_features, label_features in tqdm(validation_iterator, desc="Validation", total=expected_iter_length):
                input_feature_tensors = [torch.from_numpy(arr).float().to(self._device) for arr in input_features]
                label_feature_tensors = [torch.from_numpy(arr).float().to(self._device) for arr in label_features]
                model_output = self.model(*input_feature_tensors)
                loss = torch.sum(
                    torch.stack([get_eval_loss_function(feature)(model_output[feature], label_feature_tensors[idx]) for idx, (feature, output) in enumerate(model_output.items())])
                )
                evaluation_metrics = get_evaluation_metrics_for_model_outputs(model_output, label_feature_tensors)
                evaluation_metrics_strings = [f"{metric.value}={val}" for metric, val in evaluation_metrics.items()]
                last_loss: float = loss.item()
                total_loss += last_loss
                batch_count += 1
                logger.info(f"Average loss={total_loss / batch_count}, last loss={last_loss}, {', '.join(evaluation_metrics_strings)}")
        if batch_count == 0:
            logger.warning("Empty validation iterator encountered")
            return math.inf
        return total_loss / batch_count

    def save(self, name: str) -> None:
        torch.save(self.model.state_dict(), get_models_path(self._settings).joinpath(f"{name}.pth").absolute().as_posix())
