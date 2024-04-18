import torch
from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.base_multiple_f0_estimation_model import BaseMultipleF0EstimationModel
from pypolyphonicanalysis.settings import Settings


class BaselineNNModule(nn.Module):

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._channels = len(settings.harmonics)
        self._bins = settings.bins_per_octave * settings.n_octaves
        self._mag_base_model = nn.Sequential(*self._layer_sequence([self._channels, 16, 32, 32, 32, 32, 32], [5, 5, 5, 5, (70, 3), (70, 3)], True))
        self._phase_diff_base_model = nn.Sequential(*self._layer_sequence([self._channels, 16, 32, 32, 32, 32, 32], [5, 5, 5, 5, (70, 3), (70, 3)], True))
        self._model_head = nn.Sequential(*self._layer_sequence([64, 64, 64, 8, 1], [3, 3, (self._bins, 1), 1])[:-2])
        self._epsilon = settings.epsilon

    def _layer_sequence(self, channels: list[int], kernel_sizes: list[int | tuple[int, int]], initial_batchnorm: bool = False) -> list[nn.Module]:
        modules: list[nn.Module] = []
        if initial_batchnorm:
            modules.append(nn.BatchNorm2d(channels[0]))
        for idx in range(len(kernel_sizes)):
            modules.extend(
                [
                    nn.Conv2d(channels[idx], channels[idx + 1], kernel_sizes[idx], padding="same"),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels[idx + 1]),
                ]
            )
        return modules

    def forward(self, mag: torch.Tensor, phase_diff: torch.Tensor) -> dict[LabelFeature, torch.Tensor]:
        concat_repr = torch.concat([self._mag_base_model(mag), self._phase_diff_base_model(phase_diff)], 1)
        output_repr: torch.Tensor = self._model_head(concat_repr).squeeze(1)
        if not self.training:
            output_repr.sigmoid_()
        return {Features.SALIENCE_MAP: output_repr}


class BaselineModel(BaseMultipleF0EstimationModel):
    def _create_model(self) -> nn.Module:
        return BaselineNNModule(self._settings)

    @property
    def model_input_features(self) -> list[InputFeature]:
        return [Features.HCQT_MAG, Features.HCQT_PHASE_DIFF]

    @property
    def model_label_features(self) -> list[LabelFeature]:
        return [Features.SALIENCE_MAP]
