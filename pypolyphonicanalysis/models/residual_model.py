import torch
from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.base_multiple_f0_estimation_model import BaseMultipleF0EstimationModel
from pypolyphonicanalysis.settings import Settings


class ResidualCNNBlockModule(nn.Module):
    def __init__(self, input_channels: int, channels: int, kernel_sizes: list[int | tuple[int, int]]) -> None:
        super().__init__()
        self._residual_connection_on_first_block = input_channels == channels
        self._convs = nn.ModuleList(
            [nn.Conv2d(input_channels, channels, 5, padding="same")] + [nn.Conv2d(channels, channels, kernel_sizes[idx], padding="same") for idx in range(len(kernel_sizes))]
        )
        self._activations = nn.ModuleList([nn.ReLU()] + [nn.ReLU() for _ in range(len(kernel_sizes))])
        self._batchnorms = nn.ModuleList([nn.BatchNorm2d(channels)] + [nn.BatchNorm2d(channels) for _ in range(len(kernel_sizes))])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        for idx, (conv, activation, batchnorm) in enumerate(zip(self._convs, self._activations, self._batchnorms)):
            inp = out
            out = conv(out)
            out = activation(out)
            out = batchnorm(out)
            if idx != 0 or self._residual_connection_on_first_block:
                out += inp
        return out


class ResidualNNModule(nn.Module):

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._channels = len(settings.harmonics)
        self._bins = settings.bins_per_octave * settings.n_octaves

        self._mag_base_model = nn.Sequential(nn.BatchNorm2d(self._channels), ResidualCNNBlockModule(self._channels, 32, [5, 5, 5, (70, 3), (70, 3)]))
        self._phase_diff_base_model = nn.Sequential(nn.BatchNorm2d(self._channels), *[ResidualCNNBlockModule(self._channels, 32, [5, 5, 5, (70, 3), (70, 3)])])
        self._model_head = nn.Sequential(
            ResidualCNNBlockModule(64, 64, [3, 3]), nn.Conv2d(64, 8, (self._bins, 1), padding="same"), nn.BatchNorm2d(8), nn.Conv2d(8, 1, 1, padding="same")
        )

    def forward(self, mag: torch.Tensor, phase_diff: torch.Tensor) -> torch.Tensor:
        concat_repr = torch.concat([self._mag_base_model(mag), self._phase_diff_base_model(phase_diff)], 1)
        output_repr: torch.Tensor = self._model_head(concat_repr)
        output_repr = torch.sigmoid(output_repr)
        return output_repr.squeeze(1)


class ResidualModel(BaseMultipleF0EstimationModel):

    def _create_model(self) -> nn.Module:
        return ResidualNNModule(self._settings)

    @property
    def model_input_features(self) -> list[InputFeature]:
        return [Features.HCQT_MAG, Features.HCQT_PHASE_DIFF]

    @property
    def model_label_features(self) -> list[LabelFeature]:
        return [Features.SALIENCE_MAP]
