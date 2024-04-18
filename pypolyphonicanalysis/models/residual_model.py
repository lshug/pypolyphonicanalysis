import torch
from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.base_feature_to_salience_map_nn_module import BaseFeatureToSalienceMapNNModule
from pypolyphonicanalysis.models.base_multiple_f0_estimation_model import BaseMultipleF0EstimationModel


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


class ResidualNNModule(BaseFeatureToSalienceMapNNModule):
    def _get_feature_representation_module(self) -> nn.Module:
        return nn.Sequential(nn.BatchNorm2d(self._channels), ResidualCNNBlockModule(self._channels, 32, [5, 5, 5, (70, 3), (70, 3)]))

    def _get_joint_representation_module(self) -> nn.Module:
        return nn.Sequential(ResidualCNNBlockModule(64, 64, [3, 3]), nn.Conv2d(64, 8, (self._bins, 1), padding="same"), nn.BatchNorm2d(8), nn.Conv2d(8, 1, 1, padding="same"))


class ResidualModel(BaseMultipleF0EstimationModel):

    def _create_model(self) -> nn.Module:
        return ResidualNNModule(self.model_input_features, self._settings)

    @property
    def model_input_features(self) -> list[InputFeature]:
        return [Features.HCQT_MAG, Features.HCQT_PHASE_DIFF]

    @property
    def model_label_features(self) -> list[LabelFeature]:
        return [Features.SALIENCE_MAP]
