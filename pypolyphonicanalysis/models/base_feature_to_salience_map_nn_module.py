import abc

import torch
from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.settings import Settings


class BaseFeatureToSalienceMapNNModule(nn.Module, abc.ABC):
    def __init__(self, input_features: list[InputFeature], settings: Settings) -> None:
        super().__init__()
        self._channels = len(settings.harmonics)
        self._bins = settings.bins_per_octave * settings.n_octaves
        self._feature_representation_modules = nn.ModuleList([self._get_feature_representation_module() for _ in range(len(input_features))])
        self._joint_representation_module = self._get_joint_representation_module()
        self._flatten_channels = nn.Conv2d(8, 1, 1)

    @abc.abstractmethod
    def _get_feature_representation_module(self) -> nn.Module:
        pass

    @abc.abstractmethod
    def _get_joint_representation_module(self) -> nn.Module:
        pass

    def forward(self, input_features: list[torch.Tensor]) -> dict[LabelFeature, torch.Tensor]:
        concat_repr = torch.concat([self._feature_representation_modules[idx](input_features) for idx, input_feature in enumerate(input_features)], 1)
        head_repr = self._joint_representation_module(concat_repr)
        logits = self._flatten_channels(head_repr).squeeze(1)
        output_repr = logits
        if not self.training:
            output_repr = torch.sigmoid(output_repr)
        return {Features.SALIENCE_MAP: output_repr}
