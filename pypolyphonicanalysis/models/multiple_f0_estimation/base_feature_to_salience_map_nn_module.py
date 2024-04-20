import abc
import warnings

import torch
from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.model_utils import SelfAttention
from pypolyphonicanalysis.settings import Settings


class BaseFeatureToSalienceMapNNModule(nn.Module, abc.ABC):
    def __init__(self, input_features: tuple[InputFeature, ...], settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._depthwise = settings.use_depthwise_separable_convolution_when_possible
        self._bins = settings.bins_per_octave * settings.n_octaves
        self._feature_representation_channels = settings.feature_representation_channels
        self._concatenated_channels = settings.feature_representation_channels * len(input_features)
        self._channels_pre_flattening = settings.channels_pre_flattening
        self._feature_representation_modules = nn.ModuleList([self._get_feature_representation_module(feature) for feature in input_features])
        self._joint_representation_module = self._get_joint_representation_module()
        self._flatten_channels = nn.Conv2d(settings.channels_pre_flattening, 1, 1)
        if settings.use_self_attention:
            self._self_attention = SelfAttention(self._concatenated_channels)

    @abc.abstractmethod
    def _get_feature_representation_module(self, feature: InputFeature) -> nn.Module:
        pass

    @abc.abstractmethod
    def _get_joint_representation_module(self) -> nn.Module:
        pass

    def forward(self, input_features: list[torch.Tensor]) -> dict[LabelFeature, torch.Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Using padding='same'.*", category=UserWarning)
            concat_repr = torch.concat([self._feature_representation_modules[idx](input_feature) for idx, input_feature in enumerate(input_features)], 1)
        if self._settings.use_self_attention:
            concat_repr = self._self_attention(concat_repr)
        head_repr = self._joint_representation_module(concat_repr)
        logits = self._flatten_channels(head_repr).squeeze(1)
        output_repr = logits
        if not self.training:
            output_repr = torch.sigmoid(output_repr)
        return {Features.SALIENCE_MAP: output_repr}
