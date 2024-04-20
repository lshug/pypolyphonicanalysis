from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.multiple_f0_estimation.base_feature_to_salience_map_nn_module import BaseFeatureToSalienceMapNNModule
from pypolyphonicanalysis.models.multiple_f0_estimation.base_multiple_f0_estimation_model import BaseMultipleF0EstimationModel
from pypolyphonicanalysis.models.model_utils import Conv, ResidualCNNBlockModule


class ResidualNNModule(BaseFeatureToSalienceMapNNModule):
    def _get_feature_representation_module(self, feature: InputFeature) -> nn.Module:
        match feature:
            case Features.HCQT_MAG | Features.HCQT_PHASE_DIFF:
                channels = len(self._settings.harmonics)
            case _:
                raise NotImplementedError(f"Feature {feature} is not supported by {self}")
        return nn.Sequential(nn.BatchNorm2d(channels), ResidualCNNBlockModule(channels, self._feature_representation_channels, [5, 5, 5, (70, 3), (70, 3)], self._depthwise))

    def _get_joint_representation_module(self) -> nn.Module:
        return nn.Sequential(
            ResidualCNNBlockModule(self._concatenated_channels, 64, [3, 3], self._depthwise),
            Conv(64, self._settings.channels_pre_flattening, (self._bins, 1), self._depthwise),
            nn.BatchNorm2d(8),
        )


class ResidualModel(BaseMultipleF0EstimationModel):

    def _create_model(self) -> nn.Module:
        return ResidualNNModule(self.model_input_features, self._settings)

    @property
    def model_label_features(self) -> list[LabelFeature]:
        return [Features.SALIENCE_MAP]
