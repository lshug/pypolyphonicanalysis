from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.multiple_f0_estimation.base_feature_to_salience_map_nn_module import BaseFeatureToSalienceMapNNModule
from pypolyphonicanalysis.models.model_utils import UNet
from pypolyphonicanalysis.models.multiple_f0_estimation.base_multiple_f0_estimation_model import BaseMultipleF0EstimationModel


class UNetNNModule(BaseFeatureToSalienceMapNNModule):
    def _get_feature_representation_module(self, feature: InputFeature) -> nn.Module:
        match feature:
            case Features.HCQT_MAG | Features.HCQT_PHASE_DIFF:
                channels = len(self._settings.harmonics)
            case _:
                raise NotImplementedError(f"Feature {feature} is not supported by {self}")
        return nn.Sequential(nn.BatchNorm2d(channels), UNet(channels, self._feature_representation_channels, self._depthwise))

    def _get_joint_representation_module(self) -> nn.Module:
        return UNet(self._concatenated_channels, self._channels_pre_flattening, self._depthwise)


class UNetModel(BaseMultipleF0EstimationModel):

    def _create_model(self) -> tuple[str, nn.Module]:
        return "UNetModel", UNetNNModule(self.model_input_features, self._settings)

    @property
    def model_label_features(self) -> list[LabelFeature]:
        return [Features.SALIENCE_MAP]
