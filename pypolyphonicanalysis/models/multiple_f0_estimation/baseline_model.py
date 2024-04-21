from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.multiple_f0_estimation.base_feature_to_salience_map_nn_module import BaseFeatureToSalienceMapNNModule
from pypolyphonicanalysis.models.multiple_f0_estimation.base_multiple_f0_estimation_model import BaseMultipleF0EstimationModel
from pypolyphonicanalysis.models.model_utils import Conv


class BaselineNNModule(BaseFeatureToSalienceMapNNModule):

    def _layer_sequence(self, channels: list[int], kernel_sizes: list[int | tuple[int, int]]) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for idx in range(len(kernel_sizes)):
            modules.extend(
                [
                    Conv(channels[idx], channels[idx + 1], kernel_sizes[idx], self._depthwise),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels[idx + 1]),
                ]
            )
        return modules

    def _get_feature_representation_module(self, feature: InputFeature) -> nn.Module:
        match feature:
            case Features.HCQT_MAG | Features.HCQT_PHASE_DIFF:
                channels = len(self._settings.harmonics)
            case _:
                raise NotImplementedError(f"Feature {feature} is not supported by {self}")
        return nn.Sequential(nn.BatchNorm2d(channels), *self._layer_sequence([channels, 16, 32, 32, 32, 32, 32], [5, 5, 5, 5, (70, 3), (70, 3)]))

    def _get_joint_representation_module(self) -> nn.Module:
        return nn.Sequential(*self._layer_sequence([self._concatenated_channels, 64, 64, self._settings.channels_pre_flattening], [3, 3, (self._bins, 1)]))


class BaselineModel(BaseMultipleF0EstimationModel):
    def _create_model(self) -> tuple[str, nn.Module]:
        return "BaselineModel", BaselineNNModule(self.model_input_features, self._settings)

    @property
    def model_label_features(self) -> list[LabelFeature]:
        return [Features.SALIENCE_MAP]
