from torch import nn

from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature, Features
from pypolyphonicanalysis.models.base_feature_to_salience_map_nn_module import BaseFeatureToSalienceMapNNModule
from pypolyphonicanalysis.models.base_multiple_f0_estimation_model import BaseMultipleF0EstimationModel


class BaselineNNModule(BaseFeatureToSalienceMapNNModule):

    def _layer_sequence(self, channels: list[int], kernel_sizes: list[int | tuple[int, int]]) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for idx in range(len(kernel_sizes)):
            modules.extend(
                [
                    nn.Conv2d(channels[idx], channels[idx + 1], kernel_sizes[idx], padding="same"),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels[idx + 1]),
                ]
            )
        return modules

    def _get_feature_representation_module(self) -> nn.Module:
        return nn.Sequential(nn.BatchNorm2d(self._channels), *self._layer_sequence([self._channels, 16, 32, 32, 32, 32, 32], [5, 5, 5, 5, (70, 3), (70, 3)]))

    def _get_joint_representation_module(self) -> nn.Module:
        return nn.Sequential(*self._layer_sequence([64, 64, 64, 8], [3, 3, (self._bins, 1)]))


class BaselineModel(BaseMultipleF0EstimationModel):
    def _create_model(self) -> nn.Module:
        return BaselineNNModule(self.model_input_features, self._settings)

    @property
    def model_input_features(self) -> list[InputFeature]:
        return [Features.HCQT_MAG, Features.HCQT_PHASE_DIFF]

    @property
    def model_label_features(self) -> list[LabelFeature]:
        return [Features.SALIENCE_MAP]
