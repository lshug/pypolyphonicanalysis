from typing import Callable

import torch

from pypolyphonicanalysis.datamodel.features.features import LabelFeature, Features

TRAIN_LOSSES: dict[LabelFeature, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {Features.SALIENCE_MAP: torch.nn.BCEWithLogitsLoss()}
EVAL_LOSSES: dict[LabelFeature, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {Features.SALIENCE_MAP: torch.nn.BCELoss()}


def get_train_loss_function(feature: LabelFeature) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return TRAIN_LOSSES[feature]


def get_eval_loss_function(feature: LabelFeature) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return TRAIN_LOSSES[feature]
