from enum import Enum
from typing import Callable

import torch

from pypolyphonicanalysis.datamodel.features.features import LabelFeature, Features


class EvaluationMetrics(Enum):
    SOFT_BINARY_ACCURACY = "soft_binary_accuracy"


def soft_binary_accuracy(prediction: torch.Tensor, label: torch.Tensor) -> float:
    return float(torch.mean(((torch.round(label) == torch.round(prediction)).float())).item())


evaluation_metric_calculation_functions: dict[EvaluationMetrics, Callable[[torch.Tensor, torch.Tensor], float]] = {EvaluationMetrics.SOFT_BINARY_ACCURACY: soft_binary_accuracy}

metrics_calculatable_for_feature: dict[LabelFeature, list[EvaluationMetrics]] = {Features.SALIENCE_MAP: [EvaluationMetrics.SOFT_BINARY_ACCURACY]}


def get_evaluation_metrics_for_feature_tensor(feature: LabelFeature, tensor: torch.Tensor, label: torch.Tensor) -> dict[EvaluationMetrics, float]:
    metrics: dict[EvaluationMetrics, float] = {}
    for evaluation_metric in metrics_calculatable_for_feature[feature]:
        metrics[evaluation_metric] = evaluation_metric_calculation_functions[evaluation_metric](tensor, label)
    return metrics


def get_evaluation_metrics_for_model_outputs(model_outputs: dict[LabelFeature, torch.Tensor], label_tensors: list[torch.Tensor]) -> dict[EvaluationMetrics, float]:
    metrics: dict[EvaluationMetrics, float] = {}
    for idx, (feature, tensor) in enumerate(model_outputs.items()):
        metrics |= get_evaluation_metrics_for_feature_tensor(feature, tensor, label_tensors[idx])
    return metrics
