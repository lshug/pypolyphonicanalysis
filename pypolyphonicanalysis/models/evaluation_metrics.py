import math
from enum import Enum
from typing import Callable

import torch

from pypolyphonicanalysis.datamodel.features.features import LabelFeature, Features


class EvaluationMetrics(Enum):
    SOFT_BINARY_ACCURACY = "soft_binary_accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"


def soft_binary_accuracy(prediction: torch.Tensor, label: torch.Tensor) -> float:
    return float(torch.mean(((torch.round(label) == torch.round(prediction)).float())).item())


def precision(prediction: torch.Tensor, label: torch.Tensor) -> float:
    rounded_pred = torch.round(prediction)
    all_positives_tensor: torch.Tensor = rounded_pred == 1.0
    all_positives = float(torch.sum(all_positives_tensor.float()).item())
    true_positives = float(torch.sum(torch.round(label) * rounded_pred).item())
    if all_positives == 0:
        return math.inf
    return true_positives / all_positives


def recall(prediction: torch.Tensor, label: torch.Tensor) -> float:
    rounded_pred = torch.round(prediction)
    true_positives = float(torch.sum(torch.round(label) * rounded_pred).item())
    negatives_on_label: torch.Tensor = torch.round(label) == 0
    negatives_on_pred: torch.Tensor = rounded_pred == 0
    true_negatives = float(torch.sum(negatives_on_pred.float() * negatives_on_label.float()).item())
    false_negatives = true_negatives - float(torch.sum(negatives_on_pred.float()))
    reciprocal = true_positives + false_negatives
    if reciprocal == 0:
        return math.inf
    return true_positives / reciprocal


def f1(prediction: torch.Tensor, label: torch.Tensor) -> float:
    reciprocal = (1 / precision(prediction, label)) + (1 / recall(prediction, label))
    if reciprocal == 0:
        return math.inf
    return 2 / reciprocal


evaluation_metric_calculation_functions: dict[EvaluationMetrics, Callable[[torch.Tensor, torch.Tensor], float]] = {
    EvaluationMetrics.SOFT_BINARY_ACCURACY: soft_binary_accuracy,
    EvaluationMetrics.PRECISION: precision,
    EvaluationMetrics.RECALL: recall,
    EvaluationMetrics.F1: f1,
}

metrics_calculatable_for_feature: dict[LabelFeature, list[EvaluationMetrics]] = {
    Features.SALIENCE_MAP: [EvaluationMetrics.SOFT_BINARY_ACCURACY, EvaluationMetrics.PRECISION, EvaluationMetrics.RECALL, EvaluationMetrics.F1]
}


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
