from enum import Enum
from typing import TypedDict


class SumTrackSplitType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


class TrainTestValidationSplit(TypedDict):
    train: list[str]
    test: list[str]
    validation: list[str]
