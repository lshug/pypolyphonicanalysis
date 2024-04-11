import enum
from typing import Literal


class Features(enum.Enum):
    HCQT_MAG = 1
    HCQT_PHASE_DIFF = 2
    SALIENCE_MAP = 99


class FeatureType(enum.Enum):
    INPUT = 1
    LABEL = 2


InputFeature = Literal[Features.HCQT_MAG] | Literal[Features.HCQT_PHASE_DIFF]
LabelFeature = Literal[Features.SALIENCE_MAP]
