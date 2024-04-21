import enum


class MultitrackAlignmentStrategy(enum.Enum):
    TRIM = 0
    CYCLE = 1
    PAD = 2
