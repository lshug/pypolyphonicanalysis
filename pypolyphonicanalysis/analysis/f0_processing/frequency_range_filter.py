import numpy as np

from pypolyphonicanalysis.analysis.f0_processing.base_f0_processor import BaseF0Processor
from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.utils.utils import FloatArray, F0TimesAndFrequencies


class FrequencyRangeFilter(BaseF0Processor):
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        super().__init__()

    def process(self, recording: Recording, times: FloatArray, freqs: FloatArray) -> F0TimesAndFrequencies:
        new_freqs = freqs.copy()
        new_freqs[new_freqs < self._lower_bound] = 0
        new_freqs[new_freqs > self._upper_bound] = 0
        valid_idxs = np.any(new_freqs > 0, 1)
        return times[valid_idxs], new_freqs[valid_idxs]

    def get_stage_name(self) -> str:
        return f"frequency_range_filter_{self._lower_bound}_{self._upper_bound}"
