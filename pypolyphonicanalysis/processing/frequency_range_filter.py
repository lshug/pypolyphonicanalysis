import numpy as np

from pypolyphonicanalysis.processing.base_processor import BaseProcessor
from pypolyphonicanalysis.utils.utils import FloatArray, F0TimesAndFrequencies


class FrequencyRangeFilter(BaseProcessor):
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        super().__init__()

    def process(self, times: FloatArray, freqs: list[FloatArray]) -> F0TimesAndFrequencies:
        new_times: list[float] = []
        new_freqs: list[FloatArray] = []
        for time, freq_array in zip(times, freqs):
            new_freq_array_list: list[float] = []
            for freq in freq_array:
                if freq > self._lower_bound and freq < self._upper_bound:
                    new_freq_array_list.append(freq)
            if len(new_freq_array_list) > 0:
                new_freq_array = np.array(freq_array).astype(np.float32)
                new_times.append(time)
                new_freqs.append(new_freq_array)
        return np.array(new_times), new_freqs

    def get_stage_name(self) -> str:
        return f"frequency_range_filter_{self._lower_bound}_{self._upper_bound}"
