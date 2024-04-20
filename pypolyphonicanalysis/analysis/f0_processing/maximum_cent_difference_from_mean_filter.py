import librosa
import numpy as np

from pypolyphonicanalysis.analysis.f0_processing.base_f0_processor import BaseF0Processor
from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.utils.utils import FloatArray, F0TimesAndFrequencies


class MaximumCentDifferenceFromMeanFilter(BaseF0Processor):
    def __init__(self, max_difference: float = 1200) -> None:
        self._max_difference = max_difference
        super().__init__()

    def process(self, recording: Recording, times: FloatArray, freqs: FloatArray) -> F0TimesAndFrequencies:
        cents_above_a1 = 1200 * np.log2(freqs / librosa.note_to_hz("A1"), out=np.zeros_like(freqs), where=freqs != 0)
        mean = np.mean(cents_above_a1[cents_above_a1 != 0])
        above_lower_bound_indices = cents_above_a1 > (mean - self._max_difference)
        below_upper_bound_indices = cents_above_a1 < (mean + self._max_difference)
        within_bounds_indices = above_lower_bound_indices * below_upper_bound_indices
        new_freqs = np.where(within_bounds_indices, freqs, np.zeros_like(freqs))
        valid_idxs = np.any(new_freqs > 0, 1)
        return times[valid_idxs], new_freqs[valid_idxs]

    def get_stage_name(self) -> str:
        return f"max_cent_difference_from_mean_filter_{self._max_difference}"
