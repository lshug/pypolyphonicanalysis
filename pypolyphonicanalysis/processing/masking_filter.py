import librosa
import numpy as np
import scipy

from pypolyphonicanalysis.processing.base_processor import BaseProcessor
from pypolyphonicanalysis.utils.utils import (
    FloatArray,
    F0TimesAndFrequencies,
)


class MaskingFilter(BaseProcessor):
    def __init__(self, cent_resolution: int = 10, beta: int = 5, L: int = 43) -> None:
        super().__init__()
        if L % 2 != 1:
            raise ValueError("L must be an odd integer")
        self._cent_resolution = cent_resolution
        self._beta = beta
        self._L = L

    def process(self, times: FloatArray, freqs: FloatArray) -> F0TimesAndFrequencies:
        cents_above_a1 = 1200 * np.log2(freqs / librosa.note_to_hz("A1"), out=np.zeros_like(freqs), where=freqs != 0)
        min_cents = np.min(cents_above_a1[cents_above_a1 > self._cent_resolution]) - self._cent_resolution
        max_cents = np.max(cents_above_a1) + self._cent_resolution
        cent_grid = np.arange(min_cents, max_cents, self._cent_resolution)
        cent_bins = np.concatenate([[0], (cent_grid[1:] + cent_grid[:-1]) / 2.0, [cent_grid[-1]]])

        digitized_cents = np.digitize(cents_above_a1, cent_bins) - 1
        bin_vals = cent_bins[digitized_cents]
        grid = np.sum(np.eye(times.shape[0], cent_grid.shape[0])[digitized_cents], 1)
        grid[:, 0] = 0

        max_filtered_grid = scipy.ndimage.maximum_filter(grid, (1, self._beta * 2), mode="constant")
        median_filtered_grid = scipy.ndimage.median_filter(max_filtered_grid, (self._L - 1, 1), mode="constant")

        filtered_freq_bins_arr = cent_grid * median_filtered_grid
        np.sort(filtered_freq_bins_arr, 1)
        filtered_freq_bins_arr = filtered_freq_bins_arr[:, -freqs.shape[1] :]
        new_freqs = np.where(filtered_freq_bins_arr == bin_vals, freqs, np.zeros_like(freqs))
        valid_idxs = np.any(new_freqs > 0, 1)
        return times[valid_idxs], new_freqs[valid_idxs]

    def get_stage_name(self) -> str:
        return f"masking_filter_cent_resolution_{self._cent_resolution}_beta_{self._beta}_L_{self._L}"
