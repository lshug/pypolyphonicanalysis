import math

import librosa
import numpy as np
import scipy

from pypolyphonicanalysis.processing.base_processor import BaseProcessor
from pypolyphonicanalysis.utils.utils import (
    FloatArray,
    F0TimesAndFrequencies,
    get_voice_times_and_f0s_from_times_and_freqs,
)


class MaskingFilter(BaseProcessor):
    def __init__(self, cent_resolution: int = 10, beta: int = 5, L: int = 43) -> None:
        super().__init__()
        if L % 2 != 1:
            raise ValueError("L must be an odd integer")
        self._cent_resolution = cent_resolution
        self._beta = beta
        self._L = L

    def process(self, times: FloatArray, freqs: list[FloatArray]) -> F0TimesAndFrequencies:
        voice_times_and_freqs = get_voice_times_and_f0s_from_times_and_freqs(times, freqs)
        voice_cents_dict = {
            voice: np.array([1200 * math.log2(f0 / librosa.note_to_hz("A1")) if f0 != 0 else 0 for f0 in np.array(list(times_and_freqs.values()))])
            for voice, times_and_freqs in voice_times_and_freqs.items()
        }
        cents_matrix = np.array(list(voice_cents_dict.values()))
        min_cents = np.min([f for f in cents_matrix.flatten() if f > self._cent_resolution]) - self._cent_resolution
        max_cents = np.max(cents_matrix.flatten()) + self._cent_resolution
        cent_grid = np.arange(min_cents, max_cents, self._cent_resolution)
        cent_bins = np.concatenate([[0], (cent_grid[1:] + cent_grid[:-1]) / 2.0, [cent_grid[-1]]])
        grid = np.zeros((times.shape[0], cent_grid.shape[0]))
        zeros = np.zeros_like(grid)
        eye = np.eye(grid.shape[0], grid.shape[1])
        for cents in voice_cents_dict.values():
            digitized_cents = np.digitize(cents, cent_bins) - 1
            grid += np.where(digitized_cents.reshape(-1, 1) > 0, eye[digitized_cents], zeros)
        max_filtered_grid = scipy.ndimage.maximum_filter(grid, (1, self._beta * 2), mode="constant")
        median_filtered_grid = scipy.ndimage.median_filter(max_filtered_grid, (self._L - 1, 1), mode="constant")
        new_freq_arrs: list[FloatArray] = []
        for times_and_freqs, cents in zip(voice_times_and_freqs.values(), voice_cents_dict.values()):
            freqs_arr = list(times_and_freqs.values())
            new_freq_arrs.append(
                np.array(
                    [(freq_val if median_filtered_grid[idx][np.digitize(cent_val, cent_bins) - 1] == 1 else 0) for idx, (freq_val, cent_val) in enumerate(zip(freqs_arr, cents))]
                )
            )
        new_freqs_with_zero_rows = np.transpose(np.array(new_freq_arrs), (1, 0)).tolist()
        new_times_and_freqs = [(t, np.array([f_elem for f_elem in f if f_elem != 0])) for t, f in zip(times, new_freqs_with_zero_rows) if not np.all(f == 0)]
        new_times = np.array([t for t, f in new_times_and_freqs])
        new_freqs = [f for t, f in new_times_and_freqs]

        return new_times, new_freqs

    def get_stage_name(self) -> str:
        return f"masking_filter_cent_resolution_{self._cent_resolution}_beta_{self._beta}_L_{self._L}"
