import librosa
import numpy as np
import scipy

from pypolyphonicanalysis.analysis.f0_processing.base_f0_processor import BaseF0Processor
from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.utils.utils import (
    FloatArray,
    F0TimesAndFrequencies,
)


def masking_filter(times: FloatArray, freqs: FloatArray, cent_resolution: float, beta: int, L: int) -> FloatArray:
    cents_above_a1 = 1200 * np.log2(freqs / librosa.note_to_hz("A1"), out=np.zeros_like(freqs), where=freqs != 0)
    min_cents = np.min(cents_above_a1[cents_above_a1 > cent_resolution]) - cent_resolution
    max_cents = np.max(cents_above_a1) + cent_resolution
    cent_grid = np.arange(min_cents, max_cents, cent_resolution)
    cent_bins = np.concatenate([[0], (cent_grid[1:] + cent_grid[:-1]) / 2.0, [cent_grid[-1]]])

    digitized_cents = np.digitize(cents_above_a1, cent_bins) - 1

    single_cent_value_activation_columns = np.eye(cent_bins.shape[0], cent_bins.shape[0]).transpose(1, 0)
    digitized_cents_activation_columns = single_cent_value_activation_columns[digitized_cents]
    grid = np.sum(digitized_cents_activation_columns, 1)
    grid[:, 0] = 0
    grid = (grid >= 1).astype(np.float32)

    expanded_freqs = np.zeros((len(times), len(cent_bins)))
    expanded_freqs[np.arange(len(times))[:, np.newaxis], digitized_cents] = freqs

    max_filtered_grid = scipy.ndimage.maximum_filter(grid, (1, beta * 2), mode="constant")
    median_filtered_grid = scipy.ndimage.median_filter(max_filtered_grid, (L - 1, 1), mode="constant")
    new_freqs = np.where(median_filtered_grid == 1, expanded_freqs, np.zeros_like(expanded_freqs))

    new_freqs = np.sort(new_freqs, axis=1)
    new_freqs = new_freqs[:, -freqs.shape[1] :]

    return new_freqs


class MaskingFilter(BaseF0Processor):
    def __init__(self, cent_resolution: float = 10, beta: int = 5, L: int = 43) -> None:
        super().__init__()
        if L % 2 != 1:
            raise ValueError("L must be an odd integer")
        self._cent_resolution = cent_resolution
        self._beta = beta
        self._L = L

    def process(self, recording: Recording, times: FloatArray, freqs: FloatArray) -> F0TimesAndFrequencies:
        new_freqs = masking_filter(times, freqs, self._cent_resolution, self._beta, self._L)
        valid_idxs = np.any(new_freqs > 0, 1)
        return times[valid_idxs], new_freqs[valid_idxs]

    def get_stage_name(self) -> str:
        return f"masking_filter_cent_resolution_{self._cent_resolution}_beta_{self._beta}_L_{self._L}"
