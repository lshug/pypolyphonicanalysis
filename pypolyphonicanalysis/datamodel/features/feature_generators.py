from abc import ABC, abstractmethod
from pathlib import Path

import librosa
import numpy as np
import pumpp
from scipy.ndimage import filters

from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray, IntArray


class FeatureGenerator(ABC):
    def __init__(self, settings: Settings):
        self._settings = settings

    @abstractmethod
    def generate_features_for_sum_track(self, sum_track: SumTrack) -> list[FloatArray]:
        pass

    @property
    @abstractmethod
    def number_of_features(self) -> int:
        pass


class InputFeatureGenerator(FeatureGenerator):
    @abstractmethod
    def generate_features_for_file(self, file: Path) -> list[FloatArray]:
        pass


class LabelFeatureGenerator(FeatureGenerator):
    pass


class HCQTMagPhaseDiffGenerator(InputFeatureGenerator):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._hcqt_mag_and_phase_diff_generator_pump = pumpp.Pump(
            pumpp.feature.HCQTPhaseDiff(
                name="hcqt_phase_diff",
                sr=self._settings.sr,
                hop_length=self._settings.hop_length,
                fmin=self._settings.fmin,
                n_octaves=self._settings.n_octaves,
                over_sample=self._settings.over_sample,
                harmonics=self._settings.harmonics,
                log=True,
                conv="channels_first",
            )
        )

    @property
    def number_of_features(self) -> int:
        return 2

    def _generate_hcqt_features_from_pump(self, file: Path | None = None, y: FloatArray | None = None) -> list[FloatArray]:
        if file is None and y is None:
            raise ValueError("No input file or array provided for generating features from pump")
        if file is None:
            hcqt_data = self._hcqt_mag_and_phase_diff_generator_pump(y=y, sr=self._settings.sr)
        else:
            hcqt_data = self._hcqt_mag_and_phase_diff_generator_pump(file.absolute().as_posix())
        mag = hcqt_data["hcqt_phase_diff/mag"][0]
        phase_diff = hcqt_data["hcqt_phase_diff/dphase"][0]
        assert isinstance(mag, np.ndarray)
        assert isinstance(phase_diff, np.ndarray)
        mag = mag.transpose(0, 2, 1)
        phase_diff = phase_diff.transpose(0, 2, 1)
        return [mag, phase_diff]

    def generate_features_for_sum_track(self, sum_track: SumTrack) -> list[FloatArray]:
        return self._generate_hcqt_features_from_pump(y=sum_track.audio_array)

    def generate_features_for_file(self, file: Path) -> list[FloatArray]:
        return self._generate_hcqt_features_from_pump(file)


class SalienceMapGenerator(LabelFeatureGenerator):
    def _apply_blur_to_salience_map(
        self,
        salience_map: FloatArray,
        time_bin_idxs: IntArray,
        freq_bin_idxs: IntArray,
    ) -> FloatArray:
        blurred_salience_map = filters.gaussian_filter1d(salience_map, 1, axis=0, mode="constant")
        assert isinstance(blurred_salience_map, np.ndarray)
        min_target = np.min(blurred_salience_map[freq_bin_idxs, time_bin_idxs])
        blurred_salience_map /= min_target
        blurred_salience_map[blurred_salience_map > 1.0] = 1
        return blurred_salience_map

    def generate_features_for_sum_track(self, sum_track: SumTrack) -> list[FloatArray]:
        time_grid = librosa.core.frames_to_time(
            range(sum_track.n_frames),
            sr=self._settings.sr,
            hop_length=self._settings.hop_length,
        )
        freq_grid = librosa.cqt_frequencies(
            self._settings.n_octaves * 12 * self._settings.over_sample,
            fmin=self._settings.fmin,
            bins_per_octave=self._settings.bins_per_octave,
        )
        n_bins = self._settings.n_octaves * self._settings.bins_per_octave

        time_bins, freq_bins = [np.concatenate([[0], (grid[1:] + grid[:-1]) / 2.0, [grid[-1]]]) for grid in [time_grid, freq_grid]]
        times_and_freqs = [voice.f0_trajectory_annotation for voice in sum_track.source_multitrack]
        times = np.concatenate([voice_times_and_freqs[0] for voice_times_and_freqs in times_and_freqs])
        freqs = np.concatenate([voice_times_and_freqs[1] for voice_times_and_freqs in times_and_freqs])

        time_bin_idxs = np.digitize(times, time_bins).astype(np.int64) - 1
        freq_bin_idxs = np.digitize(freqs, freq_bins).astype(np.int64) - 1

        idx = time_bin_idxs < sum_track.n_frames
        time_bin_idxs = time_bin_idxs[idx]
        freq_bin_idxs = freq_bin_idxs[idx]

        idx = freq_bin_idxs < n_bins
        time_bin_idxs = time_bin_idxs[idx]
        freq_bin_idxs = freq_bin_idxs[idx]

        salience_map = np.zeros((n_bins, sum_track.n_frames)).astype(np.float32)
        salience_map[freq_bin_idxs, time_bin_idxs] = 1

        if self._settings.blur_salience_map:
            salience_map = self._apply_blur_to_salience_map(salience_map, time_bin_idxs, freq_bin_idxs)

        return [salience_map]

    @property
    def number_of_features(self) -> int:
        return 1
