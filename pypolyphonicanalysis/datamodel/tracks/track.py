import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import librosa

from pyrubberband import pyrb
from scipy.io import wavfile

from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray, median_group_delay, check_output_path


def get_tracks_path(settings: Settings) -> Path:
    tracks_path = Path(os.path.join(settings.data_directory_path, "tracks"))
    check_output_path(tracks_path)
    return tracks_path


def load_f0_trajectory_array_from_path(path: Path) -> tuple[FloatArray, FloatArray]:
    assert path.is_file()
    match path.suffix:
        case ".csv":
            arr = pd.read_csv(path, header=None).to_numpy()
            return arr[:, 0], arr[:, 1]
        case ".f0":
            arr = np.array(np.loadtxt(path, dtype=np.float32))
            return arr[:, 0], arr[:, 1]
        case ".npy":
            arr = np.load(path).astype(np.float32)
            return arr[:, 0], arr[:, 1]
        case _:
            raise ValueError(f"Format {path.suffix} is not supported.")


def track_is_saved(track_name: str, settings: Settings) -> bool:
    tracks_path = get_tracks_path(settings)
    track_path = tracks_path.joinpath(track_name)
    return track_path.is_dir() and track_path.joinpath(".saved").is_file()


def load_track(track_name: str, settings: Settings) -> "Track":
    tracks_path = get_tracks_path(settings)
    track_path = tracks_path.joinpath(track_name)
    assert track_path.is_dir()
    with open(track_path.joinpath("track_data.json"), "r") as f:
        track_data = json.load(f)
    return Track(
        track_name,
        Path(track_data["audio_source_path"]),
        settings,
        track_path.joinpath("f0_trajectory_annotation.npy"),
    )


class Track:
    def __init__(
        self,
        name: str,
        audio_source: FloatArray | Path,
        settings: Settings,
        f0_source: Path | tuple[FloatArray, FloatArray] | None = None,
    ) -> None:
        self._settings = settings
        self._name = name
        self._audio_source = audio_source
        self._audio_array: FloatArray | None = None
        self._f0_source = f0_source
        self._f0_trajectory_annotation: tuple[FloatArray, FloatArray] | None = None
        self._n_frames: int | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def audio_array(self) -> FloatArray:
        if self._audio_array is None:
            match self._audio_source:
                case Path():
                    self._audio_array = librosa.load(self._audio_source.absolute().as_posix(), sr=self._settings.sr, mono=True)[0]
                case _:
                    source_arr = self._audio_source
                    assert isinstance(source_arr, np.ndarray)
                    self._audio_array = source_arr
        return self._audio_array

    @property
    def n_frames(self) -> int:
        if self._n_frames is None:
            self._n_frames = self.audio_array.shape[0] // self._settings.hop_length
        return self._n_frames

    @property
    def f0_trajectory_annotation(self) -> tuple[FloatArray, FloatArray]:
        if self._f0_trajectory_annotation is None:
            match self._f0_source:
                case None:
                    self._f0_trajectory_annotation = self._get_f0_trajectory_from_audio_track()
                case Path():
                    self._f0_trajectory_annotation = load_f0_trajectory_array_from_path(self._f0_source)
                case _:
                    f0_source = self._f0_source
                    assert isinstance(f0_source, tuple)
                    self._f0_trajectory_annotation = self._f0_source
        return self._f0_trajectory_annotation

    def save(self) -> None:
        if track_is_saved(self.name, self._settings):
            return
        tracks_path = get_tracks_path(self._settings)
        track_path = tracks_path.joinpath(self.name)
        check_output_path(track_path)
        time_freq_arr = np.stack(self.f0_trajectory_annotation).transpose()
        np.save(track_path.joinpath("f0_trajectory_annotation.npy").absolute().as_posix(), time_freq_arr)
        source_path: str
        match self._audio_source:
            case Path():
                source_path = self._audio_source.absolute().as_posix()
            case _:
                wavfile.write(track_path.joinpath("audio.wav"), self._settings.sr, self.audio_array)
                source_path = track_path.joinpath("audio.wav").absolute().as_posix()
        with open(track_path.joinpath("track_data.json"), "w") as f:
            json.dump({"audio_source_path": source_path}, f)
        with open(track_path.joinpath(".saved"), "a"):
            os.utime(track_path.joinpath(".saved"), None)

    def pitch_shift(self, n_steps: float) -> "Track":
        if n_steps == 0:
            return self
        shift_prefix = f"pshift_{n_steps:.2f}_"
        track_name = f"{shift_prefix}{self._name}"
        if track_is_saved(track_name, self._settings):
            return load_track(track_name, self._settings)
        audio_array = pyrb.pitch_shift(self.audio_array, self._settings.sr, n_steps)
        frequency_multiplier = 2 ** (n_steps / 12)
        times, freqs = self.f0_trajectory_annotation
        freqs = freqs * frequency_multiplier
        return Track(track_name, audio_array, self._settings, (times, freqs))

    def pitch_shift_range(self, lb: int, ub: int) -> Iterable["Track"]:
        for semitones in range(lb, ub + 1):
            yield self.pitch_shift(semitones)

    def time_shift(self, delay: float) -> "Track":
        shift_suffix = f"time_shift_{delay}_"
        track_name = f"{shift_suffix}{self._name}"
        if track_is_saved(track_name, self._settings):
            return load_track(track_name, self._settings)
        times, freqs = self.f0_trajectory_annotation
        times = times + delay
        zeros = np.zeros(
            (
                int(
                    np.ceil(delay * self._settings.sr),
                )
            )
        ).astype(np.float32)
        audio_array = np.concatenate([zeros, self.audio_array])
        time_shifted_track = Track(track_name, audio_array, self._settings, (times, freqs))
        return time_shifted_track

    def time_shift_by_ir(self, ir: FloatArray) -> "Track":
        delay = median_group_delay(ir, self._settings.sr)
        return self.time_shift(delay)

    def _get_f0_trajectory_from_audio_track(
        self,
    ) -> tuple[FloatArray, FloatArray]:
        f0, voiced_flag, _ = librosa.pyin(y=self.audio_array, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), fill_na=0.0)
        times = librosa.times_like(f0, sr=self._settings.sr)
        for idx, flag in enumerate(voiced_flag):
            if flag is False:
                f0[idx] = 0
        return times, f0

    def trim_to_frames(self, n_frames: int) -> "Track":
        if n_frames == self._n_frames:
            return self
        if track_is_saved(f"{self.name}_trim_{n_frames}", self._settings):
            return load_track(f"{self.name}_trim_{n_frames}", self._settings)
        n_samples = n_frames * self._settings.hop_length
        return Track(
            f"{self.name}_trim_{n_frames}", self.audio_array[:n_samples], self._settings, (self.f0_trajectory_annotation[0][:n_frames], self.f0_trajectory_annotation[1][:n_frames])
        )

    def __repr__(self) -> str:
        return f"Track({self.name})"

    def __hash__(self) -> int:
        return hash(repr(self))
