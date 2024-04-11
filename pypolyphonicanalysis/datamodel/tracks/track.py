import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import librosa
import jams
from jams import FileMetadata

import muda
from muda.deformers import median_group_delay
from scipy.io import wavfile

from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray


def get_tracks_path(settings: Settings) -> Path:
    tracks_path = Path(os.path.join(settings.data_directory_path, "tracks"))
    tracks_path.mkdir(parents=True, exist_ok=True)
    return tracks_path


def load_f0_trajectory_array_from_path(path: Path) -> FloatArray:
    assert path.is_file()
    match path.suffix:
        case ".csv":
            return pd.read_csv(path, header=None).values
        case ".f0":
            return np.loadtxt(path, dtype=np.float32)
        case _:
            raise ValueError(f"Format {path.suffix} is not supported.")


def load_f0_trajectory_from_path(path: Path) -> jams.JAMS:
    assert path.is_file()
    match path.suffix:
        case ".jams":
            return jams.load(path.absolute().as_posix())
        case _:
            trajectory_array = load_f0_trajectory_array_from_path(path)
            return convert_f0_trajectory_array_to_annotation(trajectory_array[:, 0], trajectory_array[:, 1])


def convert_f0_trajectory_array_to_annotation(
    times: FloatArray,
    f0s: FloatArray,
) -> jams.JAMS:
    timestep = times[1] - times[0]
    duration = times[-1] + timestep
    f0_annotation = jams.Annotation(namespace="pitch_contour")
    for time, f0 in zip(times, f0s):
        f0_annotation.append(
            time=time,
            duration=0,
            value={"index": 0, "frequency": float(f0), "voiced": int(f0 != 0)},
            confidence=1,
        )
    return jams.JAMS(annotations=[f0_annotation], file_metadata=FileMetadata(duration=duration))


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
        track_path.joinpath("f0_trajectory_annotation.jams"),
    )


class Track:
    def __init__(
        self,
        name: str,
        audio_source_path: Path,
        settings: Settings,
        f0_source_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._name = name
        self._audio_source_path = audio_source_path
        self._audio_track: FloatArray | None = None
        self._f0_trajectory_annotation: jams.JAMS | None = None
        self._f0_source_path = f0_source_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def audio_source_path(self) -> Path:
        return self._audio_source_path

    @property
    def audio_track(self) -> FloatArray:
        if self._audio_track is None:
            self._audio_track, _ = librosa.load(self._audio_source_path.absolute().as_posix(), sr=self._settings.sr, mono=True)
        return self._audio_track

    @property
    def times_and_freqs(self) -> tuple[FloatArray, FloatArray]:
        times_freqs_dict = {
            time: event["frequency"] for time, event in dict(np.array(self.f0_trajectory_annotation.annotations[0].data)[:, [0, 2]].tolist()).items() if event["frequency"] > 0
        }
        times, freqs = np.array(list(times_freqs_dict.keys())), np.array(list(times_freqs_dict.values()))
        return times, freqs

    @property
    def f0_trajectory_annotation(self) -> jams.JAMS:
        if self._f0_trajectory_annotation is None:
            if self._f0_source_path is not None:
                self._f0_trajectory_annotation = load_f0_trajectory_from_path(self._f0_source_path)
            else:
                times, freqs = self._get_f0_trajectory_from_audio_track()
                self._f0_trajectory_annotation = convert_f0_trajectory_array_to_annotation(times, freqs)
        return self._f0_trajectory_annotation

    def save(self) -> None:
        tracks_path = get_tracks_path(self._settings)
        track_path = tracks_path.joinpath(self.name)
        track_path.mkdir(parents=True, exist_ok=True)
        self.f0_trajectory_annotation.save(track_path.joinpath("f0_trajectory_annotation.jams").absolute().as_posix())
        with open(track_path.joinpath("track_data.json"), "w") as f:
            json.dump({"audio_source_path": self._audio_source_path.absolute().as_posix()}, f)
        with open(track_path.joinpath(".saved"), "a"):
            os.utime(track_path.joinpath(".saved"), None)

    def _save_and_return_pitch_shifted_track(
        self,
        track_name: str,
        shift_suffix: str,
        pitch_shifted_track_jam_with_audio: jams.JAMS,
    ) -> "Track":
        tracks_path = get_tracks_path(self._settings)
        track_path = tracks_path.joinpath(track_name)
        track_path.mkdir(parents=True, exist_ok=True)
        audio_source_path = track_path.joinpath(f"{shift_suffix}{os.path.basename(self._audio_source_path)}")
        trajectory_path = track_path.joinpath("f0_trajectory_annotation.jams")
        wavfile.write(
            audio_source_path,
            self._settings.sr,
            pitch_shifted_track_jam_with_audio.sandbox.muda._audio["y"],
        )
        pitch_shifted_track_jam_with_audio.save(trajectory_path.absolute().as_posix(), strict=True, fmt="auto")
        with open(track_path.joinpath("track_data.json"), "w") as f:
            json.dump({"audio_source_path": audio_source_path.absolute().as_posix()}, f)
        with open(track_path.joinpath(".saved"), "a"):
            os.utime(track_path.joinpath(".saved"), None)
        return load_track(track_name, self._settings)

    def pitch_shift(self, lb: int, ub: int, n_samples: int = 5) -> Iterable["Track"]:
        current_track_jam_with_audio = muda.load_jam_audio(self.f0_trajectory_annotation, self.audio_source_path.absolute().as_posix())
        pitch_shifter = muda.deformers.LinearPitchShift(n_samples=n_samples, lower=lb, upper=ub)
        shift_iterator = pitch_shifter.transform(current_track_jam_with_audio)
        shift_iterator_idx = lb - 1
        for shift_amount in range(lb, ub + 1):
            shift_suffix = f"pitch_shift_{shift_amount}_"
            track_name = f"{shift_suffix}{self._name}"
            if track_is_saved(track_name, self._settings):
                yield load_track(track_name, self._settings)
            else:
                pitch_shifted_track_jam_with_audio = next(shift_iterator)
                shift_iterator_idx += 1
                while shift_iterator_idx < shift_amount:
                    pitch_shifted_track_jam_with_audio = next(shift_iterator)
                    shift_iterator_idx += 1
                yield self._save_and_return_pitch_shifted_track(track_name, shift_suffix, pitch_shifted_track_jam_with_audio)

    def time_shift(self, delay: float) -> "Track":
        shift_suffix = f"time_shift_{delay}_"
        track_name = f"{shift_suffix}{self._name}"
        if track_is_saved(track_name, self._settings):
            return load_track(track_name, self._settings)
        tracks_path = get_tracks_path(self._settings)
        track_path = tracks_path.joinpath(track_name)
        track_path.mkdir(parents=True, exist_ok=True)
        original_f0_annotation = self.f0_trajectory_annotation.annotations[0]
        f0_annotation = jams.Annotation(namespace="pitch_contour")
        for obs in original_f0_annotation:
            f0_annotation.append(
                time=obs.time + delay,
                duration=obs.duration,
                value=obs.value,
                confidence=obs.confidence,
            )

        duration = self.f0_trajectory_annotation.file_metadata.duration + delay
        time_shifted_jams = jams.JAMS(annotations=[f0_annotation], file_metadata=FileMetadata(duration=duration))
        trajectory_path = track_path.joinpath("f0_trajectory_annotation.jams")
        time_shifted_jams.save(trajectory_path.absolute().as_posix())
        time_shifted_track = Track(track_name, self.audio_source_path, self._settings, trajectory_path)
        time_shifted_track.save()
        return time_shifted_track

    def time_shift_by_ir(self, ir: FloatArray) -> "Track":
        delay = median_group_delay(ir, self._settings.sr)
        return self.time_shift(delay)

    def _get_f0_trajectory_from_audio_track(
        self,
    ) -> tuple[FloatArray, FloatArray]:
        f0, voiced_flag, _ = librosa.pyin(y=self.audio_track, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), fill_na=0.0)
        times = librosa.times_like(f0, sr=self._settings.sr)
        for idx, flag in enumerate(voiced_flag):
            if flag is False:
                f0[idx] = 0
        return times, f0

    def __repr__(self) -> str:
        return f"Track({self.name})"

    def __hash__(self) -> int:
        return hash(repr(self))
