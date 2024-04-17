import json
import os
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import load_track, track_is_saved
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray


def get_sum_tracks_path(settings: Settings) -> Path:
    sum_tracks_path = Path(os.path.join(settings.data_directory_path, "sum_tracks"))
    sum_tracks_path.mkdir(parents=True, exist_ok=True)
    return sum_tracks_path


def sum_track_is_saved(track_name: str, settings: Settings) -> bool:
    sum_tracks_path = get_sum_tracks_path(settings)
    sum_track_path = sum_tracks_path.joinpath(track_name)
    return sum_track_path.is_dir() and sum_track_path.joinpath(".saved").is_file()


def load_sum_track(sum_track_name: str, settings: Settings, shallow: bool = False) -> "SumTrack":
    if shallow:
        return SumTrack(sum_track_name, Path(), Multitrack([]), settings)
    sum_tracks_path = get_sum_tracks_path(settings)
    sum_track_path = sum_tracks_path.joinpath(sum_track_name)
    if not sum_track_path.is_dir():
        raise NotADirectoryError
    with open(sum_track_path.joinpath("sum_track_data.json"), "r") as f:
        sum_track_data = json.load(f)
    return SumTrack(
        sum_track_name,
        Path(sum_track_data["audio_source_path"]),
        Multitrack([load_track(track, settings) for track in sum_track_data["source_multitrack"]]),
        settings,
    )


class SumTrack:
    def __init__(
        self,
        name: str,
        audio_source: Path | FloatArray,
        source_multitrack: Multitrack,
        settings: Settings,
    ) -> None:
        self._name = name
        self._settings = settings
        self._source_multitrack = source_multitrack
        self._audio_source = audio_source
        self._audio_array: FloatArray | None = None
        self._n_frames: int | None = None
        if self._settings.save_raw_training_data:
            self.save()

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_multitrack(self) -> Multitrack:
        return self._source_multitrack

    @property
    def audio_array(self) -> FloatArray:
        if self._audio_array is None:
            match self._audio_source:
                case Path():
                    try:
                        self._audio_array = librosa.load(self._audio_source.absolute().as_posix(), sr=self._settings.sr, mono=True)[0]
                    except:
                        raise
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

    def save(self) -> None:
        sum_tracks_path = get_sum_tracks_path(self._settings)
        sum_track_path = sum_tracks_path.joinpath(self.name)
        sum_track_path.mkdir(parents=True, exist_ok=True)
        for track in self._source_multitrack:
            if not track_is_saved(track.name, self._settings):
                track.save()
        source_path: str
        match self._audio_source:
            case Path():
                source_path = self._audio_source.absolute().as_posix()
            case _:
                wavfile.write(sum_track_path.joinpath("audio.wav"), self._settings.sr, self.audio_array)
                source_path = sum_track_path.joinpath("audio.wav").absolute().as_posix()
        with open(sum_track_path.joinpath("sum_track_data.json"), "w") as f:
            json.dump(
                {
                    "audio_source_path": source_path,
                    "source_multitrack": [track.name for track in self.source_multitrack],
                },
                f,
            )
        with open(sum_track_path.joinpath(".saved"), "a"):
            os.utime(sum_track_path.joinpath(".saved"), None)

    def __repr__(self) -> str:
        return f"SumTrack({self.name})"
