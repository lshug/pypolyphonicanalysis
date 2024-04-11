import json
import os
from pathlib import Path

import librosa

from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import load_track, track_is_saved
from pypolyphonicanalysis.settings import Settings


def get_sum_tracks_path(settings: Settings) -> Path:
    sum_tracks_path = Path(os.path.join(settings.data_directory_path, "sum_tracks"))
    sum_tracks_path.mkdir(parents=True, exist_ok=True)
    return sum_tracks_path


def sum_track_is_saved(track_name: str, settings: Settings) -> bool:
    sum_tracks_path = get_sum_tracks_path(settings)
    sum_track_path = sum_tracks_path.joinpath(track_name)
    return sum_track_path.is_dir() and sum_track_path.joinpath(".saved").is_file()


def load_sum_track(sum_track_name: str, settings: Settings) -> "SumTrack":
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
        audio_source_path: Path,
        source_multitrack: Multitrack,
        settings: Settings,
    ) -> None:
        self._name = name
        self._settings = settings
        self._source_multitrack = source_multitrack
        self._audio_source_path = audio_source_path
        self._n_frames: int | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_multitrack(self) -> Multitrack:
        return self._source_multitrack

    @property
    def audio_source_path(self) -> Path:
        return self._audio_source_path

    @property
    def n_frames(self) -> int:
        if self._n_frames is None:
            self._n_frames = int(
                librosa.get_duration(
                    path=self.audio_source_path.absolute().as_posix(),
                    sr=self._settings.sr,
                )
                * self._settings.sr
                // self._settings.hop_length
            )
        return self._n_frames

    def save(self) -> None:
        sum_tracks_path = get_sum_tracks_path(self._settings)
        sum_track_path = sum_tracks_path.joinpath(self.name)
        sum_track_path.mkdir(parents=True, exist_ok=True)
        for track in self._source_multitrack:
            if not track_is_saved(track.name, self._settings):
                track.save()
        with open(sum_track_path.joinpath("sum_track_data.json"), "w") as f:
            json.dump(
                {
                    "audio_source_path": self._audio_source_path.absolute().as_posix(),
                    "source_multitrack": [track.name for track in self.source_multitrack],
                },
                f,
            )
        with open(sum_track_path.joinpath(".saved"), "a"):
            os.utime(sum_track_path.joinpath(".saved"), None)

    def __repr__(self) -> str:
        return f"SumTrack({self.name})"
