import abc
from abc import abstractmethod
from pathlib import Path

from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.sum_track import (
    SumTrack,
    load_sum_track,
    sum_track_is_saved,
)
from pypolyphonicanalysis.datamodel.tracks.track import track_is_saved
from pypolyphonicanalysis.settings import Settings


class BaseSummingStrategy(abc.ABC):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abstractmethod
    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        """Gets the name of the summed file to be generated."""

    @abstractmethod
    def _get_sum(self, multitrack: Multitrack) -> tuple[Path, Multitrack]:
        """Sums the multitrack and returns a tuple of the path to the summed audio file and processed multitrack"""

    def sum_or_retrieve(self, multitrack: Multitrack) -> SumTrack:
        """Sums the multitrack and returns a tuple of the summed track and processed multitrack"""
        sum_name = self.get_sum_track_name(multitrack)
        if sum_track_is_saved(sum_name, self._settings):
            return load_sum_track(sum_name, self._settings)
        else:
            for track in multitrack:
                if not track_is_saved(track.name, self._settings):
                    track.save()
            audio_source_path, multitrack = self._get_sum(multitrack)
            sum_track = SumTrack(sum_name, audio_source_path, multitrack, self._settings)
            sum_track.save()
            return sum_track

    @abstractmethod
    def is_summable(self, multitrack: Multitrack) -> bool:
        """Returns a boolean value indicating whether the strategy is able to sum the given multitrack."""
