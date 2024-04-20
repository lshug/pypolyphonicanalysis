import abc
import logging
from abc import abstractmethod

from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.splits import SumTrackSplitType
from pypolyphonicanalysis.datamodel.tracks.sum_track import (
    SumTrack,
    load_sum_track,
    sum_track_is_saved,
)
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray

logger = logging.getLogger(__name__)


class BaseSummingStrategy(abc.ABC):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abstractmethod
    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        """Gets the name of the summed file to be generated."""

    @abstractmethod
    def _get_sum(self, multitrack: Multitrack) -> tuple[FloatArray, Multitrack]:
        """Sums the multitrack and returns a tuple of the path to the summed audio file and processed multitrack"""

    def _trim_sum_track_to_minimum_duration(self, sum_track: SumTrack) -> SumTrack:
        return sum_track

    def sum_or_retrieve(self, multitrack: Multitrack) -> SumTrack:
        """Sums the multitrack and returns a tuple of the summed track and processed multitrack"""
        sum_name = self.get_sum_track_name(multitrack)
        if sum_track_is_saved(sum_name, self._settings):
            logger.debug(f"Loading saved sum track {sum_name}")
            return load_sum_track(sum_name, self._settings)
        else:
            logger.debug(f"Summing {multitrack} with {self}")
            sum_audio_array, multitrack = self._get_sum(multitrack)
            sum_track = SumTrack(sum_name, sum_audio_array, multitrack, self._settings)
            return sum_track

    @abstractmethod
    def is_summable(self, multitrack: Multitrack) -> bool:
        """Returns a boolean value indicating whether the strategy is able to sum the given multitrack."""

    @property
    def split_override(self) -> SumTrackSplitType | None:
        return None
