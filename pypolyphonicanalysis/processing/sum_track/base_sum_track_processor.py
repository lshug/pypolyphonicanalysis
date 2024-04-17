import abc
from abc import abstractmethod

from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack


class BaseSumTrackProcessor(abc.ABC):
    @abstractmethod
    def process(self, sum_track: SumTrack) -> SumTrack:
        """
        Returns processed times and freqs.
        """

    @abstractmethod
    def get_stage_name(self) -> str:
        """Gets processing stage name of current processor."""
