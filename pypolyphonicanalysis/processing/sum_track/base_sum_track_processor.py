import abc
from abc import abstractmethod

from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.settings import Settings


class BaseSumTrackProcessor(abc.ABC):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abstractmethod
    def process(self, sum_track: SumTrack) -> SumTrack:
        pass

    @abstractmethod
    def get_sum_track_name(self, sum_track: SumTrack) -> str:
        pass
