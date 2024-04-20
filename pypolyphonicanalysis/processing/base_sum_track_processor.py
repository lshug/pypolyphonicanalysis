import abc

from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack, sum_track_is_saved, load_sum_track
from pypolyphonicanalysis.settings import Settings


class BaseSumTrackProcessor(abc.ABC):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abc.abstractmethod
    def _process(self, sum_track: SumTrack) -> SumTrack:
        pass

    @abc.abstractmethod
    def get_sum_track_name_from_base_sumtrack_name(self, sum_track_name: str) -> str:
        pass

    def get_sum_track_name(self, sum_track: SumTrack) -> str:
        return self.get_sum_track_name_from_base_sumtrack_name(sum_track.name)

    def process_or_load(self, sum_track: SumTrack) -> SumTrack:
        if sum_track_is_saved(self.get_sum_track_name(sum_track), self._settings):
            return load_sum_track(self.get_sum_track_name(sum_track), self._settings)
        return self._process(sum_track)
