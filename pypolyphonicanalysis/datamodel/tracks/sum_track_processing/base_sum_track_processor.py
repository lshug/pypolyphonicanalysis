import abc
import logging
from abc import abstractmethod

from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack, sum_track_is_saved, load_sum_track
from pypolyphonicanalysis.settings import Settings

logger = logging.getLogger(__name__)


class BaseSumTrackProcessor(abc.ABC):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def process_or_load(self, sum_track: SumTrack) -> SumTrack:
        name = self.get_sum_track_name(sum_track)
        if sum_track_is_saved(name, self._settings):
            logger.info(f"Loading saved processed sum track {name}")
            return load_sum_track(name, self._settings)
        logger.info(f"Generating processed sum track {name}")
        return self._process(sum_track)

    @abstractmethod
    def _process(self, sum_track: SumTrack) -> SumTrack:
        pass

    def get_sum_track_name(self, sum_track: SumTrack) -> str:
        return self.get_sum_track_name_from_base_sumtrack_name(sum_track.name)

    @abstractmethod
    def get_sum_track_name_from_base_sumtrack_name(self, sum_track_name: str) -> str:
        pass
