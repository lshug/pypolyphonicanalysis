import logging
from pathlib import Path

from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack

from pypolyphonicanalysis.datamodel.tracks.sum_track import get_sum_tracks_path
from pypolyphonicanalysis.utils.utils import sum_wav_files

logger = logging.getLogger(__name__)


class DirectSum(BaseSummingStrategy):
    def is_summable(self, multitrack: Multitrack) -> bool:
        return True

    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        return f"direct_sum_{'_'.join(track.name for track in multitrack)}"

    def _get_sum(self, multitrack: Multitrack) -> tuple[Path, Multitrack]:
        sum_tracks_path = get_sum_tracks_path(self._settings)
        sum_track_name = self.get_sum_track_name(multitrack)
        sum_directory_path = sum_tracks_path.joinpath(sum_track_name)
        sum_directory_path.mkdir(parents=True, exist_ok=True)
        sum_source_audio_path = sum_directory_path.joinpath("sum.wav")
        input_files = [track.audio_source_path for track in multitrack]
        sum_wav_files(input_files, sum_source_audio_path, self._settings)
        return sum_source_audio_path, multitrack
