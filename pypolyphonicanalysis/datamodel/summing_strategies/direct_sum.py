import logging


from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack

from pypolyphonicanalysis.utils.utils import sum_wave_arrays, FloatArray

logger = logging.getLogger(__name__)


class DirectSum(BaseSummingStrategy):
    def is_summable(self, multitrack: Multitrack) -> bool:
        return True

    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        return f"direct_sum_{'_'.join(track.name for track in multitrack)}"

    def _get_sum(self, multitrack: Multitrack) -> tuple[FloatArray, Multitrack]:
        input_arrays = [track.audio_array for track in multitrack]
        array_sum = sum_wave_arrays(input_arrays)
        return array_sum, multitrack
