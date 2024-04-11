import logging

from pathlib import Path

from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack

from pypolyphonicanalysis.settings import Settings

logging.getLogger("sox").setLevel(logging.ERROR)


class GroundTruthSum(BaseSummingStrategy):
    def __init__(self, ground_truth_paths: dict[frozenset[str], Path], settings: Settings) -> None:
        super().__init__(settings)
        self._ground_truth_paths = ground_truth_paths

    def is_summable(self, multitrack: Multitrack) -> bool:
        return frozenset(track.name for track in multitrack) in self._ground_truth_paths

    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        return f"ground_truth_sum_{'_'.join(track.name for track in multitrack)}"

    def _get_sum(self, multitrack: Multitrack) -> tuple[Path, Multitrack]:
        return (
            self._ground_truth_paths[frozenset(track.name for track in multitrack)],
            multitrack,
        )
