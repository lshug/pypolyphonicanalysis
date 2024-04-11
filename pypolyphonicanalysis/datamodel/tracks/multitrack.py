from typing import Iterable, Iterator

from muda.deformers import median_group_delay

from pypolyphonicanalysis.datamodel.tracks.track import Track
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray


class Multitrack:
    def __init__(self, tracks: Iterable[Track]) -> None:
        self._tracks = tuple(tracks)

    def pitch_shift(self, lb: int, ub: int, n_samples: int = 5) -> Iterable["Multitrack"]:
        voice_iterators = [track.pitch_shift(lb, ub, n_samples) for track in self._tracks]
        for iterators in zip(*voice_iterators):
            yield Multitrack(iterators)

    def time_shift_by_ir(self, ir: FloatArray, settings: Settings) -> "Multitrack":
        delay = median_group_delay(ir, settings.sr)
        return Multitrack([track.time_shift(delay) for track in self._tracks])

    def __contains__(self, item: Track) -> bool:
        return item in self._tracks

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def __getitem__(self, item: int) -> Track:
        return self._tracks[item]
