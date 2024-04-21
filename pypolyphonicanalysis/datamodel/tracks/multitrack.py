from typing import Iterable, Iterator

from pypolyphonicanalysis.datamodel.tracks.track import Track
from pypolyphonicanalysis.datamodel.tracks.track_utils import MultitrackAlignmentStrategy
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray, median_group_delay


class Multitrack:
    def __init__(self, tracks: Iterable[Track]) -> None:
        self._aligned_tracks: tuple[Track, ...] | None = None
        self._unaligned_tracks: tuple[Track, ...] = tuple(tracks)

    @property
    def _tracks(self) -> tuple[Track, ...]:
        if self._aligned_tracks is None:
            if len(self._unaligned_tracks) > 0:
                self._aligned_tracks = self._align_tracks()
            else:
                self._aligned_tracks = tuple()
            self._unaligned_tracks = tuple(track for track in self._unaligned_tracks if track not in self._aligned_tracks)
        return self._aligned_tracks

    def save(self) -> None:
        for track in self._tracks:
            track.save()
        for track in self._unaligned_tracks:
            if track.settings.save_multitrack_tracks_pre_alignment:
                track.save()

    def _align_tracks(self) -> tuple[Track, ...]:
        settings = self._unaligned_tracks[0].settings
        max_frames = max(track.n_frames for track in self._unaligned_tracks)
        min_frames = min(track.n_frames for track in self._unaligned_tracks)
        n_frames = min_frames if settings.multitrack_alignment_strategy is MultitrackAlignmentStrategy.TRIM else max_frames
        tracks = tuple([track.align(n_frames, settings.multitrack_alignment_strategy) for track in self._unaligned_tracks])
        return tracks

    def pitch_shift(self, n_steps: float) -> "Multitrack":
        return Multitrack([track.pitch_shift(n_steps) for track in self._tracks])

    def time_shift_by_ir(self, ir: FloatArray, settings: Settings) -> "Multitrack":
        delay = median_group_delay(ir, settings.sr)
        return Multitrack([track.time_shift(delay) for track in self._tracks])

    def __repr__(self) -> str:
        return f"Multitrack([{','.join(track.name for track in self._unaligned_tracks)}])"

    def __contains__(self, item: Track) -> bool:
        return item in self._tracks

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def __getitem__(self, item: int) -> Track:
        return self._tracks[item]
