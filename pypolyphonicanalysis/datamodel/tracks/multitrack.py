from typing import Iterable, Iterator

from pypolyphonicanalysis.datamodel.tracks.track import Track
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray, median_group_delay


class Multitrack:
    def __init__(self, tracks: Iterable[Track]) -> None:
        self._tracks = tuple(tracks)
        self._untrimmed_tracks: tuple[Track, ...] = tuple()
        if len(self._tracks) > 0:
            self._trim_tracks()

    def save(self) -> None:
        for track in self._tracks:
            track.save()
        for track in self._untrimmed_tracks:
            if track.settings.save_multitrack_tracks_pre_trimming:
                track.save()

    def _trim_tracks(self) -> None:
        min_frames = min(track.n_frames for track in self._tracks)
        tracks = tuple([track.trim_to_frames(min_frames) for track in self._tracks])
        untrimmed_tracks: list[Track] = []
        for track in self._tracks:
            if track not in tracks:
                untrimmed_tracks.append(track)
        self._tracks = tracks
        self._untrimmed_tracks = tuple(untrimmed_tracks)

    def pitch_shift(self, n_steps: float) -> "Multitrack":
        return Multitrack([track.pitch_shift(n_steps) for track in self._tracks])

    def time_shift_by_ir(self, ir: FloatArray, settings: Settings) -> "Multitrack":
        delay = median_group_delay(ir, settings.sr)
        return Multitrack([track.time_shift(delay) for track in self._tracks])

    def __repr__(self) -> str:
        return f"Multitrack([{','.join(track.name for track in self._tracks)}])"

    def __contains__(self, item: Track) -> bool:
        return item in self._tracks

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def __getitem__(self, item: int) -> Track:
        return self._tracks[item]
