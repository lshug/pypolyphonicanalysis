from typing import Iterable

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track
from pypolyphonicanalysis.settings import Settings


class RecombinationDataLoader(BaseDataLoader):

    def __init__(
        self,
        dataloaders: list[BaseDataLoader],
        settings: Settings,
        tracks_per_multitrack_lb: int = 2,
        tracks_per_multitrack_ub: int = 5,
        pitch_shift_lb: int = 0,
        pitch_shift_ub: int = 0,
        with_replacement: bool = True,
        maxlen: int = 60000,
    ) -> None:
        super().__init__(True, settings, maxlen)
        self._dataloaders = dataloaders
        self._tracks_per_multitrack_lb = tracks_per_multitrack_lb
        self._tracks_per_multitrack_ub = tracks_per_multitrack_ub
        self._pitch_shift_lb = pitch_shift_lb
        self._pitch_shift_ub = pitch_shift_ub
        self._with_replacement = with_replacement
        self._maxlen = maxlen

    def _get_multitracks(self) -> Iterable[Multitrack]:
        dataloader_iters = [iter(dataloader.get_multitracks()) for dataloader in self._dataloaders]
        for _ in range(len(self)):
            loaded_tracks: list[Track] = []
            for idx, loader in enumerate(self._dataloaders):
                try:
                    loaded_tracks.extend([track for track in next(dataloader_iters[idx])])
                except StopIteration:
                    dataloader_iters[idx] = iter(loader.get_multitracks())
                    loaded_tracks.extend([track for track in next(dataloader_iters[idx])])
            number_of_tracks = self._random.randint(self._tracks_per_multitrack_lb, self._tracks_per_multitrack_ub)
            number_of_tracks = max(len(loaded_tracks), number_of_tracks)
            tracks: list[Track] = self._random.choices(loaded_tracks, k=number_of_tracks) if self._with_replacement else self._random.sample(loaded_tracks, k=number_of_tracks)
            for idx, track in enumerate(tracks):
                pitch_shift = self._random.randint(self._pitch_shift_lb, self._pitch_shift_ub)
                if pitch_shift != 0:
                    tracks[idx] = track.pitch_shift(pitch_shift)
            yield Multitrack(tracks)

    def _get_length(self) -> int:
        return min([len(dataloader) for dataloader in self._dataloaders])
