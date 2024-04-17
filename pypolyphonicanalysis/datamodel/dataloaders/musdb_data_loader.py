from pathlib import Path
from typing import Iterable, Final

import musdb

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track


class MUSDBDataLoader(BaseDataLoader):
    CORPUS_NAME: Final[str] = "MUSDB18"

    def _get_multitracks(self) -> Iterable[Multitrack]:
        db = musdb.DB(root=self.get_corpus_path(self.CORPUS_NAME).joinpath("WAV"), is_wav=True)
        for track in self._shuffle_if_enabled(list(db)):
            yield Multitrack(
                [
                    Track(
                        name=track.name.lower().replace(" ", "_"),
                        audio_source=Path(track.sources["vocals"].path),
                        settings=self._settings,
                    )
                ]
            )

    def _get_length(self) -> int:
        return 150
