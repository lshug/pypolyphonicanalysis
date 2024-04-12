from pathlib import Path
from typing import Iterable

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track
from pypolyphonicanalysis.settings import Settings


class MonophonicTrackCollectionDataLoader(BaseDataLoader):
    def __init__(self, shuffle: bool, audio_file_annotations_dict: dict[Path, Path | None], settings: Settings, maxlen: int = 60000) -> None:
        super().__init__(shuffle, settings, maxlen)
        self._audio_file_annotations_dict = audio_file_annotations_dict

    def _get_multitracks(self) -> Iterable[Multitrack]:
        for audio_file_path, annotation_path in self._shuffle_if_enabled(list(self._audio_file_annotations_dict.items())):
            yield Multitrack(
                [
                    Track(
                        name=audio_file_path.stem.lower().replace(" ", "_"),
                        audio_source=Path(audio_file_path),
                        settings=self._settings,
                        f0_source=annotation_path,
                    )
                ]
            )

    def _get_length(self) -> int:
        return len(self._audio_file_annotations_dict)
