import os
from enum import Enum
from typing import Iterable, Final

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track
from pypolyphonicanalysis.settings import Settings


class GVMMode(Enum):
    LARYNX: str = "ALRX"
    HEADSET: str = "AHDS"


class GVMDataLoader(BaseDataLoader):
    CORPUS_NAME: Final[str] = "GVM"

    def __init__(self, shuffle: bool, settings: Settings, mode: GVMMode = GVMMode.LARYNX, maxlen: int = 60000) -> None:
        super().__init__(shuffle, settings, maxlen)
        self._source_mix_affix: str = str(mode.value)
        file_list = os.listdir(self.get_corpus_path(self.CORPUS_NAME))
        song_prefixes: set[str] = {file.split("_")[0] for file in file_list if ".wav" in file}
        self._song_files_dict: dict[str, list[str]] = {}
        for song_prefix in song_prefixes:
            song_prefix_files = [file for file in file_list if song_prefix in file]
            if all(any(f"{self._source_mix_affix}{voice_idx}M" in file for file in song_prefix_files) for voice_idx in range(1, 4)):
                self._song_files_dict[song_prefix] = song_prefix_files

    def _get_multitracks(self) -> Iterable[Multitrack]:
        corpus_path = self.get_corpus_path(self.CORPUS_NAME)
        for song, files in self._shuffle_if_enabled(list(self._song_files_dict.items())):
            voice_files: tuple[str, ...] = tuple([next(iter(file for file in files if f"{self._source_mix_affix}{voice_idx}M" in file)) for voice_idx in range(1, 4)])
            tracks = [
                Track(
                    name=f"{song}_{self._source_mix_affix}{voice_idx}M",
                    audio_source_path=corpus_path.joinpath(voice_file),
                    settings=self._settings,
                )
                for voice_idx, voice_file in enumerate(voice_files, start=1)
            ]
            yield Multitrack((tracks[0], tracks[1], tracks[2]))

    def _get_length(self) -> int:
        return len(self._song_files_dict)
