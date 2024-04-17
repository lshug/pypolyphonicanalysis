from typing import Iterable

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track


class Cantoria(BaseDataLoader):
    csd_corpus_directory_name: str = "CantoriaDataset_v1.0.0"
    csd_song_prefixes = song_prefixes = ["SSS", "RRC", "EJB1", "EJB2", "VBP", "HCB", "LNG", "THM", "CEA", "YSM", "LJT1", "LJT2", "LBM1", "LBM2"]

    def _get_multitracks(self) -> Iterable[Multitrack]:
        corpus_path = self.get_corpus_path(self.csd_corpus_directory_name)
        for song_prefix in self._shuffle_if_enabled(self.csd_song_prefixes):
            track_string_template = f"Cantoria_{song_prefix}_{{}}"
            wav_template = f"{track_string_template}.wav"
            f0_template = f"{track_string_template}.csv"
            yield Multitrack(
                [
                    Track(
                        name=track_string_template.format(voice),
                        audio_source=corpus_path.joinpath("Audio").joinpath(wav_template.format(voice)),
                        settings=self._settings,
                        f0_source=corpus_path.joinpath("F0_pyin").joinpath(f0_template.format(voice)),
                    )
                    for voice in ["S", "A", "T", "B"]
                ]
            )

    def _get_length(self) -> int:
        return 14
