from typing import Iterable, Final, Mapping, NamedTuple

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track


class ESMUCSongData(NamedTuple):
    number_of_takes: int
    sopranos: int
    tenors: int
    altos: int
    basses: int


class ESMUCDataLoader(BaseDataLoader):
    CORPUS_NAME: Final[str] = " EsmucChoirDataset_v1.0.0"
    # Only certain voices in each multitrack recording have dynamic mic recordings available.
    CORPUS_SONG_TAKES_AND_ENSEMBLE: Final[Mapping[str, ESMUCSongData]] = {
        "DG_FT": ESMUCSongData(4, 4, 3, 3, 2),
        "DH1_FT": ESMUCSongData(1, 5, 2, 3, 2),
        "DH2_FT": ESMUCSongData(1, 5, 2, 3, 2),
        "SC1_FT": ESMUCSongData(3, 5, 2, 3, 2),
        "SC2_FT": ESMUCSongData(3, 5, 2, 3, 2),
        "SC3_FT": ESMUCSongData(2, 5, 2, 3, 2),
    }

    def _get_multitracks(self) -> Iterable[Multitrack]:
        corpus_path = self.get_corpus_path(self.CORPUS_NAME)
        file_string_template = "{}_take{}_{}"
        for song, song_data in self._shuffle_if_enabled(list(self.CORPUS_SONG_TAKES_AND_ENSEMBLE.items())):
            for take in self._shuffle_if_enabled(range(1, song_data.number_of_takes + 1)):
                for soprano in [f"S{idx}" for idx in range(song_data.sopranos)]:
                    for alto in [f"A{idx}" for idx in range(song_data.altos)]:
                        for tenor in [f"T{idx}" for idx in range(song_data.tenors)]:
                            for bass in [f"B{idx}" for idx in range(song_data.basses)]:
                                tracks: list[Track] = []
                                for voice in [soprano, alto, tenor, bass]:
                                    track_name = f"{file_string_template.format(song, take, voice)}"
                                    wav_filename = f"{track_name}.wav"
                                    f0_filename = f"{track_name}.f0"
                                    voice_track = Track(
                                        name=track_name,
                                        audio_source_path=corpus_path.joinpath(wav_filename),
                                        settings=self._settings,
                                        f0_source_path=corpus_path.joinpath(f0_filename),
                                    )
                                    tracks.append(voice_track)
                                yield Multitrack((tracks[0], tracks[1], tracks[2], tracks[3]))

    def _get_length(self) -> int:
        return sum(
            song_data.number_of_takes * song_data.sopranos * song_data.altos * song_data.tenors * song_data.basses for song_data in self.CORPUS_SONG_TAKES_AND_ENSEMBLE.values()
        )
