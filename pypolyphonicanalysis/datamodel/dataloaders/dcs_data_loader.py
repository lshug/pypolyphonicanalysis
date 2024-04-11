from typing import Iterable, Final, Mapping, NamedTuple

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track


class DCSSongData(NamedTuple):
    number_of_takes: int
    soprano: str
    tenor: str
    alto: str
    bass: str


class DCSDataLoader(BaseDataLoader):
    CORPUS_NAME: Final[str] = "DagstuhlChoirSet_V1.2.3"
    # Only certain voices in each multitrack recording have dynamic mic recordings available.
    CORPUS_SONG_TAKES_AND_ENSEMBLE: Final[Mapping[str, DCSSongData]] = {
        "DCS_LI_FullChoir": DCSSongData(3, "S1", "A2", "T2", "B2"),
        "DCS_LI_QuartetA": DCSSongData(6, "S2", "A1", "T1", "B1"),
        "DCS_LI_QuartetB": DCSSongData(5, "S1", "A2", "T2", "B2"),
        "DCS_TP_FullChoir": DCSSongData(4, "S1", "A2", "T2", "B2"),
        "DCS_TP_QuartetA": DCSSongData(2, "S2", "A1", "T1", "B1"),
    }
    MICROPHONE_SUFFIX: Final[str] = "DYN"

    def _get_multitracks(self) -> Iterable[Multitrack]:
        corpus_path = self.get_corpus_path(self.CORPUS_NAME)
        file_string_template = f"{{}}_Take0{{}}_{{}}_{self.MICROPHONE_SUFFIX}"
        for song, song_data in self._shuffle_if_enabled(list(self.CORPUS_SONG_TAKES_AND_ENSEMBLE.items())):
            for take in self._shuffle_if_enabled(range(1, song_data.number_of_takes + 1)):
                tracks: list[Track] = []
                for voice in [
                    song_data.soprano,
                    song_data.alto,
                    song_data.tenor,
                    song_data.bass,
                ]:
                    track_name = file_string_template.format(song, take, voice)
                    wav_filename = f"{track_name}.wav"
                    f0_filename = f"{track_name}.f0"
                    voice_track = Track(
                        name=track_name,
                        audio_source_path=corpus_path.joinpath("audio_wav_22050_mono").joinpath(wav_filename),
                        settings=self._settings,
                        f0_source_path=corpus_path.joinpath("annotations_csv_F0_PYIN").joinpath(f0_filename),
                    )
                    tracks.append(voice_track)
                yield Multitrack((tracks[0], tracks[1], tracks[2], tracks[3]))

    def _get_length(self) -> int:
        return sum(song_data.number_of_takes for song_data in self.CORPUS_SONG_TAKES_AND_ENSEMBLE.values())
