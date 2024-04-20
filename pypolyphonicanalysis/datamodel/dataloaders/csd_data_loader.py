from typing import Iterable

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.track import Track


class CSDDataloader(BaseDataLoader):
    corpus_directory_name: str = "ChoralSingingDataset"
    song_prefixes = ["CSD_ER_", "CSD_LI_", "CSD_ND_"]

    def _get_multitracks(self) -> Iterable[Multitrack]:
        corpus_path = self.get_corpus_path(self.corpus_directory_name)
        for song_prefix in self._shuffle_if_enabled(self.song_prefixes):
            track_string_template = f"{song_prefix}{{}}_{{}}"
            wav_template = f"{track_string_template}.wav"
            f0_template = f"{track_string_template}.f0"
            voice_tracks: list[list[Track]] = []
            for voice in ["soprano", "alto", "tenor", "bass"]:
                voice_tracks.append(
                    [
                        Track(
                            name=track_string_template.format(voice, idx),
                            audio_source=corpus_path.joinpath(wav_template.format(voice, idx)),
                            settings=self._settings,
                            f0_source=corpus_path.joinpath(f0_template.format(voice, idx)),
                        )
                        for idx in self._shuffle_if_enabled(range(1, 5))
                    ]
                )
            for soprano in voice_tracks[0]:
                for alto in voice_tracks[1]:
                    for tenor in voice_tracks[2]:
                        for bass in voice_tracks[3]:
                            yield Multitrack((soprano, alto, tenor, bass))

    def _get_length(self) -> int:
        number_of_songs = len(self.song_prefixes)
        number_of_voices = 4
        number_of_variations_per_voice = 4
        return number_of_songs * int(number_of_voices**number_of_variations_per_voice)
