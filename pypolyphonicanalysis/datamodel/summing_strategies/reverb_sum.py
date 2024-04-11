import os
from pathlib import Path

import librosa
import scipy

from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.summing_strategies.direct_sum import DirectSum
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack

from pypolyphonicanalysis.datamodel.tracks.sum_track import (
    get_sum_tracks_path,
    sum_track_is_saved,
    load_sum_track,
)
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import get_random_state, sum_wav_files


def get_irs_path(settings: Settings) -> Path:
    irs_path = Path(settings.data_directory_path).joinpath("irs")
    irs_path.mkdir(parents=True, exist_ok=True)
    return irs_path


class ReverbSum(BaseSummingStrategy):
    def __init__(self, settings: Settings, ir_file: str | None = None) -> None:
        super().__init__(settings)
        if ir_file is not None and get_irs_path(settings).joinpath(ir_file).is_file():
            self._ir_file = ir_file
        else:
            self._ir_file = self._get_random_ir_file()
        self._ir_array, _ = librosa.load(
            get_irs_path(self._settings).joinpath(self._ir_file).absolute().as_posix(),
            sr=self._settings.sr,
        )

    def _get_random_ir_file(self) -> str:
        irs_path = get_irs_path(self._settings)
        random_state = get_random_state(self._settings)
        ir_files = os.listdir(irs_path)
        random_ir_file = str(random_state.choice(ir_files, 1)[0])
        return random_ir_file

    def is_summable(self, multitrack: Multitrack) -> bool:
        return True

    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        return f"reverb_sum_{self._ir_file}_{'_'.join(track.name for track in multitrack)}"

    def _get_sum(self, multitrack: Multitrack) -> tuple[Path, Multitrack]:
        sum_tracks_path = get_sum_tracks_path(self._settings)
        sum_track_name = self.get_sum_track_name(multitrack)
        sum_directory_path = sum_tracks_path.joinpath(sum_track_name)
        sum_directory_path.mkdir(parents=True, exist_ok=True)
        if sum_track_is_saved(DirectSum(self._settings).get_sum_track_name(multitrack), self._settings):
            sum_noverb_audio_path = load_sum_track(DirectSum(self._settings).get_sum_track_name(multitrack), self._settings).audio_source_path
        else:
            sum_noverb_audio_path = sum_directory_path.joinpath("sum_noverb.wav")
            input_files = [track.audio_source_path for track in multitrack]
            sum_wav_files(input_files, sum_noverb_audio_path, self._settings)
        noverb_signal, _ = librosa.load(sum_noverb_audio_path.absolute().as_posix())
        convolved_signal_array = scipy.signal.convolve(noverb_signal, self._ir_array)
        sum_source_audio_path = sum_directory_path.joinpath("sum.wav")
        scipy.io.wavfile.write(sum_source_audio_path, self._settings.sr, convolved_signal_array)
        time_shifted_multitrack = multitrack.time_shift_by_ir(self._ir_array, self._settings)
        return sum_source_audio_path, time_shifted_multitrack
