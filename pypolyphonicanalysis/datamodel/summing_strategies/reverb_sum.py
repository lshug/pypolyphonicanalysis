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
    sum_track_is_saved,
    load_sum_track,
)
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import get_random_state, sum_wave_arrays, FloatArray


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

    def _get_sum(self, multitrack: Multitrack) -> tuple[FloatArray, Multitrack]:
        if sum_track_is_saved(DirectSum(self._settings).get_sum_track_name(multitrack), self._settings):
            noverb_signal = load_sum_track(DirectSum(self._settings).get_sum_track_name(multitrack), self._settings).audio_array
        else:
            noverb_signal = sum_wave_arrays([track.audio_array for track in multitrack])
        convolved_signal_array = scipy.signal.convolve(noverb_signal, self._ir_array)
        time_shifted_multitrack = multitrack.time_shift_by_ir(self._ir_array, self._settings)
        return convolved_signal_array, time_shifted_multitrack
