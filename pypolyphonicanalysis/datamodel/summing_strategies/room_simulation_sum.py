from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack

from pypolyphonicanalysis.settings import Settings

import pyroomacoustics as pra
import numpy as np

from pypolyphonicanalysis.utils.utils import FloatArray

Position = tuple[float, float, float]


class RoomSimulationSum(BaseSummingStrategy):
    def __init__(
        self, settings: Settings, room_dim: tuple[float, float, float], mic_position: Position, source_positions: list[Position], rt60: float = 0.3, max_rand_disp: float = 0
    ) -> None:
        super().__init__(settings)
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        self._max_rand_disp = max_rand_disp
        self._room_dim = room_dim
        self._mic_position = mic_position
        self._rt60 = rt60
        self._e_absorption = e_absorption
        self._max_order = max_order
        self._source_positions = source_positions

    def is_summable(self, multitrack: Multitrack) -> bool:
        return True

    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        return f"room_sim_{self._room_dim}_{self._mic_position}_{self._rt60}_{self._max_rand_disp}{'_'.join(track.name for track in multitrack)}"

    def _get_sum(self, multitrack: Multitrack) -> tuple[FloatArray, Multitrack]:
        if len(multitrack) > len(self._source_positions):
            raise ValueError("Multitrack has more tracks than provided source positions")
        room = pra.ShoeBox(
            self._room_dim,
            fs=self._settings.sr,
            materials=pra.Material(self._e_absorption),
            max_order=self._max_order,
            use_rand_ism=self._max_rand_disp != 0,
            max_rand_disp=self._max_rand_disp,
        )
        room.add_microphone_array(np.c_[list(self._mic_position)])
        for idx, track in enumerate(multitrack):
            room.add_source(self._source_positions[idx], signal=track.audio_array)
        room.simulate()

        time_shifted_multitrack = Multitrack([track.time_shift_by_ir(room.rir[0][idx]) for idx, track in enumerate(multitrack)])
        return room.mic_array.signals[0, :], time_shifted_multitrack
