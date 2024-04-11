from pathlib import Path
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.sum_track import get_sum_tracks_path

from pypolyphonicanalysis.settings import Settings

import pyroomacoustics as pra
import numpy as np

Position = tuple[float, float, float]


class RoomSimulationSum(BaseSummingStrategy):
    def __init__(
        self, settings: Settings, room_dim: tuple[float, float, float], mic_position: Position, source_positions: list[Position], rt60: float = 0.3, perturb_positions: bool = False
    ) -> None:
        super().__init__(settings)
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        self._preturb_positions = perturb_positions
        self._room_dim = room_dim
        self._mic_position = self._perturb_position_if_selected(mic_position)
        self._rt60 = rt60
        self._e_absorption = e_absorption
        self._max_order = max_order
        self._source_positions = source_positions

    def is_summable(self, multitrack: Multitrack) -> bool:
        return True

    def _perturb_position_if_selected(self, position: Position) -> Position:
        return position if self._preturb_positions else position

    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        return f"room_sim_{self._room_dim}_{self._mic_position}_{self._rt60}_{'perturb_' if self._preturb_positions else ''}{'_'.join(track.name for track in multitrack)}"

    def _get_sum(self, multitrack: Multitrack) -> tuple[Path, Multitrack]:
        if len(multitrack) > len(self._source_positions):
            raise ValueError("Multitrack has more tracks than provided source positions")
        sum_tracks_path = get_sum_tracks_path(self._settings)
        sum_track_name = self.get_sum_track_name(multitrack)
        sum_directory_path = sum_tracks_path.joinpath(sum_track_name)
        sum_directory_path.mkdir(parents=True, exist_ok=True)
        sum_source_audio_path = sum_directory_path.joinpath("sum.wav")

        room = pra.ShoeBox(self._room_dim, fs=self._settings.sr, materials=pra.Material(self._e_absorption), max_order=self._max_order)
        room.add_microphone_array(np.c_[list(self._mic_position)])
        source_positions = [self._perturb_position_if_selected(pos) for pos in self._source_positions]
        for idx, track in enumerate(multitrack):
            room.add_source(source_positions[idx], signal=track.audio_track)
        room.simulate()
        room.mic_array.to_wav(sum_source_audio_path.absolute().as_posix(), mono=True)
        time_shifted_multitrack = Multitrack([track.time_shift_by_ir(room.rir[0][idx]) for idx, track in enumerate(multitrack)])
        return sum_source_audio_path, time_shifted_multitrack
