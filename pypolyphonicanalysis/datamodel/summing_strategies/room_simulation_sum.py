from typing import TypeVar, NewType

from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack

from pypolyphonicanalysis.settings import Settings

import pyroomacoustics as pra
import numpy as np

from pypolyphonicanalysis.utils.utils import FloatArray, get_random_number_generator

Position = tuple[float, float, float]
RelativePosition = NewType("RelativePosition", Position)
Range = tuple[float, float]
PositionRange = tuple[Range, Range, Range]
RelativePositionRange = NewType("RelativePositionRange", PositionRange)

PositionType = TypeVar("PositionType", bound=Position)


def range_average(range: Range) -> float:
    return (range[0] + range[1]) / 2


def position_range_average(position_range: PositionRange) -> Position:
    return range_average(position_range[0]), range_average(position_range[1]), range_average(position_range[2])


class RoomSimulationSum(BaseSummingStrategy):
    def __init__(
        self,
        settings: Settings,
        room_dim_range: PositionRange,
        mic_position_range: RelativePositionRange,
        source_position_ranges: list[RelativePositionRange],
        rt60_range: Range = (0.1, 0.5),
        max_rand_disp_rel_range: Range = (0, 0.1),
    ) -> None:
        super().__init__(settings)
        self._rng = get_random_number_generator(settings)
        self._room_dim_range = room_dim_range
        self._max_rand_disp_range = max_rand_disp_rel_range
        self._mic_position_range = mic_position_range
        self._rt60_range = rt60_range
        self._source_position_ranges = source_position_ranges

    def is_summable(self, multitrack: Multitrack) -> bool:
        return True

    def get_sum_track_name(self, multitrack: Multitrack) -> str:
        return f"room_sim_{position_range_average(self._room_dim_range)}_{position_range_average(self._mic_position_range)}_{range_average(self._rt60_range)}_{range_average(self._max_rand_disp_range)}{'_'.join(track.name for track in multitrack)}"

    def _get_float_from_range(self, range: Range) -> float:
        return self._rng.uniform(range[0], range[1])

    def _get_position_from_position_range(self, position_range: PositionRange) -> Position:
        return (
            self._rng.uniform(position_range[0][0], position_range[0][1]),
            self._rng.uniform(position_range[1][0], position_range[1][1]),
            self._rng.uniform(position_range[2][0], position_range[2][1]),
        )

    def _get_relative_position_from_relative_position_range(self, relative_position_range: RelativePositionRange) -> RelativePosition:
        return RelativePosition(
            (
                self._rng.uniform(relative_position_range[0][0], relative_position_range[0][1]),
                self._rng.uniform(relative_position_range[1][0], relative_position_range[1][1]),
                self._rng.uniform(relative_position_range[2][0], relative_position_range[2][1]),
            )
        )

    def _get_position_from_relative_position(self, relative_position: RelativePosition, room_dim: Position) -> Position:
        return (relative_position[0] * room_dim[0], relative_position[1] * room_dim[1], relative_position[2] * room_dim[2])

    def _get_position_from_relative_position_range(self, relative_position_range: RelativePositionRange, room_dim: Position) -> Position:
        return self._get_position_from_relative_position(self._get_relative_position_from_relative_position_range(relative_position_range), room_dim)

    def _get_sum(self, multitrack: Multitrack) -> tuple[FloatArray, Multitrack]:
        if len(multitrack) > len(self._source_position_ranges):
            raise ValueError("Multitrack has more tracks than provided source positions")
        room_dim = self._get_position_from_position_range(self._room_dim_range)
        e_absorptions: float
        max_order: int
        e_absorptions, max_order = pra.inverse_sabine(self._get_float_from_range(self._rt60_range), room_dim)
        max_rand_disp = self._rng.uniform(self._max_rand_disp_range[0], self._max_rand_disp_range[1]) * min(room_dim)
        room = pra.ShoeBox(
            room_dim,
            fs=self._settings.sr,
            materials=pra.Material(e_absorptions),
            max_order=max_order,
            use_rand_ism=max_rand_disp != 0,
            max_rand_disp=max_rand_disp,
        )

        room.add_microphone_array(np.c_[list(self._get_position_from_relative_position_range(self._mic_position_range, room_dim))])
        for idx, track in enumerate(multitrack):
            room.add_source(self._get_position_from_relative_position_range(self._source_position_ranges[idx], room_dim), signal=track.audio_array)
        room.simulate()

        time_shifted_multitrack = Multitrack([track.time_shift_by_ir(room.rir[0][idx]) for idx, track in enumerate(multitrack)])
        return room.mic_array.signals[0, :], time_shifted_multitrack
