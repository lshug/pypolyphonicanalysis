import logging
from pathlib import Path

from pypolyphonicanalysis.datamodel.summing_strategies.reverb_sum import ReverbSum
from pypolyphonicanalysis.datamodel.tracks.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.tracks.splits import SumTrackSplitType
from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.cantoria_data_loader import CantoriaDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.csd_data_loader import CSDDataloader
from pypolyphonicanalysis.datamodel.dataloaders.dcs_data_loader import DCSDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.esmuc_data_loader import ESMUCDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.gvm_data_loader import GVMDataLoader
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.summing_strategies.direct_sum import DirectSum
from pypolyphonicanalysis.datamodel.summing_strategies.room_simulation_sum import RoomSimulationSum, RelativePositionRange
from pypolyphonicanalysis.datamodel.tracks.sum_track_processing.add_noise import AddNoise
from pypolyphonicanalysis.datamodel.tracks.sum_track_processing.base_sum_track_processor import BaseSumTrackProcessor
from pypolyphonicanalysis.datamodel.tracks.sum_track_processing.distort import Distort
from pypolyphonicanalysis.datamodel.tracks.sum_track_processing.filter import Filter
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store
from pypolyphonicanalysis.utils.utils import save_train_test_validation_split, check_output_path

settings = Settings()
shuffle = True


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


feature_store = get_feature_store(settings)

summing_strategies: list[BaseSummingStrategy] = [
    DirectSum(settings),
    RoomSimulationSum(
        settings,
        room_dim_range=((10, 10), (7.5, 8.5), (3.5, 4.5)),
        mic_position_range=RelativePositionRange(((0.2, 0.3), (0.48, 0.52), (0.48, 0.52))),
        source_position_ranges=[
            RelativePositionRange(((0.39, 0.41), (0.59, 0.61), (0.34, 0.36))),
            RelativePositionRange(((0.44, 0.64), (0.61, 0.63), (0.39, 0.41))),
            RelativePositionRange(((0.49, 0.51), (0.62, 0.64), (0.35, 0.38))),
            RelativePositionRange(((0.54, 0.56), (0.61, 0.63), (0.34, 0.36))),
            RelativePositionRange(((0.59, 0.61), (0.59, 0.61), (0.39, 0.41))),
        ],
        rt60_range=(0.3, 0.8),
        max_rand_disp_rel_range=(0.03, 0.06),
    ),
    ReverbSum(settings),
]
dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]] = [
    (CSDDataloader(shuffle, settings), summing_strategies[:2]),
    (GVMDataLoader(shuffle, settings), summing_strategies[:2]),
    (DCSDataLoader(shuffle, settings), summing_strategies),
    (ESMUCDataLoader(shuffle, settings), summing_strategies[:2]),
    (CantoriaDataLoader(shuffle, settings), summing_strategies),
]
pitch_shift_probabilities: dict[float, float] = {-1.5: 1, -0.5: 1, 0.5: 1, 1.5: 1}
pitch_shift_displacement_range = (-0.5, 0.5)
sum_track_processors: list[BaseSumTrackProcessor] = [AddNoise(settings), Filter(settings), Distort(settings)]
sum_track_provider = SumTrackProvider(
    settings,
    dataloaders_and_summing_strategies=dataloaders_and_summing_strategies,
    pitch_shift_probabilities=pitch_shift_probabilities,
    pitch_shift_displacement_range=pitch_shift_displacement_range,
    sum_track_processors=sum_track_processors,
)

split_dict: dict[SumTrackSplitType, list[str]] = {SumTrackSplitType.TRAIN: [], SumTrackSplitType.TEST: [], SumTrackSplitType.VALIDATION: []}

training_metadata = Path(settings.data_directory_path).joinpath("training_metadata")
check_output_path(training_metadata)
count = 0
for sum_track, split in sum_track_provider.get_sum_tracks():
    split_dict[split].append(sum_track.name)
    count += 1
    if count % 10 == 0:
        save_train_test_validation_split(
            "base_data_split",
            {"train": split_dict[SumTrackSplitType.TRAIN], "test": split_dict[SumTrackSplitType.TEST], "validation": split_dict[SumTrackSplitType.VALIDATION]},
            settings,
        )
save_train_test_validation_split(
    "base_data_split",
    {"train": split_dict[SumTrackSplitType.TRAIN], "test": split_dict[SumTrackSplitType.TEST], "validation": split_dict[SumTrackSplitType.VALIDATION]},
    settings,
)


even_numbered_GVM_indicies = [f"GVM{str(idx).zfill(3)}" for idx in range(1, len(GVMDataLoader(shuffle, settings)) + 1, 8)]


def filter_gvm_sum_tracks(sum_tracks: list[str]) -> list[str]:
    return [sum_track for sum_track in sum_tracks if not any(gvm_index in sum_track for gvm_index in even_numbered_GVM_indicies)]


save_train_test_validation_split(
    "base_data_split_nogvmpart",
    {
        "train": filter_gvm_sum_tracks(split_dict[SumTrackSplitType.TRAIN]),
        "test": filter_gvm_sum_tracks(split_dict[SumTrackSplitType.TEST]),
        "validation": filter_gvm_sum_tracks(split_dict[SumTrackSplitType.VALIDATION]),
    },
    settings,
)
