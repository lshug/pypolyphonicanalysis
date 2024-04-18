from pathlib import Path

from pypolyphonicanalysis.datamodel.data_multiplexing.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.data_multiplexing.splits import SumTrackSplitType
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
from pypolyphonicanalysis.datamodel.summing_strategies.reverb_sum import ReverbSum
from pypolyphonicanalysis.datamodel.summing_strategies.room_simulation_sum import RoomSimulationSum
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store
from pypolyphonicanalysis.utils.utils import save_train_test_validation_split

settings = Settings()
shuffle = True

feature_store = get_feature_store(settings)

summing_strategies: list[BaseSummingStrategy] = [
    DirectSum(settings),
    ReverbSum(settings),
    RoomSimulationSum(
        settings,
        room_dim_range=((10, 10), (7.5, 8.5), (3.5, 4.5)),
        mic_position_range=((0.2, 0.3), (0.48, 0.52), (0.48, 0.52)),
        source_position_ranges=[
            ((0.39, 0.41), (0.59, 0.61), (0.34, 0.36)),
            ((0.44, 0.64), (0.61, 0.63), (0.39, 0.41)),
            ((0.49, 0.51), (0.62, 0.64), (0.35, 0.38)),
            ((0.54, 0.56), (0.61, 0.63), (0.34, 0.36)),
            ((0.59, 0.61), (0.59, 0.61), (0.39, 0.41)),
        ],
        rt60_range=(0.3, 0.8),
        max_rand_disp_range=(0.03, 0.06),
    ),
]
dataset_loaders: list[BaseDataLoader] = [
    CSDDataloader(shuffle, settings),
    GVMDataLoader(shuffle, settings),
    DCSDataLoader(shuffle, settings),
    ESMUCDataLoader(shuffle, settings),
    CantoriaDataLoader(shuffle, settings),
]
dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]] = [(dl, summing_strategies) for dl in dataset_loaders]
pitch_shift_probabilities = {-2: 0.5, -1.5: 0.5, -1: 0.5, -0.3: 0.5, 0: 1, 0.3: 0.5, 1: 0.5, 1.5: 0.5, 2: 0.5}
sum_track_provider = SumTrackProvider(settings, dataloaders_and_summing_strategies=dataloaders_and_summing_strategies, pitch_shift_probabilities=pitch_shift_probabilities)

split_dict: dict[SumTrackSplitType, list[str]] = {SumTrackSplitType.TRAIN: [], SumTrackSplitType.TEST: [], SumTrackSplitType.VALIDATION: []}

training_metadata = Path(settings.data_directory_path).joinpath("training_metadata")
training_metadata.mkdir(parents=True, exist_ok=True)
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
