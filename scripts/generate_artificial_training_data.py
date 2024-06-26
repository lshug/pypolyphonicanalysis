import logging
import os
import random
from pathlib import Path

from pypolyphonicanalysis.datamodel.tracks.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.tracks.splits import SumTrackSplitType, TrainTestValidationSplit
from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.monophonic_track_collection_data_loader import MonophonicTrackCollectionDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.recombination_data_loader import RecombinationDataLoader
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.summing_strategies.room_simulation_sum import RoomSimulationSum, RelativePositionRange
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store
from pypolyphonicanalysis.utils.utils import save_train_test_validation_split, get_train_test_validation_split, check_output_path

settings = Settings()
shuffle = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

monophonic_tracks_source_path = Path(settings.data_directory_path).joinpath("corpora").joinpath("monophonic_collections").joinpath("SingingVoiceDataset").joinpath("monophonic")
monophonic_tracks: dict[Path, Path | None] = {
    Path(os.path.join(dirpath, filename)): None for (dirpath, dirnames, filenames) in os.walk(monophonic_tracks_source_path) for filename in filenames if ".wav" in filename
}
feature_store = get_feature_store(settings)

summing_strategies: list[BaseSummingStrategy] = [
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
    )
]
dataset_loaders: list[BaseDataLoader] = [
    RecombinationDataLoader(
        [MonophonicTrackCollectionDataLoader(True, monophonic_tracks, settings)],
        pitch_shift_bounds=(-2, 2),
        settings=settings,
    )
]
dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]] = [(dl, summing_strategies) for dl in dataset_loaders]
sum_track_provider = SumTrackProvider(settings, dataloaders_and_summing_strategies=dataloaders_and_summing_strategies)

split_dict: dict[SumTrackSplitType, list[str]] = {SumTrackSplitType.TRAIN: [], SumTrackSplitType.TEST: [], SumTrackSplitType.VALIDATION: []}

training_metadata = Path(settings.data_directory_path).joinpath("training_metadata")
check_output_path(training_metadata)
count = 0
for sum_track, split in sum_track_provider.get_sum_tracks():
    split_dict[split].append(sum_track.name)
    count += 1
    if count % 10 == 0:
        save_train_test_validation_split(
            "artificial_data_split",
            {"train": split_dict[SumTrackSplitType.TRAIN], "test": split_dict[SumTrackSplitType.TEST], "validation": split_dict[SumTrackSplitType.VALIDATION]},
            settings,
        )

artificial_data_split: TrainTestValidationSplit = {
    "train": split_dict[SumTrackSplitType.TRAIN],
    "test": split_dict[SumTrackSplitType.TEST],
    "validation": split_dict[SumTrackSplitType.VALIDATION],
}
save_train_test_validation_split(
    "artificial_data_split",
    artificial_data_split,
    settings,
)

try:
    base_data_split = get_train_test_validation_split("base_data_split", settings)
    joint_data_split: TrainTestValidationSplit = {
        "train": base_data_split["train"] + artificial_data_split["train"],
        "test": base_data_split["test"] + artificial_data_split["test"],
        "validation": base_data_split["validation"] + artificial_data_split["validation"],
    }
    for split_list in [joint_data_split["train"], joint_data_split["test"], joint_data_split["validation"]]:
        random.shuffle(split_list)
    save_train_test_validation_split(
        "joint_data_split",
        joint_data_split,
        settings,
    )

    base_data_split_nogvmpart = get_train_test_validation_split("base_data_split_nonogvmpart", settings)
    joint_data_split_nogvmpart: TrainTestValidationSplit = {
        "train": base_data_split_nogvmpart["train"] + artificial_data_split["train"],
        "test": base_data_split_nogvmpart["test"] + artificial_data_split["test"],
        "validation": base_data_split_nogvmpart["validation"] + artificial_data_split["validation"],
    }
    for split_list in [joint_data_split_nogvmpart["train"], joint_data_split_nogvmpart["test"], joint_data_split_nogvmpart["validation"]]:
        random.shuffle(split_list)
    save_train_test_validation_split(
        "joint_data_split_nogvmpart",
        joint_data_split,
        settings,
    )
except IOError:
    pass
