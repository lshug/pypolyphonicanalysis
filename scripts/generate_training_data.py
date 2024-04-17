import json
import os
from pathlib import Path

from pypolyphonicanalysis.datamodel.data_multiplexing.sum_track_provider import SumTrackProvider, SumTrackSplitType
from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.csd_data_loader import CSDDataloader
from pypolyphonicanalysis.datamodel.dataloaders.dcs_data_loader import DCSDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.esmuc_data_loader import ESMUCDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.gvm_data_loader import GVMDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.monophonic_track_collection_data_loader import MonophonicTrackCollectionDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.recombination_data_loader import RecombinationDataLoader
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.summing_strategies.direct_sum import DirectSum
from pypolyphonicanalysis.datamodel.summing_strategies.reverb_sum import ReverbSum
from pypolyphonicanalysis.datamodel.summing_strategies.room_simulation_sum import RoomSimulationSum
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store

settings = Settings(test_validation_size=0.4)
shuffle = True

monophonic_tracks_source_path = Path(settings.data_directory_path).joinpath("corpora").joinpath("monophonic_collections").joinpath("SingingVoiceDataset").joinpath("monophonic")
monophonic_tracks: dict[Path, Path | None] = {
    Path(os.path.join(dirpath, filename)): None for (dirpath, dirnames, filenames) in os.walk(monophonic_tracks_source_path) for filename in filenames if ".wav" in filename
}

feature_store = get_feature_store(settings)
summing_strategies: list[BaseSummingStrategy] = [
    DirectSum(settings),
    ReverbSum(settings),
    RoomSimulationSum(settings, (10, 7.5, 3.5), (2.5, 3.73, 1.76), [(4, 4.6, 1.6), (4.5, 4.8, 1.8), (5, 4.85, 1.7), (5.5, 4.8, 1.6), (6, 4.6, 1.9)], rt60=1.0, max_rand_disp=0.5),
]
dataset_loaders: list[BaseDataLoader] = [
    CSDDataloader(shuffle, settings, maxlen=3),
    GVMDataLoader(shuffle, settings, maxlen=3),
    DCSDataLoader(shuffle, settings, maxlen=3),
    ESMUCDataLoader(shuffle, settings, maxlen=3),
    RecombinationDataLoader(
        [MonophonicTrackCollectionDataLoader(True, monophonic_tracks, settings, maxlen=3), MonophonicTrackCollectionDataLoader(True, monophonic_tracks, settings, maxlen=3)],
        settings,
        maxlen=3,
    ),
]
dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]] = [(dl, summing_strategies) for dl in dataset_loaders]
sum_track_provider = SumTrackProvider(settings, dataloaders_and_summing_strategies=dataloaders_and_summing_strategies)

split_dict: dict[SumTrackSplitType, list[str]] = {SumTrackSplitType.TRAIN: [], SumTrackSplitType.TEST: [], SumTrackSplitType.VALIDATION: []}

training_metadata = Path(settings.data_directory_path).joinpath("training_metadata")
training_metadata.mkdir(parents=True, exist_ok=True)
count = 0
for sum_track, split in sum_track_provider.get_sum_tracks():
    split_dict[split].append(sum_track.name)
    count += 1
    if count % 100 == 0:
        json.dump(
            {"train": split_dict[SumTrackSplitType.TRAIN], "test": split_dict[SumTrackSplitType.TEST], "validation": split_dict[SumTrackSplitType.VALIDATION]},
            open(training_metadata.joinpath("train_test_validation_split.json"), "w"),
            indent=4,
        )
