import json
from pathlib import Path

from joblib import Parallel, delayed

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.csd_data_loader import CSDDataloader
from pypolyphonicanalysis.datamodel.dataloaders.dcs_data_loader import DCSDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.esmuc_data_loader import ESMUCDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.gvm_data_loader import GVMDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.musdb_data_loader import MUSDBDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.recombination_data_loader import RecombinationDataLoader
from pypolyphonicanalysis.datamodel.features.feature_store import FeatureStore
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import (
    BaseSummingStrategy,
)
from pypolyphonicanalysis.datamodel.summing_strategies.direct_sum import DirectSum
from pypolyphonicanalysis.datamodel.summing_strategies.reverb_sum import ReverbSum
from pypolyphonicanalysis.datamodel.summing_strategies.room_simulation_sum import RoomSimulationSum
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.settings import Settings

settings = Settings(test_validation_size=0.4)
shuffle = True

feature_store = FeatureStore(settings)
dataset_loaders: list[BaseDataLoader] = [
    CSDDataloader(shuffle, settings, maxlen=3),
    GVMDataLoader(shuffle, settings, maxlen=3),
    DCSDataLoader(shuffle, settings, maxlen=3),
    ESMUCDataLoader(shuffle, settings, maxlen=3),
    RecombinationDataLoader([MUSDBDataLoader(True, settings, maxlen=3), MUSDBDataLoader(True, settings, maxlen=3)], settings, maxlen=3),
]
summing_strategies: list[BaseSummingStrategy] = [
    DirectSum(settings),
    ReverbSum(settings),
    RoomSimulationSum(settings, (10, 7.5, 3.5), (2.5, 3.73, 1.76), [(4, 4.6, 1.6), (4.5, 4.8, 1.8), (5, 4.85, 1.7), (5.5, 4.8, 1.6), (6, 4.6, 1.9)], rt60=1.0, max_rand_disp=0.5),
]
training_datapoint_names: list[str] = []


def generate_training_points_with_augmentations(multitrack: Multitrack) -> list[str]:
    training_datapoint_names: list[str] = []
    for augmented_multitrack in multitrack.pitch_shift_range(-1, 1):
        for summing_strategy in summing_strategies:
            if summing_strategy.is_summable(augmented_multitrack):
                training_datapoint = summing_strategy.sum_or_retrieve(augmented_multitrack)
                feature_store.generate_all_features_for_sum_track(training_datapoint)
                training_datapoint_names.append(training_datapoint.name)
    return training_datapoint_names


def dump_train_metadata(training_datapoint_names: list[str]) -> None:
    train_points, test_validation_points = train_test_split(training_datapoint_names, test_size=settings.test_validation_size, random_state=settings.random_seed)
    test_points, validation_points = train_test_split(test_validation_points, test_size=settings.validation_proportion, random_state=settings.random_seed)
    training_metadata = Path(settings.data_directory_path).joinpath("training_metadata")
    training_metadata.mkdir(parents=True, exist_ok=True)
    json.dump({"train": train_points, "test": test_points, "validation": validation_points}, open(training_metadata.joinpath("train_test_validation_split.json"), "w"), indent=4)


for dataset_loader in tqdm(dataset_loaders):
    print(type(dataset_loader))
    multitrack_iterator = tqdm(dataset_loader.get_multitracks(), total=len(dataset_loader))
    for datapoint_names in Parallel(n_jobs=32, verbose=5)(delayed(generate_training_points_with_augmentations)(multitrack) for multitrack in multitrack_iterator):
        training_datapoint_names.extend(datapoint_names)
        dump_train_metadata(training_datapoint_names)
