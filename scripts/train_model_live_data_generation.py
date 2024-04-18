import logging
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from pypolyphonicanalysis.datamodel.data_multiplexing.sum_track_feature_stream_mux import SumTrackFeatureStreamMux
from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.csd_data_loader import CSDDataloader
from pypolyphonicanalysis.datamodel.dataloaders.dcs_data_loader import DCSDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.esmuc_data_loader import ESMUCDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.gvm_data_loader import GVMDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.monophonic_track_collection_data_loader import MonophonicTrackCollectionDataLoader
from pypolyphonicanalysis.datamodel.dataloaders.recombination_data_loader import RecombinationDataLoader
from pypolyphonicanalysis.datamodel.features.features import Features
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import BaseSummingStrategy
from pypolyphonicanalysis.datamodel.summing_strategies.direct_sum import DirectSum
from pypolyphonicanalysis.models.baseline_model import BaselineModel
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.datamodel.data_multiplexing.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

settings = Settings()
shuffle = False

feature_store = get_feature_store(settings)

summing_strategies: list[BaseSummingStrategy] = [
    DirectSum(settings),
    # ReverbSum(settings),
    # RoomSimulationSum(settings, (10, 7.5, 3.5), (2.5, 3.73, 1.76), [(4, 4.6, 1.6), (4.5, 4.8, 1.8), (5, 4.85, 1.7), (5.5, 4.8, 1.6), (6, 4.6, 1.9)], rt60=1.0, max_rand_disp=0.5),
]

monophonic_tracks_source_path = Path(settings.data_directory_path).joinpath("corpora").joinpath("monophonic_collections").joinpath("SingingVoiceDataset").joinpath("monophonic")
monophonic_tracks: dict[Path, Path | None] = {
    Path(os.path.join(dirpath, filename)): None for (dirpath, dirnames, filenames) in os.walk(monophonic_tracks_source_path) for filename in filenames if ".wav" in filename
}


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

pitch_shift_probabilities: dict[float, float] | None = None
dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]] = [(dl, summing_strategies) for dl in dataset_loaders]
sum_track_provider = SumTrackProvider(settings, dataloaders_and_summing_strategies=dataloaders_and_summing_strategies)
mux = SumTrackFeatureStreamMux(sum_track_provider, [Features.HCQT_MAG, Features.HCQT_PHASE_DIFF], [Features.SALIENCE_MAP], settings)

model = BaselineModel(settings)
torch_model = model.model

pitch_shift_augmentation_multiplier = 1 + (sum(prob for prob in pitch_shift_probabilities.values()) if pitch_shift_probabilities is not None else 0)
all_samples_len = (
    sum(len(loader) * len(summing_strategies_for_loader) for loader, summing_strategies_for_loader in dataloaders_and_summing_strategies)
    * settings.training_mux_number_of_samples_per_sum_track
    * pitch_shift_augmentation_multiplier
)
train_len = int((1 - settings.test_validation_size) * all_samples_len)
validation_len = int(settings.validation_proportion * (settings.test_validation_size * all_samples_len))
test_size = int((1 - settings.validation_proportion) * (settings.test_validation_size * all_samples_len))

MODEL_NAME = "model_train"
LR = 0.01
EPOCHS = 2

loss_fn = nn.KLDivLoss()
optimizer = torch.optim.AdamW(torch_model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, patience=10)

best_loss: float = math.inf
best_loss_epoch = -1
for epoch in tqdm(range(EPOCHS)):
    train, _, validation = mux.get_feature_iterators()
    model.train_on_feature_iterable(train, optimizer, train_len)
    validation_loss = model.validate_on_feature_iterable(validation, validation_len)
    scheduler.step(validation_loss)
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_loss_epoch = epoch
        model.save(MODEL_NAME)
    if epoch - best_loss_epoch >= 40:
        break
