import logging
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from pypolyphonicanalysis.datamodel.tracks.sum_track_feature_stream_mux import SumTrackFeatureStreamMux
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
from pypolyphonicanalysis.datamodel.summing_strategies.reverb_sum import ReverbSum
from pypolyphonicanalysis.datamodel.summing_strategies.room_simulation_sum import RoomSimulationSum, RelativePositionRange
from pypolyphonicanalysis.models.multiple_f0_estimation.residual_model import ResidualModel
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.datamodel.tracks.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store

logger = logging.getLogger(__name__)

settings = Settings(use_depthwise_separable_convolution_when_possible=True, use_self_attention=True, training_batch_size=8)
shuffle = False

feature_store = get_feature_store(settings)

summing_strategies: list[BaseSummingStrategy] = [
    DirectSum(settings),
    ReverbSum(settings),
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

model = ResidualModel(settings)
torch_model = model.model

pitch_shift_augmentation_multiplier = 1 + (sum(prob for prob in pitch_shift_probabilities.values()) if pitch_shift_probabilities is not None else 0)
all_samples_len = (
    sum(len(loader) * len(summing_strategies_for_loader) for loader, summing_strategies_for_loader in dataloaders_and_summing_strategies)
    * settings.training_mux_number_of_samples_per_sum_track_minute
    * 2.1
    * pitch_shift_augmentation_multiplier
)
train_len = int((1 - settings.test_validation_size) * all_samples_len)
validation_len = int(settings.validation_proportion * (settings.test_validation_size * all_samples_len))
test_size = int((1 - settings.validation_proportion) * (settings.test_validation_size * all_samples_len))

MODEL_NAME = "model_train"
LR = 0.01
EPOCHS = 150

loss_fn = nn.KLDivLoss()
optimizer = torch.optim.AdamW(torch_model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, patience=10)

best_loss: float = math.inf
best_loss_epoch = -1
for epoch in tqdm(range(EPOCHS)):
    train, _, validation = mux.get_feature_iterators()
    model.train_on_feature_iterable(train, optimizer, train_len)
    validation_loss, evaluation_metrics = model.validate_on_feature_iterable(validation, validation_len)
    scheduler.step(validation_loss)
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_loss_epoch = epoch
        model.save(MODEL_NAME)
    if epoch - best_loss_epoch >= 40:
        break
