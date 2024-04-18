import logging
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from pypolyphonicanalysis.datamodel.data_multiplexing.splits import SumTrackSplitType
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
from pypolyphonicanalysis.utils.utils import get_train_test_validation_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

settings = Settings()

split = get_train_test_validation_split('joint_split', settings)
sum_track_provider = SumTrackProvider(settings, train_test_validation_split=split)
mux = SumTrackFeatureStreamMux(sum_track_provider, [Features.HCQT_MAG, Features.HCQT_PHASE_DIFF], [Features.SALIENCE_MAP], settings)

model = BaselineModel(settings)
torch_model = model.model

train_len = len(split['train'])
validation_len = len(split['test'])
test_len = len(split['validation'])

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
    validation_loss = model.validate_on_feature_iterable(validation, validation_len)
    scheduler.step(validation_loss)
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_loss_epoch = epoch
        model.save(MODEL_NAME)
    if epoch - best_loss_epoch >= 40:
        break
