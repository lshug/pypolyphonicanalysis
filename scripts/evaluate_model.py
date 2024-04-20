from pypolyphonicanalysis.datamodel.tracks.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.evaluation.test_set_evaluator import (
    TestSetEvaluator,
)
from pypolyphonicanalysis.models.multiple_f0_estimation.baseline_model import BaselineModel
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import get_train_test_validation_split

settings = Settings()

model = BaselineModel(settings, "model")
sum_track_provider = SumTrackProvider(settings, train_test_validation_split=get_train_test_validation_split("train_test_validation_split", settings))
evaluator = TestSetEvaluator(sum_track_provider, settings=settings, max_count=40)
evaluation = evaluator.evaluate_model(model)
