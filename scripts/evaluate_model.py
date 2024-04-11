from pypolyphonicanalysis.datamodel.evaluation.test_set_evaluator import (
    TestSetEvaluator,
)
from pypolyphonicanalysis.models.baseline_model import BaselineModel
from pypolyphonicanalysis.settings import Settings


settings = Settings()

model = BaselineModel("model.pth", settings)
evaluator = TestSetEvaluator(settings=settings, max_count=10)
evaluation = evaluator.evaluate_model(model)
