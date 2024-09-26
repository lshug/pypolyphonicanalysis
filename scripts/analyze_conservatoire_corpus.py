import logging
import os
from pathlib import Path

import librosa

from pypolyphonicanalysis.analysis.analysis_runner import (
    AutomaticAnalysisRunner,
    AnalysisResults,
    EmbeddingMethod,
)
from pypolyphonicanalysis.analysis.f0_processing.maximum_cent_difference_from_mean_filter import MaximumCentDifferenceFromMeanFilter
from pypolyphonicanalysis.analysis.f0_processing.most_likely_voices_filter import MostLikelyVoicesFilter
from pypolyphonicanalysis.analysis.pitch_drift_detrenders.log_linear_detrender import LogLinearDetrender
from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.models.multiple_f0_estimation.baseline_model import BaselineModel
from pypolyphonicanalysis.analysis.f0_processing.frequency_range_filter import FrequencyRangeFilter
from pypolyphonicanalysis.analysis.f0_processing.masking_filter import MaskingFilter
from pypolyphonicanalysis.settings import Settings
from scripts.conservatoire_corpus_tools.conservatoire_corpus_utils import load_catalog_data

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.debug(f"{MostLikelyVoicesFilter}, {LogLinearDetrender} imported")

settings = Settings()
model = BaselineModel(settings, "model")

processors = [
    FrequencyRangeFilter(float(librosa.note_to_hz("G2")), float(librosa.note_to_hz("G7"))),
    MaximumCentDifferenceFromMeanFilter(),
    MaskingFilter(),
    # MostLikelyVoicesFilter(),
]

detrender = None  # LogLinearDetrender()

catalog = load_catalog_data(settings)

all_recordings: dict[str, Recording] = {}
for file in os.listdir("all_regions"):
    if ".wav" in file:
        file_path = Path(f"all_regions/{file}")
        assert file_path.exists()
        entry = catalog[file_path.name]
        recording_date = f"{entry.recording_date_year}-{entry.recording_date_month}-{entry.recording_date_day}"
        recording_name = file.split(".")[0]
        all_recordings[recording_name] = Recording(
            name=recording_name,
            file_path=file_path,
            number_of_voices=3,
            performers=entry.performers[:50] if entry.performers is not None else None,
            performer_group_type=entry.performer_group_type.name if entry.performer_group_type is not None else None,
            group_leader=entry.group_leader,
            recording_site=entry.recording_site,
            recording_date=recording_date if recording_date != "None-None-None" else None,
            recording_region=entry.recording_region,
        )


analyzer = AutomaticAnalysisRunner(Path("output/all_out"), model, processors, settings, detrender)
results_path = Path("output/all_out").joinpath("analysis_results_serialization.json")
if results_path.exists():
    results = AnalysisResults.model_validate_json(open(Path("output/all_out").joinpath("analysis_results_serialization.json")).read())
else:
    results = analyzer.generate_analysis_results(all_recordings)
    open(Path("output/all_out").joinpath("analysis_results_serialization.json"), "w").write(results.model_dump_json())


# UMAP for clustering
analyzer.perform_embedding_clustering(
    all_recordings,
    results.recording_name_harmonic_interval_gaussian_mixture_parameters,
    n_components=2,
    umap_n_neighbors=20,
    umap_min_dist=0,
    embedding_method=EmbeddingMethod.UMAP,
)


# UMAP for embedding
analyzer.perform_embedding_clustering(
    all_recordings,
    results.recording_name_harmonic_interval_gaussian_mixture_parameters,
    n_components=25,
    umap_n_neighbors=15,
    embedding_method=EmbeddingMethod.UMAP,
)

# PCA 20% variance
analyzer.perform_embedding_clustering(
    all_recordings,
    results.recording_name_harmonic_interval_gaussian_mixture_parameters,
    n_components=2,
    embedding_method=EmbeddingMethod.PCA,
)

# PCA 90% variance
analyzer.perform_embedding_clustering(
    all_recordings,
    results.recording_name_harmonic_interval_gaussian_mixture_parameters,
    n_components=25,
    embedding_method=EmbeddingMethod.PCA,
)
