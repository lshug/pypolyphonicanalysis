import logging
import os
from pathlib import Path

import librosa
from anbani.nlp.georgianisation import georgianise

from pypolyphonicanalysis.analysis.analysis_runner import (
    AutomaticAnalysisRunner,
)
from pypolyphonicanalysis.analysis.f0_processing.maximum_cent_difference_from_mean_filter import MaximumCentDifferenceFromMeanFilter
from pypolyphonicanalysis.analysis.f0_processing.most_likely_voices_filter import MostLikelyVoicesFilter
from pypolyphonicanalysis.analysis.pitch_drift_detrenders.log_linear_detrender import LogLinearDetrender
from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.models.multiple_f0_estimation.baseline_model import BaselineModel
from pypolyphonicanalysis.analysis.f0_processing.frequency_range_filter import FrequencyRangeFilter
from pypolyphonicanalysis.analysis.f0_processing.masking_filter import MaskingFilter
from pypolyphonicanalysis.settings import Settings
from scripts.conservatoire_corpus_tools.conservatoire_corpus_utils import load_catalog_data, CatalogEntry

logging.basicConfig(level=logging.INFO)

settings = Settings()
model = BaselineModel(settings, "model")

processors = [
    FrequencyRangeFilter(float(librosa.note_to_hz("G2")), float(librosa.note_to_hz("G7"))),
    MaximumCentDifferenceFromMeanFilter(),
    MaskingFilter(),
    MostLikelyVoicesFilter(),
]

detrender = LogLinearDetrender()


conservatoire_corpus_catalog = load_catalog_data(settings)
file_stem_entry_dict: dict[str, CatalogEntry] = {Path(entry.file_path).stem: entry for entry in conservatoire_corpus_catalog}


svaneti_recordings: list[Recording] = []
for file in os.listdir("audio/svaneti"):
    if ".wav" in file:
        file_path = Path(f"audio/svaneti/{file}")
        entry = file_stem_entry_dict[file_path.stem]
        recording_date = f"{entry.recording_date_year}-{entry.recording_date_month}-{entry.recording_date_day}"
        svaneti_recordings.append(
            Recording(
                name=file.split(".")[0],
                file_path=file_path,
                number_of_voices=3,
                performers=georgianise(entry.performers[:50], mode="fast") if entry.performers is not None else None,
                recording_site=georgianise(entry.recording_site, mode="fast") if entry.recording_site is not None else None,
                recording_date=recording_date if recording_date != "None-None-None" else None,
            )
        )

kakheti_recordings: list[Recording] = []
for file in os.listdir("audio/kakheti"):
    if ".wav" in file:
        kakheti_recordings.append(Recording(name=file.split(".")[0], file_path=Path(f"audio/kakheti/{file}"), number_of_voices=3))

gvm_recordings: list[Recording] = []
gvm_exceptions = [
    "GVM020_DaleKojas_LataliVillage_SoniaTserediani_20160831_AOLS5S",
    "GVM212_ZharewodaImzuiwoRalekhaTake1_LataliVillage_SoniaTserediani_20160831_AOLS5S",
]
for file in os.listdir(os.path.join(settings.data_directory_path, "corpora", "GVM")):
    if "AOLS5S.wav" in file and all(gvm_exception not in file for gvm_exception in gvm_exceptions):
        ground_truth_files: list[Path] = [
            Path(settings.data_directory_path).joinpath("corpora").joinpath("GVM").joinpath(f)
            for f in os.listdir(os.path.join(settings.data_directory_path, "corpora", "GVM"))
            if file.replace("AOLS5S.wav", "AHDS") in f
        ]
        gvm_recordings.append(
            Recording(
                name=file.split(".")[0],
                file_path=Path(os.path.join(settings.data_directory_path, "corpora", "GVM", file)),
                ground_truth_files=None if len(ground_truth_files) == 0 else tuple(ground_truth_files),
                number_of_voices=3,
            )
        )


analyzer = AutomaticAnalysisRunner(Path("output/svaneti_out/"), model, processors, settings, detrender)
svaneti_results = analyzer.generate_analysis_results(svaneti_recordings)

analyzer = AutomaticAnalysisRunner(Path("output/kakheti_out/"), model, processors, settings, detrender)
kakheti_results = analyzer.generate_analysis_results(kakheti_recordings)

analyzer = AutomaticAnalysisRunner(Path("output/gvm_out/"), model, processors, settings, detrender)
gvm_results = analyzer.generate_analysis_results(gvm_recordings)
