import os
from pathlib import Path

import librosa

from pypolyphonicanalysis.analysis.analysis_runner import (
    Recording,
    AutomaticAnalysisRunner,
)
from pypolyphonicanalysis.models.baseline_model import BaselineModel
from pypolyphonicanalysis.processing.frequency_range_filter import FrequencyRangeFilter
from pypolyphonicanalysis.processing.masking_filter import MaskingFilter
from pypolyphonicanalysis.settings import Settings

settings = Settings()
model = BaselineModel("model.pth", settings, inference_mode=True)

processors = [
    FrequencyRangeFilter(float(librosa.note_to_hz("G2")), float(librosa.note_to_hz("G7"))),
    MaskingFilter(),
]

svaneti_recordings: list[Recording] = []
for file in os.listdir("audio/svaneti"):
    if ".wav" in file:
        svaneti_recordings.append(Recording(name=file.split(".")[0], file_path=Path(f"audio/svaneti/{file}")))

kakheti_recordings: list[Recording] = []
for file in os.listdir("audio/kakheti"):
    if ".wav" in file:
        kakheti_recordings.append(Recording(name=file.split(".")[0], file_path=Path(f"audio/kakheti/{file}")))

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
            )
        )

analyzer = AutomaticAnalysisRunner(Path("output/svaneti_out/"), model, processors, settings, Path("activation_cache/"))
svaneti_results = analyzer.generate_analysis_results(svaneti_recordings)

analyzer.set_output_path(Path("output/kakheti_out/"))
kakheti_results = analyzer.generate_analysis_results(kakheti_recordings)

analyzer.set_output_path(Path("output/gvm_out/"))
gvm_results = analyzer.generate_analysis_results(gvm_recordings)
