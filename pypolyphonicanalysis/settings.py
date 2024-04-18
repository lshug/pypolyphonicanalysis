from pydantic import Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from pypolyphonicanalysis.datamodel.features.features import Features


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="pypolyphonicanalysis_", frozen=True)

    # General
    data_directory_path: str = Field(default="./data")
    random_seed: int = Field(default=123)

    # SumTrackProvider, FeatureStore
    test_validation_size: PositiveFloat = Field(default=0.2, le=1.0)
    validation_proportion: PositiveFloat = Field(default=0.5, le=1.0)
    save_raw_training_data: bool = True
    save_training_features: bool = True
    save_prediction_file_features: bool = True
    sum_track_provider_number_of_dataloader_partition_jobs: PositiveInt = 10
    sum_track_provider_number_of_multitrack_processing_jobs_per_dataloader_partition: PositiveInt = 10
    sum_track_provider_features_to_generate_early: frozenset[Features] = Field(default=frozenset({Features.HCQT_MAG, Features.HCQT_PHASE_DIFF, Features.HCQT_PHASE_DIFF}))

    # Data mux
    training_mux_number_of_active_streams: PositiveInt = 100
    training_mux_number_of_samples_per_sum_track: PositiveInt = 10

    # Training
    inference_batch_size: PositiveInt = 4
    inference_input_number_of_slices: PositiveInt = 5000
    training_batch_size: PositiveInt = 128
    training_input_number_of_slices: PositiveInt = 50

    # Audio and model parameters
    sr: PositiveInt = Field(default=22050)
    bins_per_octave: PositiveInt = Field(default=60)
    n_octaves: PositiveInt = Field(default=6)
    over_sample: PositiveInt = Field(default=5)
    harmonics: frozenset[PositiveInt] = Field(default=frozenset({1, 2, 3, 4, 5}))
    fmin: PositiveFloat = Field(default=32.7)
    hop_length: PositiveInt = Field(default=256)
    blur_salience_map: bool = Field(default=True)
    epsilon: float = 1e-10

    # Analysis
    activation_threshold: PositiveFloat = Field(default=0.5, le=1.0)
    default_figsize: tuple[PositiveInt, PositiveInt] = (30, 15)
    histogram_bins: PositiveInt = 49
    density_estimation_bandwidth: PositiveFloat = 10.0
    peak_finding_minimum_cent_distance: PositiveFloat = 50
    squeeze_harmonic_intervals_into_one_octave: bool = True
    clustering_distance_threshold: PositiveFloat = 30
