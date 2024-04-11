from pydantic import Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="pypolyphonicanalysis_", frozen=True)
    data_directory_path: str = Field(default="./data")
    random_seed: int = Field(default=123)
    test_validation_size: PositiveFloat = Field(default=0.2, le=1.0)
    validation_proportion: PositiveFloat = Field(default=0.5, le=1.0)
    inference_batch_size: int = 4
    training_batch_size: int = 1

    input_number_of_slices: int = 5000
    bins_per_octave: PositiveInt = Field(default=60)
    n_octaves: PositiveInt = Field(default=6)
    over_sample: PositiveInt = Field(default=5)
    harmonics: frozenset[PositiveInt] = Field(default=frozenset({1, 2, 3, 4, 5}))
    sr: PositiveInt = Field(default=22050)
    fmin: PositiveFloat = Field(default=32.7)
    hop_length: PositiveInt = Field(default=256)

    blur_salience_map: bool = Field(default=True)

    threshold: PositiveFloat = Field(default=0.5, le=1.0)

    default_figsize: tuple[int, int] = (30, 15)
    histogram_bins: int = 49
    density_estimation_bandwidth: float = 10.0
    peak_finding_minimum_cent_distance: float = 50
    squeeze_harmonic_intervals_into_one_octave: bool = True
    clustering_distance_threshold: float = 30

    cache_prediction_file_features: bool = True
