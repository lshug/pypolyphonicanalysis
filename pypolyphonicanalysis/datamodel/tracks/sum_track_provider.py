import logging
from enum import Enum
from typing import Iterable, TypeVar

from tqdm import tqdm

from pypolyphonicanalysis.datamodel.tracks.splits import SumTrackSplitType, TrainTestValidationSplit
from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import BaseSummingStrategy
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack, load_sum_track, sum_track_is_saved
from pypolyphonicanalysis.datamodel.tracks.sum_track_processing.base_sum_track_processor import BaseSumTrackProcessor
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import get_random_number_generator
from pypolyphonicanalysis.datamodel.features.feature_store import (
    get_feature_store,
    sum_track_n_frames_is_saved_in_feature_store,
    get_sum_track_n_frames_from_feature_store,
    feature_is_generated_for_sum_track,
)
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class SummingModes(Enum):
    RANDOM = 0
    ALL = 1


T = TypeVar("T")


def partition_list(list_to_partition: list[T], n: int) -> list[list[T]]:
    return [list_to_partition[idx::n] for idx in range(n)]


def generate_random_split(settings: Settings) -> SumTrackSplitType:
    rng = get_random_number_generator(settings)
    if rng.random() < settings.test_validation_size:
        if rng.random() < settings.validation_proportion:
            return SumTrackSplitType.VALIDATION
        return SumTrackSplitType.TEST
    return SumTrackSplitType.TRAIN


def get_multitrack_generator_from_dataloader_partition(
    dataloader_partition: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]],
    settings: Settings,
) -> Iterable[tuple[Multitrack, list[BaseSummingStrategy]]]:
    rng = get_random_number_generator(settings)
    dataloader_iters = [(iter(dataloader.get_multitracks()), summing_strategies) for dataloader, summing_strategies in dataloader_partition]
    while len(dataloader_iters) > 0:
        loader, summing_strategies = rng.choice(dataloader_iters)
        try:
            yield next(loader), summing_strategies
        except StopIteration:
            dataloader_iters.remove((loader, summing_strategies))


def process_sum_track(sum_track: SumTrack, sum_track_processors: list[BaseSumTrackProcessor], settings: Settings) -> SumTrack:
    logger.info(f"Processing sum_track {sum_track.name}")
    feature_store = get_feature_store(settings)
    for processor in sum_track_processors:
        sum_track = processor.process_or_load(sum_track)
    if settings.save_raw_training_data:
        sum_track.save()
    for feature in settings.sum_track_provider_features_to_generate_early:
        if not feature_is_generated_for_sum_track(sum_track, feature, settings):
            feature_store.generate_or_load_feature_for_sum_track(sum_track, feature)
    return sum_track


def process_multitrack_with_summing_strategies(
    multitrack: Multitrack,
    summing_strategies: list[BaseSummingStrategy],
    pitch_shift_probabilities: dict[float, float] | None,
    pitch_shift_displacement_range: tuple[float, float],
    sum_track_processors: list[BaseSumTrackProcessor] | None,
    summing_mode: SummingModes,
    settings: Settings,
    logging_level: int,
) -> list[tuple[SumTrack, SumTrackSplitType]]:
    logging.basicConfig(level=logging_level)
    if sum_track_processors is None:
        sum_track_processors = []
    multitrack_split = generate_random_split(settings)
    rng = get_random_number_generator(settings)
    augmented_multitracks: list[Multitrack] = []
    sum_tracks_with_splits: list[tuple[SumTrack, SumTrackSplitType]] = []
    logger.info(f"Augmenting multitrack {multitrack}, split {multitrack_split}")
    augmented_multitracks.append(multitrack)
    if pitch_shift_probabilities is not None:
        for shift, probability in pitch_shift_probabilities.items():
            if rng.random() <= probability:
                augmented_multitracks.append(multitrack.pitch_shift(shift, pitch_shift_displacement_range))
    for multitrack in augmented_multitracks:
        logger.info(f"Processing augmented multitrack {multitrack}, split {multitrack_split}")
        summing_strategies_to_use: list[BaseSummingStrategy] = []
        sum_tracks_with_split_preferences: list[tuple[SumTrack, SumTrackSplitType | None]] = []
        skip_processing = False
        if summing_mode == SummingModes.RANDOM:
            summing_strategies_to_use.append(rng.choice(summing_strategies))
        else:
            summing_strategies_to_use = summing_strategies
        for summing_strategy in summing_strategies_to_use:
            expected_name = summing_strategy.get_sum_track_name(multitrack)
            for processor in sum_track_processors:
                expected_name = processor.get_sum_track_name_from_base_sumtrack_name(expected_name)
            if sum_track_is_saved(expected_name, settings):
                skip_processing = True
                sum_tracks_with_split_preferences.append((load_sum_track(expected_name, settings), summing_strategy.split_override))
            else:
                sum_tracks_with_split_preferences.append((summing_strategy.sum_or_retrieve(multitrack), summing_strategy.split_override))
        for sum_track, split_preference in sum_tracks_with_split_preferences:
            if not skip_processing:
                sum_track = process_sum_track(sum_track, sum_track_processors, settings)
            sum_tracks_with_splits.append((sum_track, multitrack_split if split_preference is None else split_preference))
    return sum_tracks_with_splits


def get_multitracks_from_dataloader_partition(
    partition: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]],
    settings: Settings,
    logging_level: int,
) -> list[tuple[Multitrack, list[BaseSummingStrategy]]]:
    logging.basicConfig(level=logging_level)
    logger.info(f"Processing dataloader partition {partition}")
    if len(partition) == 0:
        return []
    total = sum(len(dataloader) for dataloader, _ in partition)
    return list(tqdm(get_multitrack_generator_from_dataloader_partition(partition, settings), "Getting multitracks from partition", total=total))


SumTrackWithSplitIterable = Iterable[tuple[SumTrack, SumTrackSplitType]]


class SumTrackProvider:
    def __init__(
        self,
        settings: Settings,
        train_test_validation_split: TrainTestValidationSplit | None = None,
        dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]] | None = None,
        pitch_shift_probabilities: dict[float, float] | None = None,
        pitch_shift_displacement_range: tuple[float, float] = (0, 0),
        sum_track_processors: list[BaseSumTrackProcessor] | None = None,
        summing_mode: SummingModes = SummingModes.ALL,
    ):
        self._train_test_validation_split = train_test_validation_split
        self._dataloaders_and_summing_strategies = dataloaders_and_summing_strategies
        self._pitch_shift_probabilities = pitch_shift_probabilities
        self._pitch_shift_displacement_range = pitch_shift_displacement_range
        self._sum_track_processors = sum_track_processors
        self._settings = settings
        self._summing_mode = summing_mode
        self._rng = get_random_number_generator(settings)
        self._feature_store = get_feature_store(settings)

    def get_sum_tracks(self, include_train: bool = True, include_test: bool = True, include_validation: bool = True) -> Iterable[tuple[SumTrack, SumTrackSplitType]]:
        if self._train_test_validation_split is not None:
            yield from self._get_sum_tracks_from_train_test_validation_split(include_train, include_test, include_validation)
        if self._dataloaders_and_summing_strategies is not None:
            yield from self._get_sum_tracks_from_dataloaders(include_train, include_test, include_validation)

    def _get_sum_tracks_from_dataloaders(self, include_train: bool, include_test: bool, include_validation: bool) -> SumTrackWithSplitIterable:
        assert self._dataloaders_and_summing_strategies is not None
        logging_level = logger.getEffectiveLevel()
        dataloader_partitions = partition_list(self._dataloaders_and_summing_strategies, self._settings.sum_track_provider_number_of_dataloader_partition_jobs)
        multitrack_lists = Parallel(n_jobs=self._settings.sum_track_provider_number_of_dataloader_partition_jobs, return_as="generator", verbose=5)(
            delayed(get_multitracks_from_dataloader_partition)(partition, self._settings, logging_level) for partition in dataloader_partitions
        )
        for multitrack_list in tqdm(multitrack_lists, desc="Processing dataloader partitions", total=len(dataloader_partitions)):
            sum_track_with_splits_lists = Parallel(n_jobs=self._settings.sum_track_provider_number_of_multitrack_processing_jobs_per_dataloader_partition, return_as="generator")(
                delayed(process_multitrack_with_summing_strategies)(
                    multitrack,
                    summing_strategies,
                    self._pitch_shift_probabilities,
                    self._pitch_shift_displacement_range,
                    self._sum_track_processors,
                    self._summing_mode,
                    self._settings,
                    logging_level,
                )
                for multitrack, summing_strategies in multitrack_list
            )
            for sum_track_with_splits_list in tqdm(sum_track_with_splits_lists, desc="Getting SumTracks from multitracks", total=len(multitrack_list)):
                for sum_track, split in sum_track_with_splits_list:
                    if (
                        (split == SumTrackSplitType.TRAIN and include_train)
                        or (split == SumTrackSplitType.TEST and include_test)
                        or (split == SumTrackSplitType.VALIDATION and include_validation)
                    ):
                        logger.debug(f"Yielding sum_track {sum_track}, split {split}")
                        yield sum_track, split

    def _shallow_load_sum_track(self, sum_track_name: str) -> SumTrack:
        if sum_track_n_frames_is_saved_in_feature_store(sum_track_name, self._settings):
            logging.info(f"Shallow-loading sum track {sum_track_name}")
            n_frames = get_sum_track_n_frames_from_feature_store(sum_track_name, self._settings)
            return load_sum_track(sum_track_name, self._settings, shallow=True, shallow_n_frames=n_frames)
        elif sum_track_is_saved(sum_track_name, self._settings):
            logging.info(f"Generating missing features for sum track {sum_track_name}")
            sum_track = load_sum_track(sum_track_name, self._settings)
            for feature in self._settings.sum_track_provider_features_to_generate_early:
                get_feature_store(self._settings).generate_or_load_feature_for_sum_track(sum_track, feature)
            return sum_track
        else:
            raise ValueError(f"Features for shallow-loading sum track {sum_track_name} not found. Saved sum track {sum_track_name} not found.")

    def _get_sum_tracks_from_train_test_validation_split(self, include_train: bool, include_test: bool, include_valiation: bool) -> SumTrackWithSplitIterable:
        assert self._train_test_validation_split is not None
        rng = self._rng
        sum_track_names_and_splits: list[tuple[str, SumTrackSplitType]] = []
        if include_train:
            sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TRAIN) for sum_track_name in self._train_test_validation_split["train"]])
        if include_test:
            sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TEST) for sum_track_name in self._train_test_validation_split["test"]])
        if include_valiation:
            sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.VALIDATION) for sum_track_name in self._train_test_validation_split["validation"]])
        rng.shuffle(sum_track_names_and_splits)
        yield from ((self._shallow_load_sum_track(sum_track_name), split_type) for sum_track_name, split_type in sum_track_names_and_splits)
