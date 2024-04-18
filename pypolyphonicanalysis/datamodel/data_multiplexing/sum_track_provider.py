from enum import Enum
from typing import Iterable, TypeVar

from tqdm import tqdm

from pypolyphonicanalysis.datamodel.data_multiplexing.splits import SumTrackSplitType, TrainTestValidationSplit
from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import BaseSummingStrategy
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack, load_sum_track, sum_track_is_saved
from pypolyphonicanalysis.processing.sum_track.base_sum_track_processor import BaseSumTrackProcessor
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import get_random_number_generator
from pypolyphonicanalysis.datamodel.features.feature_store import get_feature_store
from joblib import Parallel, delayed


class SummingModes(Enum):
    RANDOM = 0
    ALL = 1


T = TypeVar("T")


def partition_list(list_to_partition: list[T], n: int) -> Iterable[list[T]]:
    for idx in range(0, n):
        yield list_to_partition[idx::n]


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


def process_multitrack_with_summing_strategies(
    multitrack: Multitrack,
    summing_strategies: list[BaseSummingStrategy],
    pitch_shift_probabilities: dict[float, float] | None,
    pitch_shift_displacement_range: tuple[float, float],
    sum_track_processors: list[BaseSumTrackProcessor] | None,
    summing_mode: SummingModes,
    settings: Settings,
) -> list[tuple[SumTrack, SumTrackSplitType]]:
    rng = get_random_number_generator(settings)
    feature_store = get_feature_store(settings)
    augmented_multitracks: list[Multitrack] = []
    sum_tracks_with_splits: list[tuple[SumTrack, SumTrackSplitType]] = []
    if pitch_shift_probabilities is None:
        augmented_multitracks.append(multitrack)
    else:
        for shift, probability in pitch_shift_probabilities.items():
            if rng.random() <= probability:
                displacement = rng.uniform(pitch_shift_displacement_range[0], pitch_shift_displacement_range[1])
                augmented_multitracks.append(multitrack.pitch_shift(shift + displacement))
    for multitrack in augmented_multitracks:
        if summing_mode == SummingModes.RANDOM:
            sum_tracks = [rng.choice(summing_strategies).sum_or_retrieve(multitrack)]
        else:
            sum_tracks = [summing_strategy.sum_or_retrieve(multitrack) for summing_strategy in summing_strategies]
        for sum_track in sum_tracks:
            if sum_track_processors is not None:
                for processor in sum_track_processors:
                    sum_track = processor.process(sum_track)
            for feature in settings.sum_track_provider_features_to_generate_early:
                feature_store.generate_or_load_feature_for_sum_track(sum_track, feature)
            sum_tracks_with_splits.append((sum_track, generate_random_split(settings)))
    return sum_tracks_with_splits


def get_multitracks_from_dataloader_partition(
    partition: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]],
    settings: Settings,
) -> list[tuple[Multitrack, list[BaseSummingStrategy]]]:
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

    def get_sum_tracks(self) -> Iterable[tuple[SumTrack, SumTrackSplitType]]:
        if self._train_test_validation_split is not None:
            yield from self._get_sum_tracks_from_train_test_validation_split()
        if self._dataloaders_and_summing_strategies is not None:
            yield from self._get_sum_tracks_from_dataloaders()

    def _get_sum_tracks_from_dataloaders(self) -> SumTrackWithSplitIterable:
        assert self._dataloaders_and_summing_strategies is not None
        dataloader_partitions = list(partition_list(self._dataloaders_and_summing_strategies, self._settings.sum_track_provider_number_of_dataloader_partition_jobs))
        multitrack_lists = Parallel(n_jobs=self._settings.sum_track_provider_number_of_dataloader_partition_jobs, return_as="generator", verbose=5)(
            delayed(get_multitracks_from_dataloader_partition)(partition, self._settings) for partition in dataloader_partitions
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
                )
                for multitrack, summing_strategies in multitrack_list
            )
            for sum_track_with_splits_list in tqdm(sum_track_with_splits_lists, desc="Getting SumTracks from multitracks", total=len(multitrack_list)):
                for sum_track, split in sum_track_with_splits_list:
                    yield sum_track, split

    def _load_or_shallow_load_sum_track(self, sum_track_name: str) -> SumTrack:
        if sum_track_is_saved(sum_track_name, self._settings):
            sum_track = load_sum_track(sum_track_name, self._settings, False)
            for feature in self._settings.sum_track_provider_features_to_generate_early:
                get_feature_store(self._settings).generate_or_load_feature_for_sum_track(sum_track, feature)
            return sum_track
        return load_sum_track(sum_track_name, self._settings, True)

    def _get_sum_tracks_from_train_test_validation_split(self) -> SumTrackWithSplitIterable:
        assert self._train_test_validation_split is not None
        rng = self._rng
        sum_track_names_and_splits: list[tuple[str, SumTrackSplitType]] = []
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TRAIN) for sum_track_name in self._train_test_validation_split["train"]])
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TEST) for sum_track_name in self._train_test_validation_split["test"]])
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.VALIDATION) for sum_track_name in self._train_test_validation_split["validation"]])
        rng.shuffle(sum_track_names_and_splits)
        yield from ((self._load_or_shallow_load_sum_track(sum_track_name), split_type) for sum_track_name, split_type in sum_track_names_and_splits)
