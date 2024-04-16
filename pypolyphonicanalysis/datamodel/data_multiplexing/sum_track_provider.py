from enum import Enum
from typing import Iterable, TypedDict

from pypolyphonicanalysis.datamodel.dataloaders.base_data_loader import BaseDataLoader
from pypolyphonicanalysis.datamodel.summing_strategies.base_summing_strategy import BaseSummingStrategy
from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack, load_sum_track, sum_track_is_saved
from pypolyphonicanalysis.processing.sum_track.base_sum_track_processor import BaseSumTrackProcessor
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import get_random_number_generator


class SumTrackSplitType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


class TrainTestValidationSplit(TypedDict):
    train: list[str]
    test: list[str]
    validation: list[str]


SumTrackWithSplitIterable = Iterable[tuple[SumTrack, SumTrackSplitType]]


class SumTrackProvider:
    def __init__(
        self,
        train_test_validation_split: TrainTestValidationSplit | None,
        dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]] | None,
        pitch_shift_probabilities: dict[float, float] | None,
        sum_track_processors: list[BaseSumTrackProcessor] | None,
        settings: Settings,
    ):
        self._train_test_validation_split = train_test_validation_split
        self._dataloaders_and_summing_strategies = dataloaders_and_summing_strategies
        self._pitch_shift_probabilities = pitch_shift_probabilities
        self._sum_track_processors = sum_track_processors
        self._settings = settings
        self._rng = get_random_number_generator(settings)

    def _generate_random_split(self) -> SumTrackSplitType:
        if self._rng.random() < self._settings.test_validation_size:
            if self._rng.random() < self._settings.validation_proportion:
                return SumTrackSplitType.VALIDATION
            return SumTrackSplitType.TEST
        return SumTrackSplitType.TRAIN

    def get_sum_tracks(self) -> Iterable[tuple[SumTrack, SumTrackSplitType]]:
        if self._train_test_validation_split is not None:
            yield from self._get_sum_tracks_from_train_test_validation_split()
        if self._dataloaders_and_summing_strategies is not None:
            yield from self._get_sum_tracks_from_dataloaders()

    def _get_sum_tracks_from_dataloaders(self) -> SumTrackWithSplitIterable:
        assert self._dataloaders_and_summing_strategies is not None
        dataloader_iters = [(idx, iter(loader.get_multitracks()), summing_strategies) for idx, (loader, summing_strategies) in enumerate(self._dataloaders_and_summing_strategies)]
        while len(dataloader_iters) > 0:
            loader_idx, loader, summing_strategies = self._rng.choice(dataloader_iters)
            try:
                multitrack = next(loader)
                augmented_multitracks: list[Multitrack] = []
                if self._pitch_shift_probabilities is None:
                    augmented_multitracks.append(multitrack)
                else:
                    for shift, probability in self._pitch_shift_probabilities.items():
                        if self._rng.random() <= probability:
                            augmented_multitracks.append(multitrack.pitch_shift(shift))
                for multitrack in augmented_multitracks:
                    sum_track = self._rng.choice(summing_strategies).sum_or_retrieve(multitrack)
                    if self._sum_track_processors is not None:
                        for processor in self._sum_track_processors:
                            sum_track = processor.process(sum_track)
                    yield sum_track, self._generate_random_split()
            except StopIteration:
                dataloader_iters.pop(loader_idx)

    def _get_sum_tracks_from_train_test_validation_split(self) -> SumTrackWithSplitIterable:
        assert self._train_test_validation_split is not None
        rng = self._rng
        sum_track_names_and_splits: list[tuple[str, SumTrackSplitType]] = []
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TRAIN) for sum_track_name in self._train_test_validation_split["train"]])
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TEST) for sum_track_name in self._train_test_validation_split["test"]])
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.VALIDATION) for sum_track_name in self._train_test_validation_split["validation"]])
        rng.shuffle(sum_track_names_and_splits)
        for sum_track_name, split_type in sum_track_names_and_splits:
            shallow = not sum_track_is_saved(sum_track_name, self._settings)
            yield load_sum_track(sum_track_name, self._settings, shallow), split_type
