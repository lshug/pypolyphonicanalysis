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


class SumTrackProvider:
    def __init__(
        self, dataloaders_and_summing_strategies: list[tuple[BaseDataLoader, list[BaseSummingStrategy]]], sum_track_processors: list[BaseSumTrackProcessor], settings: Settings
    ):
        self._dataloaders_and_summing_strategies = dataloaders_and_summing_strategies
        self._sum_track_processors = sum_track_processors
        self._settings = settings
        self._rng = get_random_number_generator(settings)

    def _generate_random_split(self) -> SumTrackSplitType:
        if self._rng.random() < self._settings.test_validation_size:
            if self._rng.random() < self._settings.validation_proportion:
                return SumTrackSplitType.VALIDATION
            return SumTrackSplitType.TEST
        return SumTrackSplitType.TRAIN

    def get_sum_tracks(self, pitch_shift_probabilities: dict[float, float] | None) -> Iterable[tuple[SumTrack, SumTrackSplitType]]:
        dataloader_iters = [(idx, iter(loader.get_multitracks()), summing_strategies) for idx, (loader, summing_strategies) in enumerate(self._dataloaders_and_summing_strategies)]
        while len(dataloader_iters) > 0:
            loader_idx, loader, summing_strategies = self._rng.choice(dataloader_iters)
            try:
                multitrack = next(loader)
                augmented_multitracks: list[Multitrack] = []
                if pitch_shift_probabilities is None:
                    augmented_multitracks.append(multitrack)
                else:
                    for shift, probability in pitch_shift_probabilities.items():
                        if self._rng.random() <= probability:
                            augmented_multitracks.append(multitrack.pitch_shift(shift))
                for multitrack in augmented_multitracks:
                    sum_track = self._rng.choice(summing_strategies).sum_or_retrieve(multitrack)
                    for processor in self._sum_track_processors:
                        sum_track = processor.process(sum_track)
                    yield sum_track, self._generate_random_split()
            except StopIteration:
                dataloader_iters.pop(loader_idx)

    @classmethod
    def get_sum_tracks_from_train_test_validation_split(
        cls, train_test_validation_split: TrainTestValidationSplit, settings: Settings
    ) -> Iterable[tuple[SumTrack, SumTrackSplitType]]:
        rng = get_random_number_generator(settings)
        sum_track_names_and_splits: list[tuple[str, SumTrackSplitType]] = []
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TRAIN) for sum_track_name in train_test_validation_split["train"]])
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.TEST) for sum_track_name in train_test_validation_split["test"]])
        sum_track_names_and_splits.extend([(sum_track_name, SumTrackSplitType.VALIDATION) for sum_track_name in train_test_validation_split["validation"]])
        rng.shuffle(sum_track_names_and_splits)
        for sum_track_name, split_type in sum_track_names_and_splits:
            shallow = not sum_track_is_saved(sum_track_name, settings)
            yield load_sum_track(sum_track_name, settings, shallow), split_type
