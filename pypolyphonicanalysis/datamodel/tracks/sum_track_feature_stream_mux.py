import logging
from collections import Counter
from typing import Iterator

from pypolyphonicanalysis.datamodel.tracks.sum_track_provider import SumTrackProvider
from pypolyphonicanalysis.datamodel.tracks.splits import SumTrackSplitType
from pypolyphonicanalysis.datamodel.features.feature_stream import FeatureStream
from pypolyphonicanalysis.datamodel.features.features import InputFeature, LabelFeature
from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray, get_random_number_generator

FeatureStreamIteratorType = Iterator[tuple[list[FloatArray], list[FloatArray]]]

logger = logging.getLogger(__name__)


class SumTrackFeatureStreamMux:
    def __init__(
        self,
        sum_track_provider: SumTrackProvider,
        input_features: list[InputFeature],
        label_features: list[LabelFeature],
        settings: Settings,
        include_train: bool = True,
        include_test: bool = True,
        include_validation: bool = True,
    ) -> None:
        self._number_of_streams = settings.training_mux_number_of_active_streams
        self._sum_track_provider = sum_track_provider
        self._input_features = input_features
        self._label_features = label_features
        self._rng = get_random_number_generator(settings)
        self._include_train = include_train
        self._include_test = include_test
        self._include_validation = include_validation
        self._settings = settings

    def _get_multiplexed_feature_iterator_for_split_type(
        self,
        active_streams: list[tuple[FeatureStream, SumTrackSplitType]],
        sum_track_iterator: Iterator[tuple[SumTrack, SumTrackSplitType]],
        active_stream_counts: Counter[FeatureStream],
        queues: dict[SumTrackSplitType, list[tuple[list[FloatArray], list[FloatArray]]]],
        split_type: SumTrackSplitType,
    ) -> FeatureStreamIteratorType:
        first = True
        while first or len(queues[split_type]) != 0 or len(active_streams) != 0:
            queue_sizes = {k: len(v) for k, v in queues.items()}
            logger.info(f"Multiplex loop for {split_type}, queue sizes: {queue_sizes}")
            first = False
            if len(active_streams) < self._number_of_streams:
                try:
                    sum_track, sum_track_split_type = next(sum_track_iterator)
                    active_streams.append((FeatureStream(sum_track, self._input_features, self._label_features, self._settings), sum_track_split_type))
                except StopIteration:
                    logger.info("Multiplexer ran out of unused streams")
                    pass
            if len(queues[split_type]) != 0:
                yield queues[split_type].pop(-1)
            if len(active_streams) > 0:
                active_stream, sum_track_split_type = self._rng.choice(active_streams)
                queues[sum_track_split_type].append(next(active_stream))
                active_stream_counts[active_stream] += 1
                if active_stream_counts[active_stream] >= active_stream.maximum_number_of_samples:
                    del active_stream_counts[active_stream]
                    active_streams.remove((active_stream, sum_track_split_type))

    def get_feature_iterators(self) -> tuple[FeatureStreamIteratorType, FeatureStreamIteratorType, FeatureStreamIteratorType]:
        sum_track_iterator = iter(self._sum_track_provider.get_sum_tracks(self._include_train, self._include_test, self._include_validation))
        queues: dict[SumTrackSplitType, list[tuple[list[FloatArray], list[FloatArray]]]] = {
            SumTrackSplitType.TRAIN: [],
            SumTrackSplitType.TEST: [],
            SumTrackSplitType.VALIDATION: [],
        }
        active_streams: list[tuple[FeatureStream, SumTrackSplitType]] = []
        active_stream_counter: Counter[FeatureStream] = Counter()
        train_iterable = self._get_multiplexed_feature_iterator_for_split_type(active_streams, sum_track_iterator, active_stream_counter, queues, SumTrackSplitType.TRAIN)
        test_iterable = self._get_multiplexed_feature_iterator_for_split_type(active_streams, sum_track_iterator, active_stream_counter, queues, SumTrackSplitType.TEST)
        valid_iterable = self._get_multiplexed_feature_iterator_for_split_type(active_streams, sum_track_iterator, active_stream_counter, queues, SumTrackSplitType.VALIDATION)
        return iter(train_iterable), iter(test_iterable), iter(valid_iterable)
