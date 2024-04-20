import abc
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, TypeVar, Sequence, cast

from pypolyphonicanalysis.datamodel.tracks.multitrack import Multitrack
from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import get_random_number_generator

SequenceType = TypeVar("SequenceType")

logger = logging.getLogger(__name__)


class BaseDataLoader(abc.ABC):
    def __init__(self, shuffle: bool, settings: Settings, maxlen: int = 6000) -> None:
        self._settings = settings
        self._shuffle = shuffle
        self._random = get_random_number_generator(settings)
        self._maxlen = maxlen
        self._count = 0

    def _shuffle_if_enabled(self, iterable: Sequence[SequenceType]) -> Sequence[SequenceType]:
        if self._shuffle:
            return cast(Sequence[SequenceType], self._random.sample(iterable, len(iterable)))
        return iterable

    def get_multitracks(self) -> Iterable[Multitrack]:
        for multitrack in self._get_multitracks():
            logger.debug(f"Yielding multitrack {multitrack} from {self}")
            yield multitrack
            self._count += 1
            if self._count > self._maxlen:
                break

    @abstractmethod
    def _get_multitracks(self) -> Iterable[Multitrack]:
        pass

    def __len__(self) -> int:
        return min(self._get_length(), self._maxlen)

    @abstractmethod
    def _get_length(self) -> int:
        pass

    def get_corpus_path(self, corpus_name: str) -> Path:
        return Path(self._settings.data_directory_path).joinpath("corpora").joinpath(corpus_name)
