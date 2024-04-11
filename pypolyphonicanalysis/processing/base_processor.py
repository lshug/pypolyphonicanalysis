import abc
from abc import abstractmethod

from pypolyphonicanalysis.utils.utils import FloatArray, F0TimesAndFrequencies


class BaseProcessor(abc.ABC):
    @abstractmethod
    def process(self, times: FloatArray, freqs: list[FloatArray]) -> F0TimesAndFrequencies:
        """
        Returns processed times and freqs.
        """

    @abstractmethod
    def get_stage_name(self) -> str:
        """Gets processing stage name of current processor."""
