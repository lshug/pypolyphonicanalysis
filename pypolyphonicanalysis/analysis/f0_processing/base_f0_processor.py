import abc
from abc import abstractmethod

from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.utils.utils import FloatArray, F0TimesAndFrequencies


class BaseF0Processor(abc.ABC):
    @abstractmethod
    def process(self, recording: Recording, times: FloatArray, freqs: FloatArray) -> F0TimesAndFrequencies:
        """
        Returns processed times and freqs.
        """

    @abstractmethod
    def get_stage_name(self) -> str:
        """Gets processing stage name of current processor."""
