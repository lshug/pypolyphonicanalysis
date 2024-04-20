import abc


from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.utils import FloatArray


class BasePitchDriftDetrender(abc.ABC):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abc.abstractmethod
    def get_correction_values(self, times: FloatArray, freqs: FloatArray) -> FloatArray:
        pass

    def detrend(self, freqs: FloatArray, correction_values: FloatArray) -> FloatArray:
        new_freqs = freqs * 2 ** (-correction_values.reshape(-1, 1) / 1200)
        return new_freqs
