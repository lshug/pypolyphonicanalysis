import librosa
import numpy as np
from sklearn.linear_model import LinearRegression

from pypolyphonicanalysis.analysis.pitch_drift_detrenders.base_pitch_drift_detrender import BasePitchDriftDetrender
from pypolyphonicanalysis.utils.utils import FloatArray


class LogLinearDetrender(BasePitchDriftDetrender):

    def get_correction_values(self, times: FloatArray, freqs: FloatArray) -> FloatArray:
        cents_above_a1 = 1200 * np.log2(freqs / librosa.note_to_hz("A1"), out=np.zeros_like(freqs), where=freqs != 0)
        regression_idxs = np.any(cents_above_a1 > 0, 1)
        regression_cents = cents_above_a1[regression_idxs]
        regression_times = times[regression_idxs]
        x_points: list[float] = []
        y_points: list[float] = []
        for time, cent_vals in zip(regression_times, regression_cents):
            for cent_val in cent_vals:
                x_points.append(time)
                y_points.append(cent_val)
        reg = LinearRegression()
        reg.fit(np.array(x_points).reshape(-1, 1), np.array(y_points))
        correction_values: FloatArray = reg.predict(times.reshape(-1, 1)) - reg.intercept_
        return correction_values
