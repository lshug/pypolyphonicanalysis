from audiomentations import Compose, PeakingFilter, BandPassFilter, BandStopFilter, HighPassFilter, HighShelfFilter, LowPassFilter, LowShelfFilter

from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.processing.base_sum_track_processor import BaseSumTrackProcessor
from pypolyphonicanalysis.settings import Settings


class Filter(BaseSumTrackProcessor):
    def __init__(
        self,
        settings: Settings,
        p: float = 0.1,
        peaking_cond_p: float = 0.1,
        bandpass_cond_p: float = 0.1,
        bandstop_cond_p: float = 0.1,
        highpass_cond_p: float = 0.1,
        highshelf_cond_p: float = 0.1,
        lowpass_cond_p: float = 0.1,
        lowshelf_cond_p: float = 0.1,
    ) -> None:
        super().__init__(settings)
        self._augment = Compose(
            [
                PeakingFilter(p=peaking_cond_p),
                BandPassFilter(p=bandpass_cond_p),
                BandStopFilter(p=bandstop_cond_p),
                HighPassFilter(p=highpass_cond_p),
                HighShelfFilter(p=highshelf_cond_p),
                LowPassFilter(p=lowpass_cond_p),
                LowShelfFilter(p=lowshelf_cond_p),
            ],
            p=p,
            shuffle=True,
        )

    def _process(self, sum_track: SumTrack) -> SumTrack:
        augmented = self._augment(sum_track.audio_array, self._settings.sr)
        return SumTrack(self.get_sum_track_name(sum_track), augmented, sum_track.source_multitrack, self._settings)

    def get_sum_track_name_from_base_sumtrack_name(self, sum_track_name: str) -> str:
        return f"{sum_track_name}_add_filter"
