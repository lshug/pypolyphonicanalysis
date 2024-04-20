from audiomentations import Compose, Aliasing, BitCrush, ClippingDistortion, TanhDistortion

from pypolyphonicanalysis.datamodel.tracks.sum_track import SumTrack
from pypolyphonicanalysis.processing.base_sum_track_processor import BaseSumTrackProcessor
from pypolyphonicanalysis.settings import Settings


class Distort(BaseSumTrackProcessor):
    def __init__(
        self, settings: Settings, p: float = 0.2, aliasing_cond_p: float = 0.1, bitcrush_cond_p: float = 0.1, clipping_cond_p: float = 0.1, tanh_cond_p: float = 0.1
    ) -> None:
        super().__init__(settings)
        self._augment = Compose(
            [
                Aliasing(p=aliasing_cond_p),
                BitCrush(p=bitcrush_cond_p),
                ClippingDistortion(p=clipping_cond_p),
                TanhDistortion(p=tanh_cond_p),
            ],
            p=p,
            shuffle=True,
        )

    def _process(self, sum_track: SumTrack) -> SumTrack:
        augmented = self._augment(sum_track.audio_array, self._settings.sr)
        return SumTrack(self.get_sum_track_name(sum_track), augmented, sum_track.source_multitrack, self._settings)

    def get_sum_track_name_from_base_sumtrack_name(self, sum_track_name: str) -> str:
        return f"{sum_track_name}_distort"
