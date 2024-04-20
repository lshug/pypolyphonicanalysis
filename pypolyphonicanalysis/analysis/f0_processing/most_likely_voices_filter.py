from itertools import combinations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from pypolyphonicanalysis.analysis.f0_processing.base_f0_processor import BaseF0Processor
from pypolyphonicanalysis.analysis.recording import Recording
from pypolyphonicanalysis.utils.utils import FloatArray, F0TimesAndFrequencies


class MostLikelyVoicesFilter(BaseF0Processor):
    def __init__(self, correct_time_slices_only_model_weight: float = 100, all_time_slices_model_weight: float = 0.5) -> None:
        self._correct_time_slices_only_model_weight = correct_time_slices_only_model_weight
        self._all_time_slices_model_weight = all_time_slices_model_weight

    def process(self, recording: Recording, times: FloatArray, freqs: FloatArray) -> F0TimesAndFrequencies:
        if recording.number_of_voices is None:
            return times, freqs
        current_number_of_voices = freqs.shape[1]
        new_freqs = freqs.copy()
        correct_time_slices_idxs = np.sum(freqs != 0, axis=1) == recording.number_of_voices
        greater_than_correct_time_slices_idxs: FloatArray = np.sum(freqs != 0, axis=1) > recording.number_of_voices
        freqs_correct = freqs[correct_time_slices_idxs]
        correct_time_slices_only_model = OneClassSVM(gamma="auto")
        correct_time_slices_only_model.fit(freqs_correct)
        all_time_slices_model = IsolationForest(random_state=0)
        all_time_slices_model.fit(freqs)
        for idx in range(len(freqs)):
            if greater_than_correct_time_slices_idxs[idx]:
                original_freqs = freqs[idx]
                best_combination: FloatArray = original_freqs
                best_combination_score: float = -1
                for comb in combinations(original_freqs, recording.number_of_voices):
                    score: float = 0
                    comb_freqs = np.array(tuple([0] * (current_number_of_voices - recording.number_of_voices)) + comb).reshape(1, -1)
                    score += self._correct_time_slices_only_model_weight * correct_time_slices_only_model.score_samples(comb_freqs)[0]
                    score += self._all_time_slices_model_weight * all_time_slices_model.score_samples(comb_freqs)[0]
                    if score > best_combination_score:
                        best_combination_score = score
                        best_combination = comb_freqs
                new_freqs[idx] = best_combination
        new_freqs = new_freqs[:, -recording.number_of_voices :]
        return times, new_freqs

    def get_stage_name(self) -> str:
        return "most_likely_voices_filter"
