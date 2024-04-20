import csv
import itertools
import json
import os
import random
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import cast

import librosa
import numpy as np
import numpy.typing as npt
import scipy
from matplotlib import pyplot as plt

from pypolyphonicanalysis.datamodel.tracks.splits import TrainTestValidationSplit
from pypolyphonicanalysis.settings import Settings

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
F0TimesAndFrequencies = tuple[FloatArray, FloatArray]


def check_output_path(output_path: Path) -> None:
    if output_path.is_file():
        raise ValueError("Output path must be a directory")
    output_path.mkdir(parents=True, exist_ok=True)


def get_train_test_validation_split(split_name: str, settings: Settings) -> TrainTestValidationSplit:
    split: TrainTestValidationSplit = json.load(open(Path(settings.data_directory_path).joinpath("training_metadata").joinpath(f"{split_name}.json")))
    return split


def save_train_test_validation_split(split_name: str, split: TrainTestValidationSplit, settings: Settings) -> None:
    json.dump(split, open(Path(settings.data_directory_path).joinpath("training_metadata").joinpath(f"{split_name}.json"), "w"), indent=4)


@cache
def get_random_state(settings: Settings) -> np.random.RandomState:
    return np.random.RandomState(settings.random_seed)


@cache
def get_random_number_generator(settings: Settings) -> random.Random:
    return random.Random(settings.random_seed)


def median_group_delay(y: FloatArray, sr: int, n_fft: int = 2048, rolloff_value: int = -24) -> float:
    """
    From Muda.
    Compute the average group delay for a signal

    Parameters
    ----------
    y : np.ndarray
        the signal

    sr : int > 0
        the sampling rate of `y`

    n_fft : int > 0
        the FFT window size

    rolloff_value : int > 0
        If provided, only estimate the groupd delay of the passband that
        above the threshold, which is the rolloff_value below the peak
        on frequency response.

    Returns
    -------
    mean_delay : float
        The mediant group delay of `y` (in seconds)

    """
    if rolloff_value > 0:
        # rolloff_value must be strictly negative
        rolloff_value = -rolloff_value

    # frequency response
    _, h_ir = scipy.signal.freqz(y, a=1, worN=n_fft, whole=False, plot=None)

    # convert to dB(clip function avoids the zero value in log computation)
    power_ir = 20 * np.log10(np.clip(np.abs(h_ir), 1e-8, 1e100))

    # set up threshold for valid range
    threshold = max(power_ir) + rolloff_value

    _, gd_ir = scipy.signal.group_delay((y, 1), n_fft)

    return float(np.median(gd_ir[power_ir > threshold]) / sr)


def get_estimated_times_and_frequencies_from_salience_map(
    pitch_activation_mat: FloatArray,
    settings: Settings,
    remove_negatives: bool = False,
) -> F0TimesAndFrequencies:
    n_time_frames = pitch_activation_mat.shape[1]
    freq_grid = librosa.cqt_frequencies(
        n_bins=settings.n_octaves * 12 * settings.over_sample,
        fmin=settings.fmin,
        bins_per_octave=settings.bins_per_octave,
    )
    time_grid = librosa.core.frames_to_time(range(n_time_frames), sr=settings.sr, hop_length=settings.hop_length)

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= settings.activation_threshold)
    est_freqs: list[list[float]] = [[] for _ in range(len(time_grid))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freq_grid[f])
    est_freqs_arrays: list[FloatArray] = [np.array(lst) for lst in est_freqs]
    if remove_negatives:
        for arr_idx in range(len(est_freqs_arrays)):
            est_freqs_arrays[arr_idx] = est_freqs_arrays[arr_idx][est_freqs_arrays[arr_idx] > 0]
    max_len = max([arr.shape[0] for arr in est_freqs_arrays])
    freqs = np.zeros((len(est_freqs_arrays), max_len)).astype(np.float32)
    for arr_idx, arr in enumerate(est_freqs_arrays):
        freqs[arr_idx][(max_len - arr.shape[0]) :] = arr
    return time_grid, freqs


def get_voice_times_and_f0s_from_csv(filename: str) -> dict[int, dict[float, float]]:
    voices: dict[int, dict[float, float]] = defaultdict(dict)
    times: list[float] = []
    with open(filename) as fhandle:
        csv_reader = csv.reader(fhandle, delimiter="\t")
        for row in csv_reader:
            time = float(row[0])
            times.append(time)
            for idx, freqstr in enumerate(row[1:]):
                voices[idx][time] = float(freqstr)
    for voice in voices.keys():
        for time in times:
            voices[voice].setdefault(time, 0)
    return voices


def sonify_trajectory_with_sinusoid(traj: FloatArray, sr: int = 44100, amplitude: float = 0.3, smooth_len: int = 11) -> FloatArray:
    audio_len = int(traj[-1][0] * sr)
    if traj.shape[1] < 3:
        confidence = np.zeros(traj.shape[0]).astype(np.float32)
        confidence[traj[:, 1] > 0] = amplitude
    else:
        confidence = traj[:, 2]
    x_soni = np.zeros(audio_len).astype(np.float32)
    amplitude_mod = np.zeros(audio_len)
    sine_len = int(traj[1, 0] * sr)
    t = np.arange(0, sine_len) / sr
    phase = 0
    for idx in np.arange(0, traj.shape[0]):
        cur_f = traj[idx, 1]
        cur_amp = confidence[idx]
        if cur_f == 0:
            phase = 0
            continue
        cur_soni = np.sin(2 * np.pi * (cur_f * t + phase))
        diff = np.maximum(0, (idx + 1) * sine_len - len(x_soni))
        if diff > 0:
            x_soni[idx * sine_len : (idx + 1) * sine_len - diff] = cur_soni[:-diff]
            amplitude_mod[idx * sine_len : (idx + 1) * sine_len - diff] = cur_amp
        else:
            x_soni[idx * sine_len : (idx + 1) * sine_len - diff] = cur_soni
            amplitude_mod[idx * sine_len : (idx + 1) * sine_len - diff] = cur_amp
        phase += cur_f * sine_len / sr
        phase -= 2 * np.round(phase / 2)
    amplitude_mod = np.convolve(amplitude_mod, np.hanning(smooth_len) / np.sum(np.hanning(smooth_len)), "same").astype(np.float32)
    x_soni = x_soni * amplitude_mod
    return x_soni


def save_f0_trajectories_csv(path: Path, times: list[float], freqs: FloatArray) -> None:
    with open(path, "w") as f:
        csv_writer = csv.writer(f, delimiter="\t")
        time: float
        frequencies: FloatArray
        for time, frequencies in zip(times, freqs):
            row = [time]
            row.extend([x for x in frequencies if x > 0])
            csv_writer.writerow(row)


def plot_predictions(
    times: FloatArray,
    freqs: FloatArray,
    name: str,
    output_path: Path,
    figsize: tuple[int, int],
) -> None:
    check_output_path(output_path)
    plt.figure(figsize=figsize)
    plt.title(f"{name}, estimated F0s")
    plt.ylabel("Cents above A1")
    plt.xlabel("Time (sec)")
    freqs_per_voice = freqs.transpose()
    color_cycle = iter(itertools.cycle([".r", ".g", ".b", ".y", ".c", ".m"]))
    for idx in range(len(freqs_per_voice)):
        voice_freqs = freqs_per_voice[idx]
        valid_idxs = voice_freqs > 0
        voice_freqs = voice_freqs[valid_idxs]
        cents = 1200 * np.log2(voice_freqs / librosa.note_to_hz("A1"))
        plt.plot(times[valid_idxs], cents, next(color_cycle), label=f"Voice {idx}")
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{name}.jpg"))
    plt.close()


def save_reconstructed_audio(
    est_times: FloatArray,
    est_freqs: FloatArray,
    name: str,
    output_path: Path,
    settings: Settings,
) -> None:
    check_output_path(output_path)
    freqs_per_voice = est_freqs.transpose()
    number_of_voices = len(freqs_per_voice)
    voice_audios: list[FloatArray] = []
    for idx in range(number_of_voices):
        voice_freqs = freqs_per_voice[idx]
        voice_audio = sonify_trajectory_with_sinusoid(
            np.array(list(zip(est_times, voice_freqs))),
            sr=settings.sr,
            amplitude=1 / number_of_voices,
        )
        voice_path = os.path.join(output_path, f"{name}_reconstruction_voice{idx}.wav")
        scipy.io.wavfile.write(voice_path, settings.sr, voice_audio)
        voice_audios.append(voice_audio)
    joint_path = os.path.join(output_path, f"{name}_reconstruction_allvoices.wav")
    joint_audio = sum(voice_audios)
    scipy.io.wavfile.write(joint_path, settings.sr, joint_audio)


def convert_times_and_freqs_arrays_to_lists(
    est_times: FloatArray,
    est_freqs: list[FloatArray],
) -> tuple[list[float], list[list[float]]]:
    return est_times.tolist(), [arr.tolist() for arr in est_freqs]


def sum_wave_arrays(input_arrays: list[FloatArray]) -> FloatArray:
    input_arrays = [np.copy(input_array) for input_array in input_arrays]
    max_length = max(arr.shape[0] for arr in input_arrays)
    for idx, arr in enumerate(input_arrays):
        if arr.shape[0] < max_length:
            new_arr = np.zeros((max_length,)).astype(np.float32)
            new_arr[: arr.shape[0]] = arr
            input_arrays[idx] = new_arr
    return cast(FloatArray, np.sum(input_arrays, 0))
