import csv
import itertools
import math
import os
import random
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import scipy
from matplotlib import pyplot as plt

from pypolyphonicanalysis.settings import Settings

FloatArray = np.ndarray[Any, np.dtype[np.float32]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
F0TimesAndFrequencies = tuple[FloatArray, list[FloatArray]]


def check_output_path(output_path: Path) -> None:
    if output_path.is_file():
        raise ValueError("Output path must be a directory")
    output_path.mkdir(parents=True, exist_ok=True)


@cache
def get_random_state(settings: Settings) -> np.random.RandomState:
    return np.random.RandomState(settings.random_seed)


@cache
def get_random_number_generator(settings: Settings) -> random.Random:
    return random.Random(settings.random_seed)


def get_estimated_times_and_frequencies_from_salience_map(
    pitch_activation_mat: FloatArray,
    thresh: float,
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

    idx = np.where(peak_thresh_mat >= thresh)
    est_freqs: list[list[float]] = [[] for _ in range(len(time_grid))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freq_grid[f])

    est_freqs_arrays = [np.array(lst) for lst in est_freqs]

    if remove_negatives:
        for i, (tms, fqs) in enumerate(zip(time_grid, est_freqs_arrays)):
            if any(fqs <= 0):
                est_freqs_arrays[i] = np.array([f for f in fqs if f > 0])

    return time_grid, est_freqs_arrays


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


def get_voice_times_and_f0s_from_times_and_freqs(times: FloatArray, freqs: list[FloatArray]) -> dict[int, dict[float, float]]:
    voices: dict[int, dict[float, float]] = defaultdict(dict)
    number_of_voices = max(len(freq_arr) for freq_arr in freqs)
    for time, freq_array in zip(times, freqs):
        for voice_idx in range(number_of_voices):
            if voice_idx < len(freq_array):
                voices[voice_idx][time] = freq_array[voice_idx]
            else:
                voices[voice_idx][time] = 0
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


def save_f0_trajectories_csv(path: Path, times: list[float], freqs: list[FloatArray]) -> None:
    with open(path, "w") as f:
        csv_writer = csv.writer(f, delimiter="\t")
        time: float
        frequencies: FloatArray
        for time, frequencies in zip(times, freqs):
            row = [time]
            row.extend(frequencies)
            csv_writer.writerow(row)


def plot_predictions(
    est_times: FloatArray,
    est_freqs: list[FloatArray],
    name: str,
    output_path: Path,
    figsize: tuple[int, int],
) -> None:
    check_output_path(output_path)
    plt.figure(figsize=figsize)
    plt.title(f"{name}, estimated F0s")
    plt.ylabel("Cents above A1")
    plt.xlabel("Time (sec)")
    voice_times_and_f0s = get_voice_times_and_f0s_from_times_and_freqs(est_times, est_freqs)
    color_cycle = iter(itertools.cycle([".r", ".g", ".b", ".y", ".c", ".m"]))
    for voice, times_and_f0s in voice_times_and_f0s.items():
        times, f0s = zip(*sorted(list(times_and_f0s.items()), key=lambda time_and_f0: time_and_f0[0]))
        times, f0s, cents = zip(*[(time, f0, 1200 * math.log(f0 / librosa.note_to_hz("A1"), 2)) for time, f0 in zip(times, f0s) if f0 != 0])
        plt.plot(times, cents, next(color_cycle), label=f"Voice {voice}")
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{name}.jpg"))
    plt.close()


def save_reconstructed_audio(
    est_times: FloatArray,
    est_freqs: list[FloatArray],
    name: str,
    output_path: Path,
    settings: Settings,
) -> None:
    check_output_path(output_path)
    voice_times_and_f0s = get_voice_times_and_f0s_from_times_and_freqs(est_times, est_freqs)
    number_of_voices = len(voice_times_and_f0s)
    voice_audios: list[FloatArray] = []
    for voice, times_and_f0s in voice_times_and_f0s.items():
        times, f0s = zip(*sorted(list(times_and_f0s.items()), key=lambda time_and_f0: time_and_f0[0]))
        voice_audio = sonify_trajectory_with_sinusoid(
            np.array(list(zip(times, f0s))),
            sr=settings.sr,
            amplitude=1 / number_of_voices,
        )
        voice_path = os.path.join(output_path, f"{name}_reconstruction_voice{voice}.wav")
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


def sum_wav_files(input_files: list[Path], output_path: Path, settings: Settings) -> None:
    input_arrays: list[FloatArray] = []
    for file in input_files:
        arr, _ = librosa.load(file.absolute().as_posix(), sr=settings.sr)
        input_arrays.append(arr)
    max_length = max(arr.shape[0] for arr in input_arrays)
    for idx, arr in enumerate(input_arrays):
        if arr.shape[0] < max_length:
            new_arr = np.zeros((max_length,)).astype(np.float32)
            new_arr[: arr.shape[0]] = arr
            input_arrays[idx] = new_arr
    array_sum = np.sum(input_arrays, 0)
    scipy.io.wavfile.write(output_path, settings.sr, array_sum)
