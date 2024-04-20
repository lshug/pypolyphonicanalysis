from pathlib import Path

from scipy.io.wavfile import _read_riff_chunk, _skip_unknown_chunk
import struct
import numpy as np
from pydub import AudioSegment
import os
from collections import defaultdict
import shutil

from pypolyphonicanalysis.settings import Settings
from scripts.conservatoire_corpus_tools.conservatoire_corpus_utils import (
    extract_cd_item_track_from_cipher,
    remove_leading_zeros,
    load_raw_catalog_data,
    get_conservatoire_corpus_path,
)

RESPLIT_ALL = True
to_be_cut_tapes = []


def readmarkers(file: Path) -> list[int]:
    fid = open(file, "rb")
    fsize = _read_riff_chunk(fid)
    cue = []
    while fid.tell() < fsize[0]:
        chunk_id = fid.read(4)
        if chunk_id == b"cue ":
            size, numcue = struct.unpack("<ii", fid.read(8))
            for c in range(numcue):
                id, position, datachunkid, chunkstart, blockstart, sampleoffset = struct.unpack("<iiiiii", fid.read(24))
                cue.append(position)
        else:
            _skip_unknown_chunk(fid, False)
    fid.close()
    return cue


def get_tape_name_for_file(tape: str, track_idx: int) -> str:
    return tape


def get_track_number_for_file(tape: str, track_idx: int) -> str:
    original_filename_list: list[tuple[str, str]] = tracks[tape]
    synthetic_name = track_idx + 1
    if tape == "219":
        if track_idx < 8:
            return str(track_idx + 1)
        elif track_idx == 8:
            return "43"
        elif track_idx == 9:
            return "44"
        return str(track_idx - 1)
    return original_filename_list[track_idx][0] if track_idx < len(original_filename_list) else str(synthetic_name)


def get_track_name_for_file(tape: str, track_idx: int) -> str:
    original_filename_list = tracks[tape]
    if tape == "219":
        if track_idx < 8:
            return original_filename_list[track_idx][1]
        elif track_idx == 8:
            return original_filename_list[42][1]
        elif track_idx == 9:
            return original_filename_list[43][1]
        return original_filename_list[track_idx - 2][1]
    return original_filename_list[track_idx][1] if track_idx < len(original_filename_list) else ""


def cut_file(tape: str, corpus_path: Path) -> None:
    out_dir = corpus_path.joinpath(f"tapes/{tape}")
    filename = corpus_path.joinpath(f"tapes/{tape}/{tape}.wav")
    old = AudioSegment.from_wav(filename.as_posix())
    markers = sorted(set((np.array(readmarkers(filename))).tolist()))
    print(
        f"Cutting tape {tape},",
        "found",
        len(markers),
        "markers, catalogue for tape contains",
        len(tracks[tape]),
        "tracks.",
        "!" if len(markers) != len(tracks[tape]) and len(tracks[tape]) != 0 else "",
    )
    markers.append(int(old.frame_count()))
    for idx in range(len(markers) - 1):
        new = old.get_sample_slice(markers[idx], markers[idx + 1])
        new.export(
            out_dir.joinpath(f"{get_tape_name_for_file(tape, idx)}-{get_track_number_for_file(tape, idx)}-{get_track_name_for_file(tape, idx)}.wav").as_posix(),
            format="wav",
        )


def postprocess(tape: str, corpus_path: Path) -> None:
    split_points = {"35": 19, "36": 21}
    if tape == "35" or tape == "36":
        try:
            shutil.rmtree(corpus_path.joinpath(f"tapes/{tape}a"))
        except Exception:
            pass
        try:
            shutil.rmtree(corpus_path.joinpath(f"tapes/{tape}b"))
        except Exception:
            pass
        os.mkdir(corpus_path.joinpath(f"tapes/{tape}a"))
        os.mkdir(corpus_path.joinpath(f"tapes/{tape}b"))
        for f in os.listdir(corpus_path.joinpath(f"tapes/{tape}")):
            if int(f.split("-")[1]) < split_points[tape]:
                shutil.copy(
                    corpus_path.joinpath(f"tapes/{tape}/{f}"),
                    corpus_path.joinpath(f"tapes/{tape}a/{f}"),
                )
            else:
                shutil.copy(
                    corpus_path.joinpath(f"tapes/{tape}/{f}"),
                    corpus_path.joinpath(f"tapes/{tape}b/{f}"),
                )
        shutil.rmtree(corpus_path.joinpath(f"tapes/{tape}"))


settings = Settings()
corpus_path = get_conservatoire_corpus_path(settings)
data = load_raw_catalog_data(settings)


cds: set[str] = set()
tapes: set[str] = set()
tracks: dict[str, list[tuple[str, str]]] = defaultdict(list)
for item in data:
    cd, tape, track = extract_cd_item_track_from_cipher(str(item["Sifri"]))
    tapes.add(remove_leading_zeros(tape))
    tracks[tape].append((track, str(item["saTauri"])))
    item["tape"] = tape
    item["track"] = track
    item["file"] = None

for tracklist in tracks.values():
    tracklist.sort(key=lambda c: c[0])

tracks["35"] = [*tracks["35a"], *tracks["35b"]]
tracks["36"] = [*tracks["36a"], *tracks["36b"]]

if RESPLIT_ALL:
    to_be_cut_tapes = [x for x in os.listdir(corpus_path.joinpath("uncut"))]

for tape in to_be_cut_tapes:
    shutil.rmtree(corpus_path.joinpath(f"tapes/{tape}"), ignore_errors=True)
    shutil.copytree(corpus_path.joinpath(f"uncut/{tape}"), corpus_path.joinpath(f"tapes/{tape}"))
    cut_file(tape, corpus_path)
    os.remove(corpus_path.joinpath(f"tapes/{tape}/{tape}.wav"))
    postprocess(tape, corpus_path)
