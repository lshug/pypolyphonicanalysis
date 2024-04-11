from typing import BinaryIO, Final
import struct
from pydub import AudioSegment
import os
import sys
from gooey import Gooey, GooeyParser

DESCRIPTION = """Splits WAV files according to cue markers.
Developed as a part of the "Computational System for Analysis and Modelling of Georgian Traditional
Music Based On Archival Field Recordings" project for the Tbilisi State Conservatoire.
"""

LICENSE = """
License:
    MIT License

    Copyright (c) 2024 Levan Shugliashvili

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

IS_BIG_ENDIAN: Final[bool] = sys.byteorder != "little"


def skip_unknown_chunk(fid: BinaryIO, is_big_endian: bool) -> None:
    if is_big_endian:
        fmt = ">I"
    else:
        fmt = "<I"
    data = fid.read(4)
    if data:
        size = struct.unpack(fmt, data)[0]
        fid.seek(size, 1)
        if size % 2:
            fid.seek(1, 1)


def read_riff_chunk(fid: BinaryIO) -> tuple[int, bool]:
    str1 = fid.read(4)
    if str1 == b"RIFF":
        is_big_endian = False
        fmt = "<I"
    elif str1 == b"RIFX":
        is_big_endian = True
        fmt = ">I"
    else:
        raise ValueError(f"File format {repr(str1)} not understood. Only " "'RIFF' and 'RIFX' supported.")

    # Size of entire file
    file_size = struct.unpack(fmt, fid.read(4))[0] + 8

    str2 = fid.read(4)
    if str2 != b"WAVE":
        raise ValueError(f"Not a WAV file. RIFF form type is {repr(str2)}.")

    return file_size, is_big_endian


def read_markers(file: str) -> list[int]:
    """
    Based on https://stackoverflow.com/a/20396562
    """
    fid = open(file, "rb")
    fsize = read_riff_chunk(fid)
    cue = []
    while fid.tell() < fsize[0]:
        chunk_id = fid.read(4)
        if chunk_id == b"cue ":
            size, numcue = struct.unpack("<ii", fid.read(8))
            for c in range(numcue):
                id, position, datachunkid, chunkstart, blockstart, sampleoffset = struct.unpack("<iiiiii", fid.read(24))
                cue.append(position)
        else:
            skip_unknown_chunk(fid, IS_BIG_ENDIAN)
    fid.close()
    return cue


@Gooey(
    program_name="WAV Cue Splitter",
    menu=[
        {
            "name": "Help",
            "items": [
                {
                    "type": "AboutDialog",
                    "menuTitle": "About",
                    "name": "WAV Cue Splitter",
                    "description": DESCRIPTION,
                    "version": "1.0.0",
                    "copyright": "2024",
                    "developer": "Levan Shugliashvili",
                    "license": LICENSE,
                }
            ],
        }
    ],
)  # type: ignore
def main() -> None:
    parser = GooeyParser(description="WAV cue splitter")
    parser.add_argument("filename", widget="FileChooser")
    parser.add_argument("output_directory", widget="DirChooser")
    args = parser.parse_args()
    filename = args.filename
    filename_without_extension = os.path.split(filename)[1].replace(".wav", "").replace(".WAV", "")
    output_directory = args.output_directory
    markers: list[int] = sorted(set(read_markers(filename)))
    unsplit_audio = AudioSegment.from_wav(filename)

    print(f"{filename}: {unsplit_audio.frame_count()} samples at {unsplit_audio.frame_rate}Hz sample rate.")
    print(f"Found {len(markers)} cues.")
    print("Cue samples:", markers)
    print("Cue seconds:", [marker / unsplit_audio.frame_rate for marker in markers])

    markers.append(unsplit_audio.frame_count() + 1)

    for idx in range(len(markers) - 1):
        try:
            output_path = os.path.join(output_directory, f"{filename_without_extension}-{idx + 1}.wav")
            split_audio = unsplit_audio.get_sample_slice(markers[idx], markers[idx + 1])
            split_audio.export(output_path, format="wav")
            print(f"Exported track {idx+1} to {output_path}")
        except Exception:
            print(f"Failed to export track {idx+1}")
    print("Done.")


main()
