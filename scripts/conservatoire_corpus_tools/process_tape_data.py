import json
import os
from collections import defaultdict

from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.conservatoire_corpus_utils import (
    load_raw_catalog_data,
    extract_cd_item_track_from_cipher,
    remove_leading_zeros,
    get_conservatoire_corpus_path,
)

settings = Settings()
corpus_path = get_conservatoire_corpus_path(settings)

data = load_raw_catalog_data(settings)

cds: set[str] = set()
tapes: set[str] = set()
tracks: dict[str, list[str]] = defaultdict(list)
extra_cds = set()
for item in data:
    cd, tape, track = extract_cd_item_track_from_cipher(str(item["Sifri"]))
    tapes.add(remove_leading_zeros(tape))
    item["tape"] = tape
    item["track"] = track
    item["file"] = None
    cds.add(cd)
    if "d." in cd:
        extra_cds.add(cd)


found_tapes = [d for d in os.listdir(corpus_path.joinpath("tapes")) if os.path.isdir(corpus_path.joinpath(f"tapes/{d}"))]
missing_tapes = []
for tape in tapes:
    if tape not in found_tapes:
        missing_tapes.append(tape)
        print("Missing tape", tape)


uncatalogued_missing_tapes = []
for idx in range(1, 231):
    variations = [
        str(idx),
        str(idx) + "a",
        str(idx) + "b",
        str(idx) + "g",
        str(idx) + "bg",
    ]
    if not any(variation in found_tapes for variation in variations) and not any(variation in missing_tapes for variation in variations):
        uncatalogued_missing_tapes.append(str(idx))
        print("Uncatalogued missing tape", idx)

uncatalogued_non_missing_tapes = []
for tape in sorted(found_tapes):
    if tape not in tapes:
        uncatalogued_non_missing_tapes.append(tape)
        print("Uncatalogued non-missing tape", tape)

missing_tapes = sorted(
    missing_tapes,
    key=lambda s: int(s.replace("a", "").replace("b", "").replace("g", "")),
)
open(corpus_path.joinpath("tapes/missing_tapes.txt"), "w").write("\n".join(missing_tapes))
uncatalogued_missing_tapes = sorted(
    uncatalogued_missing_tapes,
    key=lambda s: int(s.replace("a", "").replace("b", "").replace("g", "")),
)
open(corpus_path.joinpath("tapes/uncatalogued_missing_tapes.txt"), "w").write("\n".join(uncatalogued_missing_tapes))
uncatalogued_non_missing_tapes = sorted(
    uncatalogued_non_missing_tapes,
    key=lambda s: int(s.replace("a", "").replace("b", "").replace("g", "")),
)
open(corpus_path.joinpath("tapes/uncatalogued_non_missing_tapes.txt"), "w").write("\n".join(uncatalogued_non_missing_tapes))


cut_tapes = sorted(
    [tape for tape in found_tapes],
    key=lambda s: int(s.replace("a", "").replace("b", "").replace("g", "")),
)

all_cut_files = set()
for tape in cut_tapes:
    if tape in uncatalogued_non_missing_tapes:
        continue
    for f in os.listdir(corpus_path.joinpath(f"tapes/{tape}")):
        all_cut_files.add(f"tapes/{tape}/{f}")
print(len(all_cut_files), "available track files")

notfound = []
for item in sorted(
    [d for d in data if d["tape"] in cut_tapes],
    key=lambda d: cut_tapes.index(str(d["tape"])),
):
    tape = str(item["tape"])
    track = str(item["track"])
    tape_variations = [tape, tape.replace("a", "").replace("b", "").replace("g", "")]
    file_list = os.listdir(corpus_path.joinpath(f"tapes/{tape}"))
    found = False
    for f in file_list:
        for tape_variation in tape_variations:
            if (
                f.startswith(f"{tape_variation}-{remove_leading_zeros(track)}.mp3")
                or f.startswith(f"{tape_variation}-{remove_leading_zeros(track)}.wav")
                or f.startswith(f"{tape_variation}-{track}-")
                or f.startswith(f"{tape_variation}-{remove_leading_zeros(track)}-")
            ):
                found = True
                item["file"] = f"tapes/{tape}/{f}"
                all_cut_files.remove(f"tapes/{tape}/{f}")
                break
        if found:
            break
    if not found:
        notfound.append((tape, track, item["Sifri"], item["saTauri"]))
        print("Not found:", tape, track)
open(corpus_path.joinpath("tapes/missing_items.txt"), "w").write("\n".join(str(t) for t in notfound))

print(len(notfound), "tracks not found")


for f in sorted(all_cut_files):
    print("Orphan track", f)
print(len(all_cut_files), "orphan files")

json.dump(
    {item["Sifri"]: item["file"] for item in data if item["file"] is not None},
    open(corpus_path.joinpath("tapes/track_file_dict.json"), "w"),
    indent=4,
)
