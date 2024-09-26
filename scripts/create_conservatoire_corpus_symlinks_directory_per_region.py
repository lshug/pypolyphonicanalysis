import os
from collections import defaultdict
from pathlib import Path

from pypolyphonicanalysis.settings import Settings
from scripts.conservatoire_corpus_tools.conservatoire_corpus_utils import load_catalog_data, CatalogEntry, get_conservatoire_corpus_path

settings = Settings(data_directory_path="/home/user/PycharmProjects/pypolyphonicanalysis/data")

regions_to_exclude = ["yaraCai-CerqezeTi"]

conservatoire_corpus_catalog = list(load_catalog_data(settings).values())
region_entries_dict: dict[str, list[CatalogEntry]] = defaultdict(list)
omit_count = 0
for entry in conservatoire_corpus_catalog:
    if entry.recording_region is None:
        omit_count += 1
        continue
    if len(entry.instruments) != 0 or (entry.sample_type is not None and "sakravieri" in entry.sample_type):
        omit_count += 1
        continue
    region_entries_dict[entry.recording_region].append(entry)
print(f"Omitted {omit_count}/{len(conservatoire_corpus_catalog)} entries")

corpus_path = get_conservatoire_corpus_path(settings)
regions_path = Path("./regions")
regions_path.mkdir(exist_ok=True)
all_regions_path = Path("./all_regions")
all_regions_path.mkdir(exist_ok=True)

for region, entries in region_entries_dict.items():
    if region in regions_to_exclude or len(entries) == 0:
        continue
    region_path = regions_path.joinpath(region)
    region_path.mkdir(exist_ok=True)
    for entry in entries:
        original_file_path = corpus_path.joinpath(entry.file_path)
        assert original_file_path.exists()
        region_symlink_path = region_path.joinpath(original_file_path.name)
        all_regions_symlink_path = all_regions_path.joinpath(original_file_path.name)
        os.symlink(original_file_path, region_symlink_path)
        os.symlink(original_file_path, all_regions_symlink_path)
