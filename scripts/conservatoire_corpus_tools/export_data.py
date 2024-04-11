import json
from collections import defaultdict
from pydub import AudioSegment

from pypolyphonicanalysis.settings import Settings
from pypolyphonicanalysis.utils.conservatoire_corpus_utils import (
    load_raw_catalog_data,
    extract_cd_item_track_from_cipher,
    CatalogEntry,
    get_conservatoire_corpus_path,
)


def process_recording_date_year(field_str: str | None, uncertain_fields: set[str]) -> int | None:
    if field_str is None or field_str == "?":
        return None
    if "?" in field_str:
        uncertain_fields.add("recording_date_year")
        return int(field_str.split("?")[0])
    return int(field_str)


def process_recording_date_month(field_str: str | None, uncertain_fields: set[str]) -> int | None:
    if field_str is None or field_str == "?":
        return None
    if "(" in field_str:
        uncertain_fields.add("recording_date_month")
        field_str = field_str.split("(")[1].split(")")[0]
    if "?" in field_str:
        uncertain_fields.add("recording_date_month")
        field_str = field_str.split("?")[0]
    if "-" in field_str:
        uncertain_fields.add("recording_date_month")
        field_str = field_str.split("-")[0]
    return int(field_str)


def process_recording_date_day(field_str: str | None, uncertain_fields: set[str]) -> int | None:
    if field_str is None or field_str == "?":
        return None
    if "-" in field_str:
        uncertain_fields.add("recording_date_day")
        return int(field_str.split("-")[0])
    if "," in field_str:
        uncertain_fields.add("recording_date_day")
        return int(field_str.split(",")[0])
    return int(field_str)


settings = Settings()
corpus_path = get_conservatoire_corpus_path(settings)
data = load_raw_catalog_data(settings)

categories = defaultdict(set)
fields = list(data[0].keys())
for field in fields:
    for item in data:
        data_dict = dict(item)
        categories[field].add(data_dict[field])

uncertain_categories: list[str] = []
nullable_categories: list[str] = []
direct_categories: list[str] = []
for category, values in categories.items():
    nullable = False
    uncertain = False
    if any(isinstance(value, str) and "?" in value for value in values):
        uncertain_categories.append(category)
        uncertain = True
    if any(value == "0" or value is None or (isinstance(value, int) and value == 0) for value in values):
        nullable_categories.append(category)
        nullable = True
    if not nullable and not uncertain:
        direct_categories.append(category)

tags = set()
for x in categories["Temat_Janruli_jgufi"]:
    if not isinstance(x, str):
        continue
    for y in x.split(","):
        tags.add(y.replace("?", ""))


track_file_dict = json.load(open(corpus_path.joinpath("tapes/track_file_dict.json")))
entries: list[CatalogEntry] = []
for item in data:
    uncertain_fields: set[str] = set()
    catalog_code = str(item["Sifri"])
    filename = track_file_dict[catalog_code]
    _, tape, track = extract_cd_item_track_from_cipher(catalog_code)
    total_duration_seconds = int(AudioSegment.from_wav(corpus_path.joinpath(filename).as_posix()).duration_seconds)
    entries.append(
        CatalogEntry(
            id=item["ID"],
            title=item["saTauri"],
            catalog_code=catalog_code,
            tape=tape,
            track=track,
            file_path=filename,
            notes=item["SeniSvna"],
            performers=item["Semsruleblebi"],
            recording_date_year=process_recording_date_year(item["Caweris_weli"], uncertain_fields),
            recording_date_month=process_recording_date_month(item["Caweris_Tve"], uncertain_fields),
            recording_date_day=process_recording_date_day(item["Caweris_ricxvi"], uncertain_fields),
            recording_site=item["Caweris_adgili"],
            recording_creator=item["Camweri"],
            catalog_entry_contributor=item["aRmweris_vinaoba"],
            nonlexical_starting_vocabes=item["sawyisi_uSinaarso_fraza"],
            lexical_starting_vocabes=item["sawyisi_Sinaarsiani_fraza"],
            thematic_and_genre_tags=item["Temat_Janruli_jgufi"],
            # authenticity
            # sample_type
            # performance_type
            # number_of_voices
            # polyphony_form
            instruments=item["sakravi"],
            # instrument_tuning
            # repertoire_group_type
            # performer_group_type
            nationality=item["erovnuli_kuTvnileba"],
            dialect=item["dialeqti"],
            group_leader=item["jgufis_xelmZRvaneli"],
            technical_characteristics=item["teqnikuri_maxaSiaTebeli"],
            # musical_duration_seconds
            total_duration_seconds=total_duration_seconds,
            schema_version=1,
            fields_with_uncertain_values=uncertain_fields | {k for k, v in dict(item).items() if isinstance(v, str) and "?" in v},
        )
    )

json.dump(
    [entry.model_dump_json() for entry in entries],
    open(corpus_path.joinpath("exported_data.json"), "w"),
    indent=4,
)
