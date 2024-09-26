import json
from collections import defaultdict
from pathlib import Path

from pydub import AudioSegment

from pypolyphonicanalysis.settings import Settings
from scripts.conservatoire_corpus_tools.conservatoire_corpus_utils import (
    load_raw_catalog_data,
    extract_cd_item_track_from_cipher,
    CatalogEntry,
    get_conservatoire_corpus_path,
    GroupType,
    Instrument,
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


def process_recording_region(field_str: str | None) -> str | None:
    if field_str is None:
        return None
    region_correction_dict: dict[str, str] = {
        "xando(ukanamxari)": "qarTli",
        "Telavi": "kaxeTi",
        "Sromisubani": "guria",
        "akTi": "guria",
        "afxazeTi assr": "afxazeTi",
        "afxazeTis assr": "afxazeTi",
    }
    region_str = field_str.split(",")[0]
    region_str = region_correction_dict.get(region_str, region_str)
    return region_str


def process_group_type(field_str: str | None, uncertain_fields: set[str], field_name: str) -> GroupType | None:
    if field_str is None or field_str == "?" or field_str.strip() == "":
        return None
    if "?" in field_str:
        uncertain_fields.add(field_name)
    if "solo" in field_str:
        return GroupType.SOLO
    if "qalTa" in field_str and "mamakacTa" in field_str:
        return GroupType.MIXED
    if "bavSvTa" in field_str:
        return GroupType.CHILDRENS
    if "Sereuli" in field_str:
        return GroupType.MIXED
    if "qalTa" in field_str:
        return GroupType.WOMENS
    if "mamakacTa" in field_str:
        return GroupType.MENS
    if "monacvle" in field_str:
        return GroupType.ALTERNATING
    elif "SeuzRudavi" in field_str:
        return GroupType.UNRESTRICTED
    return None


def process_repertoire_group_type(field_str: str | None, uncertain_fields: set[str]) -> GroupType | None:
    return process_group_type(field_str, uncertain_fields, "repertoire_group_type")


def process_performer_group_type(field_str: str | None, uncertain_fields: set[str]) -> GroupType | None:
    return process_group_type(field_str, uncertain_fields, "performer_group_type")


def process_instrument(field_str: str | None, uncertain_fields: set[str]) -> list[Instrument]:
    if field_str is None:
        return []
    field_str = field_str.replace("xarsaA", "xarsa?")
    if "?" in field_str or ".." in field_str:
        uncertain_fields.add("instruments")
    field_str = field_str.replace("?", "").replace(", ..", "")
    if "2 an sami " in field_str:
        uncertain_fields.add("instruments")
        field_str = field_str.replace("2 an sami ", "")
    if "0" in field_str:
        return []
    if field_str == "simebiani":
        uncertain_fields.add("instruments")
    field_str = field_str.replace("salamuri, ueno", "salamuri_ueno")
    field_str = field_str.replace("salamuri ueno", "salamuri_ueno")
    field_str = field_str.replace("dasartyami, klaviSebiani", "dasartyami_klaviSebiani")
    field_str = field_str.replace(" (Congurebi)", "")
    field_str = field_str.replace("Congurebi", "Conguri")
    field_str = field_str.replace(";", ",")
    field_str = field_str.replace(" fxaCica?", "")
    field_str = field_str.replace("Wuniri da simebiani", "Wuniri, simebiani")
    field_str = field_str.replace("simebiani (xemiani) ", "simebiani (xemiani), ")
    field_str = field_str.replace("simebiani (xemiani)", "simebian-xemiani")
    field_str = field_str.replace("garmoni  daira", "garmoni, daira")
    field_str = field_str.replace("salamuri, 2 an 3 ", "Casaberi, salamuri, ")
    field_str = field_str.replace("2 ", "")
    field_str = field_str.replace("klaviSiani, garmoni, daira", "klaviSiani, garmoni, dasartyami, daira")
    field_str = field_str.replace("garmoni da", "garmoni, ")
    field_str = field_str.replace("Conguri da wyvili", "simebiani, Conguri, ")
    field_str = field_str.replace("fxawCaA", "fxawCa")
    field_str = field_str.replace("forte-piano", "fortepiano")
    field_str = field_str.replace("1", "")
    field_str = field_str.replace(".", "")
    field_str = field_str.replace("Conguri-fanduri", "Conguri, simebiani, fanduri")
    if field_str[-1] == ",":
        field_str = field_str[:-1]
    if "," in field_str:
        number_of_commas = 0
        new_field_str = ""
        for c in field_str:
            if c == "," and number_of_commas % 2 == 0:
                new_field_str += ":"
            else:
                new_field_str += c
            if c == ",":
                number_of_commas += 1
        field_str = new_field_str
    field_str_divisions = [x.strip() for x in field_str.split(",")]
    field_str_divisions = [x.split(":")[int(":" in x)].strip() for x in field_str_divisions]
    instrument_list: list[Instrument] = []
    for division in field_str_divisions:
        for instrument_enum_field in Instrument:
            if instrument_enum_field.name == division:
                instrument_list.append(instrument_enum_field)
    return instrument_list


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
    file_path = track_file_dict[catalog_code]
    _, tape, track = extract_cd_item_track_from_cipher(catalog_code)
    total_duration_seconds = int(AudioSegment.from_wav(corpus_path.joinpath(file_path).as_posix()).duration_seconds)
    entries.append(
        CatalogEntry(
            id=item["ID"],
            title=item["saTauri"],
            catalog_code=catalog_code,
            tape=tape,
            track=track,
            file_name=Path(file_path).name,
            file_path=file_path,
            notes=item["SeniSvna"],
            performers=item["Semsruleblebi"],
            recording_date_year=process_recording_date_year(item["Caweris_weli"], uncertain_fields),
            recording_date_month=process_recording_date_month(item["Caweris_Tve"], uncertain_fields),
            recording_date_day=process_recording_date_day(item["Caweris_ricxvi"], uncertain_fields),
            recording_site=item["Caweris_adgili"],
            recording_region=process_recording_region(item["Caweris_adgili"]),
            recording_creator=item["Camweri"],
            catalog_entry_contributor=item["aRmweris_vinaoba"],
            nonlexical_starting_vocabes=item["sawyisi_uSinaarso_fraza"],
            lexical_starting_vocabes=item["sawyisi_Sinaarsiani_fraza"],
            thematic_and_genre_tags=item["Temat_Janruli_jgufi"],
            # authenticity
            sample_type=item["nimuSis_saxeoba"],
            # performance_type
            # number_of_voices
            # polyphony_form
            instruments=process_instrument(item["sakravi"], uncertain_fields),
            # instrument_tuning
            repertoire_group_type=process_repertoire_group_type(item["repertuaris_jgufi"], uncertain_fields),
            performer_group_type=process_performer_group_type(item["SemsrulebelTa_jgufi"], uncertain_fields),
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
