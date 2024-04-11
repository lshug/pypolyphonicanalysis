import json
from enum import Enum
from pathlib import Path

from typing import TypedDict

from pydantic import Field, PositiveInt, BaseModel

from pypolyphonicanalysis.settings import Settings


class RawCatalogData(TypedDict):
    ID: int
    Semsruleblebi: str | None
    Caweris_ricxvi: str | None
    SeniSvna: str | None
    sityvieri_teqsti: str | None
    notebi: str | None
    rigiTi_nomeri: str | None
    Sifri: str
    saTauri: str
    sawyisi_uSinaarso_fraza: str | None
    sawyisi_Sinaarsiani_fraza: str | None
    Temat_Janruli_jgufi: str | None
    atributika: str | None
    Sesrulebis_auTenturoba: str | None
    nimuSis_saxeoba: str | None
    SemsrulebelTa_raodenoba: str | None
    Sesrulebis_forma: str | None
    xmebis_raodenoba: str | None
    mravalxmianobis_forma: str | None
    sakravi: str | None
    sakravis_wyoba: str | None
    repertuaris_jgufi: str | None
    SemsrulebelTa_jgufi: str | None
    erovnuli_kuTvnileba: str | None
    dialeqti: str | None
    jgufis_xelmZRvaneli: str | None
    Caweris_adgili: str | None
    Caweris_weli: str | None
    Caweris_Tve: str | None
    Camweri: str | None
    teqnikuri_maxaSiaTebeli: str | None
    qronometraJi: str | None
    aRmweris_vinaoba: str | None
    audio_video: str | None
    tape: str | None
    track: str | None
    file: str | None


class ThematicTag(Enum):
    pass


class GenreTag(Enum):
    pass


ThematicAndGenreTag = ThematicTag | GenreTag


class AuthenticityClass(Enum):
    pass


class RecordingType(Enum):
    pass


class PerformingGroupType(Enum):
    pass


class PerformanceType(Enum):
    pass


class PolyphonyForm(Enum):
    pass


class Instrument(Enum):
    pass


class Tuning(Enum):
    pass


class GroupType(Enum):
    pass


class Nationality(Enum):
    pass


class Dialect(Enum):
    pass


class Identity(BaseModel):
    name: str


class CatalogEntry(BaseModel):
    id: int
    title: str
    catalog_code: str
    tape: str
    track: str
    file_path: str

    notes: str | None = None
    # performers: list[Identity] | None = None
    performers: str | None = None
    recording_date_year: PositiveInt | None = Field(None, ge=1900, le=2024)
    recording_date_month: PositiveInt | None = Field(None, ge=1, le=12)
    recording_date_day: PositiveInt | None = Field(None, ge=1, le=31)
    recording_site: str | None = None
    # recording_creator: Identity | None = None
    recording_creator: str | None = None
    # catalog_entry_contributor: Identity | None = None
    catalog_entry_contributor: str | None = None

    nonlexical_starting_vocabes: str | None = None
    lexical_starting_vocabes: str | None = None

    # thematic_and_genre_tags: list[ThematicAndGenreTag] = Field([])
    thematic_and_genre_tags: str | None = None

    authenticity: AuthenticityClass | None = None
    sample_type: RecordingType | None = None
    performing_group_type: GroupType | None = None
    performance_type: PerformanceType | None = None
    number_of_voices: PositiveInt | None = None
    polyphony_form: PolyphonyForm | None = None
    # instruments: list[Instrument] = Field([])
    instruments: str | None = None
    instrument_tuning: Tuning | None = None
    repertoire_group_type: GroupType | None = None
    performer_group_type: GroupType | None = None
    # nationality: Nationality | None = None
    nationality: str | None = None
    # dialect: Dialect | None = None
    dialect: str | None = None
    # group_leader: Identity | None = None
    group_leader: str | None = None
    technical_characteristics: str | None = None
    musical_duration_seconds: int | None = None
    total_duration_seconds: int

    schema_version: int = 1
    fields_with_uncertain_values: set[str] = Field(set())


def get_conservatoire_corpus_path(settings: Settings) -> Path:
    return Path(settings.data_directory_path).joinpath("corpora").joinpath("conservatoire_corpus")


def load_raw_catalog_data(settings: Settings) -> list[RawCatalogData]:
    corpus_path = get_conservatoire_corpus_path(settings)
    data: list[RawCatalogData] = json.load(open(corpus_path.joinpath("data.json")))
    updated_data = json.load(open(corpus_path.joinpath("updates.json")))
    data_id_idx_dict = {d["ID"]: idx for idx, d in enumerate(data)}
    for update in updated_data:
        id = update["ID"]
        idx = data_id_idx_dict.get(id, len(data))
        data[idx] = update
    data = [d for d in data if d["Sifri"] is not None]
    for item in data:
        for k, v in item.items():
            if v == "0" or v == "" or v == "00" or v == "?":
                if k == "saTauri" and v == "?":
                    item[k] = "Unknown"  # type: ignore
                else:
                    item[k] = None  # type: ignore
    return data


def load_catalog_data(settings: Settings) -> list[CatalogEntry]:
    corpus_path = get_conservatoire_corpus_path(settings)
    return [CatalogEntry(**item) for item in json.load(open(corpus_path.joinpath("exported_data.json")))]


def remove_leading_zeros(s: str) -> str:
    if s == "00":
        return "0"
    if s == "0":
        return "0"
    while s[0] == "0":
        s = s[1:]
    return s


def extract_cd_item_track_from_cipher(cipher: str) -> tuple[str, str, str]:
    parts = [cipher.split(",")[0], cipher[cipher.index(",") + 1 :]]
    prefix, suffix = [fix.strip() for fix in parts]
    cd = prefix
    suffix = suffix[2:]
    tape = remove_leading_zeros(suffix.split("-")[0])
    track = suffix[suffix.index("-") + 1 :]

    return cd, tape, track
