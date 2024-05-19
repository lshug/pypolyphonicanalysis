from pathlib import Path

from pydantic import BaseModel, PositiveInt


class Recording(BaseModel, frozen=True):
    name: str
    file_path: Path
    number_of_voices: PositiveInt | None = None
    ground_truth_files: tuple[Path, ...] | None = None
    performers: str | None = None
    performer_group_type: str | None = None
    group_leader: str | None = None
    recording_site: str | None = None
    recording_date: str | None = None
    recording_creator: str | None = None
