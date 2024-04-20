from pathlib import Path

from pydantic import BaseModel, PositiveInt


class Recording(BaseModel, frozen=True):
    name: str
    file_path: Path
    number_of_voices: PositiveInt | None = None
    ground_truth_files: tuple[Path, ...] | None = None
