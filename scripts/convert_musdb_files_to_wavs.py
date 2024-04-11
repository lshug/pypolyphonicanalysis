from pathlib import Path

from musdb.tools import musdb_convert

from pypolyphonicanalysis.settings import Settings

settings = Settings()

musdb_path = Path(settings.data_directory_path).joinpath("corpora").joinpath("MUSDB18")
musdb_convert([musdb_path.absolute().as_posix(), musdb_path.joinpath("WAV").absolute().as_posix()])
