"""
Local database implementation

Super simple, mainly designed as a caching helper for simple climate model runs
"""

from __future__ import annotations

from pathlib import Path

import filelock
import pandas as pd
import pandas_indexing as pix
from attrs import define


@define
class GCDB:
    """
    A database for gcages (GCDB)
    """

    db_dir: Path
    """
    Path in which the database is stored

    Both the index and the data files will be written in this directory.
    """

    @property
    def index_file(self) -> Path:
        return self.db_dir / "index.csv"

    @property
    def index_file_lock_path(self) -> Path:
        return f"{self.index_file}.lock"

    @property
    def index_file_lock(self) -> Path:
        return filelock.FileLock(self.index_file_lock_path)

    def get_new_data_file_path(self, file_id: int) -> Path:
        file_path = self.db_dir / f"{file_id}.csv"
        if file_path.exists():
            raise FileExistsError(file_path)

        return file_path

    def load(
        self, lock_acquire_timeout: float = 10.0, progress: bool = False
    ) -> pd.DataFrame:
        # TODO: add locators
        with self.index_file_lock.acquire(timeout=lock_acquire_timeout):
            index = pd.MultiIndex.from_frame(pd.read_csv(self.index_file)).to_frame()

        files_to_load = index["filepath"].unique()

        if progress:
            import tqdm.autonotebook as tqdman

            files_to_load = tqdman.tqdm(files_to_load)

        data_l = [pd.read_csv(f) for f in files_to_load]
        loaded = pix.concat(data_l).set_index(index.index.droplevel("filepath").names)
        # Don't love this, but fine for now while we don't have datetime columns
        loaded.columns = loaded.columns.astype(float)

        return loaded

    def load_metadata(self, lock_acquire_timeout: float = 10.0) -> pd.MultiIndex:
        with self.index_file_lock.acquire(timeout=lock_acquire_timeout):
            index = pd.MultiIndex.from_frame(pd.read_csv(self.index_file))

        res = index.droplevel("filepath")
        return res

    def save(self, data: pd.DataFrame, lock_acquire_timeout: int = 10.0) -> None:
        # Save entire frame into single file.
        # If user wants things broken up,
        # they should do that at their level.
        with self.index_file_lock.acquire(timeout=lock_acquire_timeout):
            if self.index_file.exists():
                index = pd.read_csv(self.index_file)
                # Check for overlaps
                # add new metadata to index

            else:
                index = data.index.to_frame(index=False)
                data_file_path = self.get_new_data_file_path(file_id=0)
                index["filepath"] = data_file_path

            index.to_csv(self.index_file, index=False)
            data.to_csv(data_file_path)
