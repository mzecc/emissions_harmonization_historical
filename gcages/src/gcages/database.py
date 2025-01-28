"""
Local database implementation

Super simple, mainly designed as a caching helper for simple climate model runs
"""

from __future__ import annotations

import os
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
import pandas_indexing as pix
from attrs import define


class AlreadyInDBError(ValueError):
    """
    Raised when saving data would overwrite data which is already in the database
    """

    def __init__(
        self, data: pd.DataFrame, already_in_db: np.typing.NDArray[np.bool]
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        data
            Data that we are trying to save

        already_in_db
            Rows that are already in the database
        """
        error_msg = (
            "The following rows are already in the database:\n"
            f"{data.index[already_in_db]}"
        )
        super().__init__(error_msg)


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

    def save(
        self,
        data: pd.DataFrame,
        lock_acquire_timeout: int = 10.0,
        allow_overwrite: bool = False,
    ) -> None:
        # Save entire frame into single file.
        # If user wants things broken up,
        # they should do that before calling `save`.
        with self.index_file_lock.acquire(timeout=lock_acquire_timeout):
            if self.index_file.exists():
                index_existing = pd.read_csv(self.index_file)
                metadata_existing = pd.MultiIndex.from_frame(
                    index_existing.drop("filepath", axis="columns")
                )
                already_in_db = data.index.isin(metadata_existing)
                if already_in_db.any() and not allow_overwrite:
                    raise AlreadyInDBError(data=data, already_in_db=already_in_db)

                file_id = len(index_existing["filepath"].unique())
                data_file_path = self.get_new_data_file_path(file_id=file_id)
                index_data = data.index.to_frame(index=False)
                index_data["filepath"] = data_file_path

                if already_in_db.any():
                    if not allow_overwrite:  # pragma: no cover
                        msg = "Should have already raised above"
                        raise AssertionError(msg)

                    index_existing_keep = _update_index_for_overwrite(
                        db=self,
                        filepaths_existing=index_existing.set_index(
                            metadata_existing.names
                        )["filepath"],
                        data_to_write=data,
                        already_in_db=already_in_db,
                        file_id_base=file_id,
                    )

                    files_to_remove = set(index_existing["filepath"]).difference(
                        set(index_existing_keep["filepath"])
                    )
                    for ftr in files_to_remove:
                        os.remove(ftr)

                    index = pd.concat([index_existing_keep, index_data])

                else:
                    index = pd.concat([index_existing, index_data])

            else:
                data_file_path = self.get_new_data_file_path(file_id=0)
                index = data.index.to_frame(index=False)
                index["filepath"] = data_file_path

            index.to_csv(self.index_file, index=False)
            data.to_csv(data_file_path)


def _update_index_for_overwrite(
    db: GCDB,
    filepaths_existing: pd.Series[Path],
    data_to_write: pd.DataFrame,
    already_in_db: np.typing.NDArray[np.bool],
    file_id_base: int,
) -> pd.MultiIndex:
    remove_loc = filepaths_existing.index.isin(data_to_write.index[already_in_db])

    filepaths_remove = set(filepaths_existing[remove_loc])
    filepaths_keep = set(filepaths_existing[~remove_loc])
    filepaths_overlap = filepaths_remove.intersection(filepaths_keep)

    filepaths_out = filepaths_existing.copy()
    if filepaths_overlap:
        for i, file in enumerate(filepaths_overlap):
            overlap_file_data = pd.read_csv(file).set_index(
                filepaths_existing.index.names
            )
            overlap_idxs = overlap_file_data.index.isin(data_to_write.index)
            non_overlap_data = overlap_file_data[~overlap_idxs]
            non_overlap_data_file_path = db.get_new_data_file_path(
                file_id=file_id_base + i + 1
            )
            non_overlap_data.to_csv(non_overlap_data_file_path)
            filepaths_out.loc[~overlap_idxs] = non_overlap_data_file_path

    index_out = filepaths_out[~remove_loc].reset_index()

    return index_out
