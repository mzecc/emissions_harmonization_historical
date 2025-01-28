"""
Tests of our database (`gcages.database`)
"""

from __future__ import annotations

import itertools
import re
from functools import partial
from pathlib import Path

import filelock
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.database import GCDB, AlreadyInDBError


def create_test_df(
    *,
    n_scenarios: int,
    n_variables: int,
    n_runs: int,
    timepoints: np.typing.NDArray[np.floating],
    units: str,
) -> pd.DataFrame:
    """
    Create a [`pd.DataFrame`][pandas.DataFrame] to use in testing

    This uses the idea of simple climate model runs,
    where you have a number of scenarios,
    each of which has a number of variables
    from a number of different model runs
    with output for a number of different time points.
    """
    idx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            (
                (s, v, r, units)
                for s, v, r in itertools.product(
                    [f"scenario_{i}" for i in range(n_scenarios)],
                    [f"variable_{i}" for i in range(n_variables)],
                    [i for i in range(n_runs)],
                )
            ),
            columns=["scenario", "variable", "run", "units"],
        )
    )

    n_ts = n_scenarios * n_variables * n_runs
    df = pd.DataFrame(
        50.0
        * np.linspace(0.3, 1, n_ts)[:, np.newaxis]
        * np.linspace(0, 1, timepoints.size)[np.newaxis, :]
        + np.random.default_rng().random((n_ts, timepoints.size)),
        columns=timepoints,
        index=idx,
    )

    return df


def test_save_and_load(tmpdir):
    start = create_test_df(
        n_scenarios=10,
        n_variables=1,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="Mt",
    )

    db = GCDB(tmpdir)

    db.save(start)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        start.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load()

    pd.testing.assert_frame_equal(start, loaded)


def test_save_multiple_and_load(tmpdir):
    db = GCDB(tmpdir)

    all_saved_l = []
    for units in ["Mt", "Gt", "Tt"]:
        to_save = create_test_df(
            n_scenarios=10,
            n_variables=1,
            n_runs=3,
            timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
            units=units,
        )

        db.save(to_save)
        all_saved_l.append(to_save)

    all_saved = pix.concat(all_saved_l)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        all_saved.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load()

    pd.testing.assert_frame_equal(all_saved, loaded)


def test_save_overwrite_error(tmpdir):
    db = GCDB(tmpdir)

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_variables=1,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    dup = cdf(units="m")
    db.save(dup)

    to_save = pix.concat([dup, cdf(units="km")])

    error_msg = re.escape(
        f"The following rows are already in the database:\n{dup.index}"
    )
    with pytest.raises(AlreadyInDBError, match=error_msg):
        db.save(to_save)


def test_save_overwrite_force(tmpdir):
    db = GCDB(Path(tmpdir))

    cdf = partial(
        create_test_df,
        n_scenarios=10,
        n_variables=1,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
    )

    original = cdf(units="m")
    db.save(original)

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load()

    pd.testing.assert_frame_equal(original, loaded)

    original_overwrite = cdf(units="m")
    updated = pix.concat([original_overwrite, cdf(units="km")])

    # With force, we can overwrite
    db.save(updated, allow_overwrite=True)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the new file, but not the original file.
    db_files = list(db.db_dir.glob("*.csv"))
    assert set([f.name for f in db_files]) == {"1.csv", "index.csv"}

    # Check that the data was overwritten with new data
    try:
        pd.testing.assert_frame_equal(original, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same DataFrame,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        updated.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load()

    pd.testing.assert_frame_equal(updated, loaded)


def test_save_overwrite_force_half_overlap(tmpdir):
    db = GCDB(Path(tmpdir))

    cdf = partial(
        create_test_df,
        n_variables=5,
        n_runs=3,
        timepoints=np.array([2010.0, 2020.0, 2025.0, 2030.0]),
        units="m",
    )

    original = cdf(n_scenarios=6)
    db.save(original)

    # Make sure that our data saved correctly
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        original.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load()

    pd.testing.assert_frame_equal(original, loaded)

    original_overwrite = cdf(n_scenarios=3)

    # With force, we can overwrite
    db.save(original_overwrite, allow_overwrite=True)

    # As a helper, check we've got the number of files we expect.
    # This is testing implementation, so could be removed in future.
    # Expect to have the index file plus the new file plus the newly written file
    # (to handle the need to split the original data so we can keep only what we need),
    # but not the original file.
    db_files = list(db.db_dir.glob("*.csv"))
    assert set([f.name for f in db_files]) == {"1.csv", "2.csv", "index.csv"}

    # Check that the data was overwritten with new data
    overlap_idx = original.index.isin(original_overwrite.index)
    overlap = original.loc[overlap_idx]
    try:
        pd.testing.assert_frame_equal(overlap, original_overwrite)
    except AssertionError:
        pass
    else:
        # Somehow got the same values,
        # so there won't be any difference in the db.
        msg = "Test won't do anything"
        raise AssertionError(msg)

    update_exp = pix.concat([original.loc[~overlap_idx], original_overwrite])
    db_metadata = db.load_metadata()
    metadata_compare = db_metadata
    pd.testing.assert_index_equal(
        update_exp.index, metadata_compare, exact="equiv", check_order=False
    )

    loaded = db.load()

    pd.testing.assert_frame_equal(update_exp, loaded)


def test_locking_load(tmpdir):
    db = GCDB(tmpdir)

    # Put some data in the db so there's something to lock
    db.save(
        create_test_df(
            n_scenarios=1,
            n_variables=1,
            n_runs=1,
            timepoints=np.array([10.0]),
            units="Mt",
        )
    )

    # Acquire the lock already here
    with db.index_file_lock:
        # We can't re-acquire it to load
        with pytest.raises(filelock.Timeout):
            db.load(lock_acquire_timeout=0.0)

        with pytest.raises(filelock.Timeout):
            db.load(lock_acquire_timeout=0.1)


def test_locking_load_metadata(tmpdir):
    db = GCDB(tmpdir)

    # Put some data in the db so there's something to lock
    db.save(
        create_test_df(
            n_scenarios=1,
            n_variables=1,
            n_runs=1,
            timepoints=np.array([10.0]),
            units="Mt",
        )
    )

    # Acquire the lock already here
    with db.index_file_lock:
        # We can't re-acquire it to load
        with pytest.raises(filelock.Timeout):
            db.load_metadata(lock_acquire_timeout=0.0)

        with pytest.raises(filelock.Timeout):
            db.load_metadata(lock_acquire_timeout=0.1)


def test_locking_save(tmpdir):
    db = GCDB(tmpdir)

    # Acquire the lock already here
    with db.index_file_lock:
        # We can't re-acquire it to load
        with pytest.raises(filelock.Timeout):
            db.save("not_used", lock_acquire_timeout=0.0)

        with pytest.raises(filelock.Timeout):
            db.save("not_used", lock_acquire_timeout=0.1)


# - test partial loading
# - test metadata loading
# - test checking what is in what
# - test deletion
