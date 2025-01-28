"""
Tests of our database (`gcages.database`)
"""

from __future__ import annotations

import itertools

import filelock
import numpy as np
import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.database import GCDB


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


# - test overwrite-saving
# - test partial loading
# - test metadata loading
# - test checking what is in what
# - test deletion
