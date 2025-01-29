"""
Input/output supporting code
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_timeseries_csv(
    fp: Path,
    lower_col_names: bool = True,
    index_columns: list[str] | None = None,
    out_column_type: type | None = None,
) -> pd.DataFrame:
    """
    Load a time series-like CSV

    In other words, a CSV that has metadata columns
    and then some time columns.

    Parameters
    ----------
    fp
        File path to load

    lower_col_names
        Convert the column names to all lower case as part of loading.

        Note, if `lower_col_names` is `True`,
        the column names are converted to lower case
        before the index is set.

    index_columns
        Columns to treat as metadata from the loaded CSV.

        If not provided, we try and infer the columns
        based on whether they look like time columns or not.

    out_column_type
        The type to apply to the output columns
        that are not part of the index.

    Returns
    -------
    :
        Loaded data
    """
    # TODO: tests of other paths e.g. datetime columns,
    # year columns.
    out = pd.read_csv(fp)

    if lower_col_names:
        out.columns = out.columns.str.lower()

    if index_columns is None:
        raise NotImplementedError(index_columns)

    out = out.set_index(index_columns)

    if out_column_type is not None:
        out.columns = out.columns.astype(out_column_type)

    return out
