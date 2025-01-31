"""
Helpers for pandas

TODO: push this into pandas indexing
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def multi_index_match(
    df: pd.DataFrame, idx: pd.MultiIndex
) -> np.typing.NDArray[np.bool]:
    """
    Multi-index match

    This works even if the order of levels in `idx` is not the same as in `df`.

    Parameters
    ----------
    df
        Data from which to look up rows

    idx
        Multi-index to use for the look up

    Returns
    -------
    :
        Array of indexes of `df` that are in `idx`.
    """
    # TODO: add examples to docstring
    # BUG: The fact that you need this suggests there is a bug in pandas to me.
    idx_reordered = df.index.reorder_levels(
        [*idx.names, *(set(df.index.names) - {*idx.names})]
    )
    return idx_reordered.isin(idx)


def multi_index_lookup(df: pd.DataFrame, idx: pd.MultiIndex) -> pd.DataFrame:
    """
    Multi-index look up

    This works even if the order of levels in `idx` is not the same as in `df`.

    Parameters
    ----------
    df
        Data from which to look up rows

    idx
        Multi-index to use for the look up

    Returns
    -------
    :
        Rows of `df` that overlap with the values in `idx`.
    """
    # TODO: add examples to docstring
    return df.loc[multi_index_match(df, idx)]
