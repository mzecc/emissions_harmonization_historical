"""
Helpers for pandas

TODO: push this into pandas indexing
"""

from __future__ import annotations

import pandas as pd


def multi_index_lookup(df: pd.DataFrame, idx: pd.MultiIndex) -> pd.DataFrame:
    # The fact that you need this suggests there is a bug in pandas to me.
    idx_reordered = df.index.reorder_levels(
        [*idx.names, *(set(df.index.names) - {*idx.names})]
    )
    rows_to_get = idx_reordered.isin(idx)

    return df.loc[rows_to_get]
