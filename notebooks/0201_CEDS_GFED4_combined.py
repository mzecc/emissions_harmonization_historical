# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# import external packages and functions
from pathlib import Path

import pandas as pd
import xarray as xr
from pandas_indexing import set_openscm_registry_as_default
import pandas_indexing as pix
from pandas_indexing.core import isna

from emissions_harmonization_historical.constants import (
    CEDS_PROCESSING_ID,
    DATA_ROOT,
)

# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
ceds_release = "2024_07_08"
gfed_release = "GFED4.1s"

ceds_data_folder = DATA_ROOT / Path("national", "ceds", "data_raw")
gfed_data_folder = DATA_ROOT / Path("national", "gfed", "data_raw")

ceds_processed_national = DATA_ROOT / Path(
    "national", "ceds", "processed", f"ceds_cmip7_national_{CEDS_PROCESSING_ID}.csv")

gfed_processed_national = DATA_ROOT / Path(
    "national", "gfed", "processed", "gfed_cmip7_national_alpha.csv")

processed_output_file_international = DATA_ROOT / Path(
    "national", "ceds_gfed", "processed", f"ceds_gfed_cmip7_international_{CEDS_PROCESSING_ID}.csv")

# %%
gfed = pd.read_csv(gfed_processed_national)
ceds = pd.read_csv(ceds_processed_national)

# %%
gfed['unit'] = gfed['unit'].apply(lambda x: x if x.endswith('/yr') else f"{x}/yr")

# %%
gfed['unit'].unique()

# %%
ceds['unit'].unique()
