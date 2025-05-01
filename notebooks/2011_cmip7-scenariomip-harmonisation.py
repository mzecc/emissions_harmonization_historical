# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Harmonisation
#
# Here we do the harmonisation.
# This is done model by model.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %%
import pandas_indexing as pix
import pandas_openscm
from gcages.aneris_helpers import harmonise_all
from gcages.completeness import assert_all_groups_are_complete
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "REMIND-MAgPIE 3.5-4.10"

# %%
model_clean = model.replace(" ", "_").replace(".", "p")
model_clean

# %%
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""}
HISTORICAL_GRIDDING_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"gridding-historical-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
HISTORICAL_GRIDDING_FILE

# %%
# TODO: load historical non-gridding data
# TODO: split out io function to just load all default historical data
#       so we can use it here and in the actual harmonisation

# %%
IN_DIR = DATA_ROOT / "cmip7-scenariomip-workflow" / "pre-processing" / CMIP7_SCENARIOMIP_PRE_PROCESSING_ID
in_db = OpenSCMDB(
    db_dir=IN_DIR,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

in_db.load_metadata().shape

# %%
OUT_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonised_{model_clean}_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}_{SCENARIO_TIME_ID}_{CMIP7_SCENARIOMIP_PRE_PROCESSING_ID}.csv"
)
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)
# OUT_FILE

# %% [markdown]
# ## Load data

# %%
pre_processed_model = in_db.load(pix.isin(model=model, stage="gridding_emissions"), progress=True).reset_index(
    "stage", drop=True
)
if pre_processed_model.empty:
    raise AssertionError

pre_processed_model

# %%
model_variables_regions = pre_processed_model.pix.unique(["variable", "region"])
# model_variables_regions

# %%
historical_emissions_gridding_model = (
    load_timeseries_csv(
        HISTORICAL_GRIDDING_FILE,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
).openscm.mi_loc(model_variables_regions)

if historical_emissions_gridding_model.empty:
    raise AssertionError

assert_all_groups_are_complete(historical_emissions_gridding_model, model_variables_regions)

historical_emissions_gridding_model

# %%
assert False, "Add overrides here"

# %%
harmonise_all(
    scenarios=pre_processed_model,
    history=historical_emissions_gridding_model,
    year=HARMONISATION_YEAR,
)
