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

# %% [markdown]
# # Compile historical emissions
#
# Here we compile the historical emissions
# needed for gridding and the global workflow.

# %% [markdown]
# ## Imports

# %%
from functools import partial

import pandas_indexing as pix
import pandas_openscm
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
HARMONISATION_GRIDDING_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"gridding-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
# HARMONISATION_GRIDDING_EMISSIONS_FILE

# %%
HARMONISATION_GLOBAL_WORKFLOW_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"global-workflow-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
# HARMONISATION_GLOBAL_WORKFLOW_EMISSIONS_FILE

# %%
OUT_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load data

# %%
gridding_harmonisation_emissions = load_timeseries_csv(
    HARMONISATION_GRIDDING_EMISSIONS_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if gridding_harmonisation_emissions.empty:
    raise AssertionError

gridding_harmonisation_emissions.columns.name = "year"
# gridding_harmonisation_emissions

# %%
global_workflow_harmonisation_emissions_raw_names = load_timeseries_csv(
    HARMONISATION_GLOBAL_WORKFLOW_EMISSIONS_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if global_workflow_harmonisation_emissions_raw_names.empty:
    raise AssertionError

global_workflow_harmonisation_emissions_raw_names.columns.name = "year"
# global_workflow_harmonisation_emissions

# %%
assert_units_match_wishes(global_workflow_harmonisation_emissions_raw_names)

# %%
global_workflow_harmonisation_emissions_raw_names = update_index_levels_func(
    global_workflow_harmonisation_emissions_raw_names, {"unit": strip_pint_incompatible_characters_from_unit_string}
)
global_workflow_harmonisation_emissions

# %%
global_workflow_harmonisation_emissions = update_index_levels_func(
    global_workflow_harmonisation_emissions_raw_names,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        ),
        "unit": strip_pint_incompatible_characters_from_unit_string,
    },
)
global_workflow_harmonisation_emissions

# %% [markdown]
# ## Combine and save

# %%
harmonisation_emissions_raw_names = pix.concat(
    [gridding_harmonisation_emissions, global_workflow_harmonisation_emissions_raw_names]
).pix.assign(model="history", scenario="history")
# harmonisation_emissions_raw_names

# %%
harmonisation_emissions = pix.concat(
    [gridding_harmonisation_emissions, global_workflow_harmonisation_emissions]
).pix.assign(model="history", scenario="history")
harmonisation_emissions

# %%
harmonisation_emissions.to_csv(OUT_FILE)
OUT_FILE
