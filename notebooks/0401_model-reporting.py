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
# # Model reporting
#
# Here we check the reporting of a given model.

# %% [markdown]
# ## Imports

# %%
import textwrap
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
from gcages.cmip7_scenariomip.pre_processing import (
    get_required_model_region_index_input,
    get_required_world_index_input,
)
from gcages.completeness import get_missing_levels
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
)

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"
output_dir: str = "../data/reporting-checking"

# %%
output_dir_model = Path(output_dir) / SCENARIO_TIME_ID / model.replace(".", "-").replace(" ", "-")
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %%
pd.set_option("display.max_colwidth", None)

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
SCENARIO_DB = OpenSCMDB(
    db_dir=SCENARIO_PATH / SCENARIO_TIME_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

SCENARIO_DB.load_metadata().shape

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Scenarios

# %%
model_raw = SCENARIO_DB.load(pix.isin(model=model), progress=True)
if model_raw.empty:
    raise AssertionError

# model_raw

# %% [markdown]
# ## Check completeness

# %% [markdown]
# Extract the model data, keeping:
#
# - only reported timesteps
# - only data from 2015 onwards (we don't care about data before this)
# - only data up to 2100 (while reporting is still weird for data post 2100)

# %%
model_df = model_raw.loc[:, 2015:2100].dropna(how="all", axis="columns")
if model_df.empty:
    raise AssertionError
# model_df

# %% [markdown]
# ### Figure out the model-specific regions

# %%
model_regions = [r for r in model_df.pix.unique("region") if r.startswith(model.split(" ")[0])]
if not model_regions:
    raise AssertionError
# model_regions

# %%
missing_world = get_missing_levels(model_df.index, get_required_world_index_input(), unit_col="unit")
if missing_world.empty:
    print("Nothing missing at the World level")

else:
    missing_world_df = missing_world.to_frame(index=False)
    missing_world_df.to_csv(output_dir_model / "missing-world.csv", index=False)

    print("The following timeseries are missing at the World level")
    display(missing_world_df)  # noqa: F821

# %%
model_region_missing = get_missing_levels(
    model_df.index, get_required_model_region_index_input(model_regions), unit_col="unit"
)

if model_region_missing.empty:
    print("Nothing missing at the regional level")

else:
    model_region_missing_df = model_region_missing.to_frame(index=False)
    model_region_missing_df.to_csv(output_dir_model / "missing-model-region.csv", index=False)

    print("The following timeseries are missing at the model region level")
    display(model_region_missing_df)  # noqa: F821

# %%
if not model_region_missing.empty:
    model_region_missing_variables = model_region_missing_df["variable"].unique()
    all_regions_missing_the_same = (
        model_region_missing_df.groupby("region")["variable"]
        .apply(lambda x: set(x) == set(model_region_missing_variables))
        .all()
    )

    if all_regions_missing_the_same:
        print("All regions are missing the same variables")

    else:
        missing_by_region = model_region_missing_df.groupby("region")["variable"].apply(lambda x: x.values)
        missing_in_all_regions = set(missing_by_region.iloc[0])
        for mr in missing_by_region:
            missing_in_all_regions = missing_in_all_regions.intersection(set(mr))

        print("Missing in all regions")
        print("======================")
        print()
        print(textwrap.indent("\n".join(sorted(missing_in_all_regions)), prefix="- "))

        print()
        print("Missing for specific regions")
        print("============================")
        for region, mr in missing_by_region.items():
            region_missing_specific = set(mr) - missing_in_all_regions
            print()
            print(region)
            if region_missing_specific:
                print(textwrap.indent("\n".join(sorted(region_missing_specific)), prefix="- "))
            else:
                print("* Nothing missing beyond what is missing in all regions")
