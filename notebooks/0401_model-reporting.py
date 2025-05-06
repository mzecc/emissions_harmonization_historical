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

import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
import pandas as pd
import pandas_indexing as pix
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
model: str = "REMIND-MAgPIE 3.5-4.10"
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

model_df.columns.name = "year"
# model_df

# %% [markdown]
# ### Figure out the model-specific regions

# %%
model_regions = [r for r in model_df.pix.unique("region") if r.startswith(model.split(" ")[0])]
if not model_regions:
    raise AssertionError
# model_regions

# %%
# # This is how we generated salted data
# from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
# from gcages.cmip7_scenariomip.pre_processing.reaggregation import ReaggregatorBasic
# from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import get_required_timeseries_index
# from gcages.completeness import get_missing_levels
# from gcages.index_manipulation import create_levels_based_on_existing, set_new_single_value_levels
# import numpy as np
# from pandas_openscm.index_manipulation import update_index_levels_func


# reaggregator = ReaggregatorBasic(model_regions=model_regions)
# pre_processor = CMIP7ScenarioMIPPreProcessor(
#     reaggregator=reaggregator,
#     n_processes=None,  # run serially
# )

# required_index = get_required_timeseries_index(
#             model_regions=model_regions,
#             world_region=reaggregator.world_region,
#             region_level=reaggregator.region_level,
#             variable_level=reaggregator.variable_level,
#         )

# saltable_l = []
# saltable_category_l = []
# for scenario, sdf in model_df.groupby("scenario"):
#     if len(saltable_l) > 2:
#         break

#     if any(sp in scenario for sp in saltable_category_l):
#         continue

#     try:
#         pre_processor(sdf)
#         print(scenario)
#         saltable_l.append(sdf)
#         saltable_category_l.append(scenario.split("- ")[-1].split(" ")[0])
#         continue
#     except Exception:
#         pass

#     variable_unit_map = {"|".join(v.split("|")[:2]): u
#  for v, u in sdf.index.droplevel(sdf.index.names.difference(["variable", "unit"])).drop_duplicates().to_list()}
#     def guess_unit(v_in: str) -> str:
#         for k, v in variable_unit_map.items():
#             if v_in.startswith(f"{k}|") or v_in ==k:
#                 return v

#     mls = get_missing_levels(
#         sdf.index,
#         required_index,
#         unit_col=reaggregator.unit_level,
#     )

#     zeros_hack = pd.DataFrame(
#         np.zeros((mls.shape[0], sdf.shape[1])),
#         columns=sdf.columns,
#         index=create_levels_based_on_existing(mls, {"unit": ("variable", guess_unit)})
#     )
#     zeros_hack = set_new_single_value_levels(
# zeros_hack, {"model": model, "scenario": scenario}).reorder_levels(sdf.index.names)

#     sdf_full = pix.concat([sdf, zeros_hack])

#     try:
#         pre_processor(sdf_full)
#         print(f"{scenario} after filling missing required")
#         saltable_l.append(sdf_full)
#         saltable_category_l.append(scenario.split("- ")[-1].split(" ")[0])
#         continue
#     except Exception:
#         pass

# salted = pix.concat(saltable_l)
# salted = update_index_levels_func(
#    salted, {"model": lambda _: "model_1", "region": lambda x: x.replace(model, "model_1")})
# salted.to_csv("~/salted.csv")
# salted

# %% [markdown]
# ### Define the required timeseries index
#
# If your model reports domestic aviation at the regional level, this will be fine.
# If not, we'll need to add some different code.

# %%
required_timeseries_index = gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.get_required_timeseries_index(
    model_regions
)
# required_timeseries_index

# %%
missing = get_missing_levels(model_df.index, required_timeseries_index, unit_col="unit")

world_locator = missing.get_level_values("region") == "World"
missing_world = missing[world_locator]
missing_model_region = missing[~world_locator]

# %%
if missing_world.empty:
    print("Nothing missing at the World level")

else:
    missing_world_df = missing_world.to_frame(index=False)
    missing_world_df.to_csv(output_dir_model / "missing-world.csv", index=False)

    print("The following timeseries are missing at the World level")
    display(missing_world_df)  # noqa: F821

# %%
if missing_model_region.empty:
    print("Nothing missing at the regional level")

else:
    missing_model_region_df = missing_model_region.to_frame(index=False)
    missing_model_region_df.to_csv(output_dir_model / "missing-model-region.csv", index=False)

    print("The following timeseries are missing at the model region level")
    display(missing_model_region_df)  # noqa: F821

# %%
if not missing_model_region.empty:
    missing_model_region_variables = missing_model_region_df["variable"].unique()
    all_regions_missing_the_same = (
        missing_model_region_df.groupby("region")["variable"]
        .apply(lambda x: set(x) == set(missing_model_region_variables))
        .all()
    )

    # Slightly slow way to do this, but ok
    missing_by_region = missing_model_region_df.groupby("region")["variable"].apply(lambda x: x.values)
    missing_in_all_regions = set(missing_by_region.iloc[0])
    for mr in missing_by_region:
        missing_in_all_regions = missing_in_all_regions.intersection(set(mr))

    print("Missing in all regions")
    print("======================")
    print()
    print(textwrap.indent("\n".join(sorted(missing_in_all_regions)), prefix="- "))
    print()

    if all_regions_missing_the_same:
        print("All regions are missing the same variables")

    else:
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
