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

# create IAM region mappings using common-definitions
from emissions_harmonization_historical.constants import DATA_ROOT

# %%
cmip7_history_file = DATA_ROOT / Path("combined_cmip7_history.csv")
region_mapping_file = DATA_ROOT / Path("regionmapping", "region_df.csv")

# file name for output
# TODO: add versioning / ID to this file
iamc_commondefinitions_regions_processed_output_file = DATA_ROOT / Path("iamc_regions_cmip7_history.csv")

# %%
cmip7_history = pd.read_csv(cmip7_history_file)
cmip7_history

# %%
region_mapping = pd.read_csv(region_mapping_file)
region_mapping = region_mapping.rename(columns={"name": "model_region", "hierarchy": "model", "iso3": "iso_list"})
region_mapping = region_mapping[["model_region", "model", "iso_list"]]
region_mapping = region_mapping.dropna(
    subset=["iso_list"]
)  # don't try to aggregate anything if there are no countries defined for a specific region
region_mapping["iso_list"] = region_mapping["iso_list"].str.lower()
region_mapping["iso_list"] = region_mapping["iso_list"].apply(
    lambda x: x.strip("[]").replace("'", "").split(", ")
)  # transform from Series/string to list-like object which is iterable
region_mapping

# %%
cmip7_history[(cmip7_history["region"].isin(["usa"]))]

# %%
agg_data = []
for _, row in region_mapping.iterrows():
    model_region = row["model_region"]
    model = row["model"]
    iso_list = row["iso_list"]

    # Filter historical data based on the model and region in iso_list
    filtered_history = cmip7_history[(cmip7_history["region"].isin(iso_list))]

    # Group by the remaining columns and aggregate the year columns
    numeric_datacols = filtered_history.select_dtypes(include="number").columns
    aggregated_df_one_region = filtered_history.groupby(["model", "scenario", "variable", "unit"], as_index=False)[
        numeric_datacols
    ].sum()
    aggregated_df_one_region["region"] = model_region

    # append
    agg_data.append(aggregated_df_one_region)


# Concatenate all aggregated DataFrames
history_for_all_iamc_regions = pd.concat(agg_data, ignore_index=True)
# Move 'region' between 'scenario' and 'variable'
columns_order = ["model", "scenario", "region", "variable", "unit", *numeric_datacols]
history_for_all_iamc_regions = history_for_all_iamc_regions[columns_order]

# %%
history_for_all_iamc_regions

# %%
iamc_commondefinitions_regions_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
history_for_all_iamc_regions.to_csv(iamc_commondefinitions_regions_processed_output_file, index=False)
iamc_commondefinitions_regions_processed_output_file
