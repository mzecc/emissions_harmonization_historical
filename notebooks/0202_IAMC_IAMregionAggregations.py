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

import numpy as np
import pandas as pd

# create IAM region mappings using common-definitions
from emissions_harmonization_historical.constants import DATA_ROOT

# %%
# OPTIONAL
# create the region mapping .csv file using common-definitions
# https://github.com/IAMconsortium/common-definitions
# first, clone this repository to the same path as the emissions_harmonization_historical repository

region_mapping_path = DATA_ROOT.parents[0] / "src/emissions_harmonization_historical/region_mapping.py"
# %run {str(region_mapping_path)}

# %%
cmip7_history_file = DATA_ROOT / Path("combined-processed-output", "cmip7_history.csv")
region_mapping_file = DATA_ROOT / Path("regionmapping", "region_df.csv")

# file name for output
# TODO: add versioning / ID to this file
iamc_commondefinitions_regions_processed_output_file = DATA_ROOT / Path(
    "combined-processed-output", "iamc_regions_cmip7_history.csv"
)
iamc_commondefinitions_regions_history_missing_iso_vsCEDSGFED = DATA_ROOT / Path(
    "combined-processed-output", "iamc_regions_cmip7_history_missing_iso_vsCEDSGFED.csv"
)
iamc_commondefinitions_regions_history_missing_iso = DATA_ROOT / Path(
    "combined-processed-output", "iamc_regions_cmip7_history_missing_iso.csv"
)

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
# run a few tests to ensure processing went as intended
pd.testing.assert_index_equal(history_for_all_iamc_regions.columns, cmip7_history.columns, check_order=False)
np.testing.assert_array_equal(history_for_all_iamc_regions["region"].unique(), region_mapping["model_region"].unique())

# %%
# test whether the aggregation worked, for a few sample regions
for r in [
    "North America (R10)",
    "AIM 3.0|North Africa",
    "COFFEE 1.5|Caspian Sea",
    "GCAM 7.1|Europe_Eastern",
    "IMAGE 3.4|Rest of South America",
    "MESSAGEix-GLOBIOM 2.1-R12|Pacific OECD",
    "REMIND-MAgPIE 3.4-4.8|Latin America and the Caribbean",
    "WITCH 6.0|United States of America",
]:
    # derive test region from aggregated dataframe
    test_df = history_for_all_iamc_regions[history_for_all_iamc_regions["region"] == r]

    # sum over same region from cmip7_history dataframe
    countries = region_mapping.loc[region_mapping["model_region"] == r, "iso_list"]
    test_countries = countries.iloc[0]

    test_data = cmip7_history[(cmip7_history["region"].isin(test_countries))]
    region_df = test_data.groupby(["model", "scenario", "variable", "unit"], as_index=False).sum()

    # Add a new column 'model_region' with the value "AIM 3.0|North Africa"
    region_df["region"] = r

    # reorder columns
    columns = list(region_df.columns)
    columns.insert(2, columns.pop(columns.index("region")))

    # Reassign the DataFrame to the new column order
    region_df = region_df[columns]

    # reindex both dfs
    test_df = test_df.reset_index(drop=True)
    region_df = region_df.reset_index(drop=True)

    # test both aggregates against one another
    pd.testing.assert_frame_equal(test_df, region_df)

# %%
iamc_commondefinitions_regions_processed_output_file.parent.mkdir(exist_ok=True, parents=True)
history_for_all_iamc_regions.to_csv(iamc_commondefinitions_regions_processed_output_file, index=False)
iamc_commondefinitions_regions_processed_output_file

# %% [markdown]
# # Check countries covered by (a) different historical data sources, and (b) different IAMs in common-definitions

# %%
cmip7_history["scenario"].unique()

# %%
# extract list of unique iso codes present in respective historical dataset (post-processing)
hist_sources = cmip7_history["scenario"].unique()

hist_sources_countries = pd.DataFrame(columns=["model", "iso_list"])  # create empty dataframe

for src in hist_sources:
    df = cmip7_history[cmip7_history["scenario"] == src]
    print(src)

    iso_list = []

    for i in np.arange(0, len(df)):
        iso_list.append(df["region"])

    unique = list(set(item for sublist in iso_list for item in sublist))
    unique.sort()

    temp_df = pd.DataFrame({"model": src, "iso_list": [unique]})
    hist_sources_countries = pd.concat([hist_sources_countries, temp_df], ignore_index=True)

hist_sources_countries

# %%
# extract list of unique iso codes present in each IAM, as well as the R5/9/10 region aggregations

iams = region_mapping["model"].unique()

missing_iso = pd.DataFrame(
    columns=["model", "iso_list", "missing_vs_ceds", "missing_vs_gfed", "missing_from_ceds", "missing_from_gfed"]
)

for m in iams:
    df = region_mapping[region_mapping["model"] == m]

    iso_list = []

    for i in np.arange(0, len(df)):
        iso_list.append(df["iso_list"].iloc[i])

    unique = list(set(item for sublist in iso_list for item in sublist))
    unique.sort()

    # compare against ceds and gfed respectively
    # list the iso codes present in the respective historical dataset but not in the IAM region aggregations
    missing_vs_ceds = list(set(hist_sources_countries["iso_list"][0]) - set(unique))
    missing_vs_gfed = list(set(hist_sources_countries["iso_list"][1]) - set(unique))

    missing_from_ceds = list(set(unique) - set(hist_sources_countries["iso_list"][0]))
    missing_from_gfed = list(set(unique) - set(hist_sources_countries["iso_list"][1]))

    temp_df = pd.DataFrame(
        {
            "model": m,
            "iso_list": [unique],
            "missing_vs_ceds": [missing_vs_ceds],
            "missing_vs_gfed": [missing_vs_gfed],
            "missing_from_ceds": [missing_from_ceds],
            "missing_from_gfed": [missing_from_gfed],
        }
    )

    missing_iso = pd.concat([missing_iso, temp_df], ignore_index=True)

missing_iso

# %%
missing_iso.to_csv(iamc_commondefinitions_regions_history_missing_iso_vsCEDSGFED, index=False)

# %%
# also provide as differently formatted dataframe
all_iso = pd.concat([missing_iso, hist_sources_countries])
all_iso = all_iso[["model", "iso_list"]]
all_iso

# %%
# Flatten the list of iso codes and remove duplicates (and sort)
unique_iso = sorted(set(iso for sublist in all_iso["iso_list"] for iso in sublist))

# Create a new DataFrame with unique iso codes as rows
iso_present = pd.DataFrame({"iso": unique_iso})


# Add a column for each model, indicating if the iso code is in the model's iso_list
for model, iso_list in zip(all_iso["model"], all_iso["iso_list"]):
    iso_present[model] = iso_present["iso"].apply(lambda x: x in iso_list)

# %%
iso_present.to_csv(iamc_commondefinitions_regions_history_missing_iso, index=False)
