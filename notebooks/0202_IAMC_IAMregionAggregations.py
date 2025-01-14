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
# create IAM region mappings using common-definitions
from nomenclature import countries
from nomenclature.definition import DataStructureDefinition
from nomenclature.processor import RegionProcessor

from nomenclature.code import Code, MetaCode, RegionCode, VariableCode
from nomenclature.config import CodeListConfig, NomenclatureConfig
from nomenclature.error import ErrorCollector, custom_pydantic_errors, log_error
from nomenclature.nuts import nuts


import pandas as pd

from emissions_harmonization_historical.constants import (
    CEDS_PROCESSING_ID,
    DATA_ROOT
)

# %%
ceds_release = "2024_07_08"

ceds_data_folder = DATA_ROOT / Path("national", "ceds", "data_raw")

ceds_processed_national = DATA_ROOT / Path(
    "national", "ceds", "processed", f"ceds_cmip7_national_{CEDS_PROCESSING_ID}.csv")

definitions = DATA_ROOT / Path(
    "definitions")

mappings = DATA_ROOT / Path(
    "mappings")

ceds_country_list = DATA_ROOT / Path(
    "national", "ceds", "processed", "ceds_country_list.csv")

gfed_country_list = DATA_ROOT / Path(
    "national", "gfed", "processed", "gfed4_country_list.csv")

region_output_file = DATA_ROOT / Path(
    "national", "ceds", "region_df.csv")

mapping_output_file = DATA_ROOT / Path(
    "national", "ceds", "mappings_df.csv")

# %%
# a DSD stores the definitions info for regions, variables, scenarios
dsd = DataStructureDefinition(definitions)
# dsd.region stores a dictionary of <region name>: <RegionCode object>

# %%
# a RegionCode object stores the info for a region defined in the YAML files, such as the model
region_df = pd.DataFrame(
    [(r.name, r.hierarchy, r.countries, r.iso3_codes) for r in dsd.region.values()],
    columns=["name", "hierarchy", "countries", "iso3"]
)
# fill currently empty iso3 column
# Function to fetch ISO3 codes
def get_iso3_list(country_list):
    if not country_list:
        return None
    iso3_list = []
    for country in country_list:
        try:
            iso3 = countries.get(name=country)
            if iso3:
                # iso3 codes are lower caps in the historical datasets
                iso3_code = iso3.alpha_3.lower()
                # make sure serbia and kosovo are handled combined (required because of SSPs) and not double counted
                if iso3_code == 'srb':
                    continue
                elif iso3_code == 'kos':
                    iso3_list.append('srb_ksv')
                else:
                    iso3_list.append(iso3_code)
            else:
                iso3_list.append(None)  # If no match found
        except Exception:
            iso3_list.append(None)
    return iso3_list
# fill the iso3 column
region_df["iso3"] = region_df["countries"].apply(get_iso3_list)

# %%
region_df

# %%
# the RegionProcessor creates the mappings object
# the mappings are defined at the model level in a RegionAggregationMapping object
# each RAM contains, among other things, the list of "common regions" (regions resulting
# from aggregation) and their constituents
rp = RegionProcessor.from_directory(mappings, dsd)
rows = []
for ram in rp.mappings.values():
    rows.extend(
        [
            (
                ram.model,
                common_region.name,
                [ram.rename_mapping[nr] for nr in common_region.constituent_regions],
            )
            for common_region in ram.common_regions
        ]
    )
mappings_df = pd.DataFrame(
    rows,
    columns=["model(s)", "common_region", "constituent_regions"],
)


region_df.to_csv(region_output_file)
mappings_df.to_csv(mapping_output_file)

# %% [markdown]
# check if country codes are missing

# %%
# Extract unique ISO3 codes
unique_iso3 = set()
for entry in region_df['iso3']:
    if entry:  # Skip None values
        unique_iso3.update(entry)  # Add all ISO3 codes from the list to the set

# Convert set to sorted list (optional)
mapping_countries = sorted(unique_iso3)

# %%
ceds_countries = pd.read_csv(ceds_country_list)
gfed_countries = pd.read_csv(gfed_country_list)

# %%
set_ceds = set(ceds_countries['country'])
set_gfed = set(gfed_countries['country'])
set_mapping = set(mapping_countries)
set_gfed - set_mapping
