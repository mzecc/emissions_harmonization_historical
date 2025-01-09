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

import openscm_units
import pandas as pd
import pandas_indexing as pix
from pandas_indexing.core import isna

from emissions_harmonization_historical.ceds import add_global, get_map, read_CEDS
from emissions_harmonization_historical.constants import CEDS_PROCESSING_ID, DATA_ROOT

# set unit registry
pix.units.set_openscm_registry_as_default()


# %%
ur = openscm_units.unit_registry
Q = openscm_units.unit_registry.Quantity

# %% [markdown]
# Set paths

# %%
ceds_release = "2024_07_08"
ceds_data_folder = DATA_ROOT / Path("national", "ceds", "data_raw")
ceds_sector_mapping_file = DATA_ROOT / Path("national", "ceds", "data_aux", "sector_mapping.xlsx")
ceds_processed_output_file_national = DATA_ROOT / Path(
    "national", "ceds", "processed", f"ceds_cmip7_national_{CEDS_PROCESSING_ID}.csv"
)
ceds_processed_output_file_global = DATA_ROOT / Path(
    "national", "ceds", "processed", f"ceds_cmip7_global_{CEDS_PROCESSING_ID}.csv"
)

# %% [markdown]
# Specify species to processes

# %%
# use all species covered in CEDS
species = [
    "BC",
    "CH4",
    "CO",
    "CO2",
    "N2O",  # new, to have regional, was global in CMIP6
    "NH3",
    "NMVOC",  # assumed to be equivalent to IAMC-style reported VOC
    "NOx",
    "OC",
    "SO2",
]

# %% [markdown]
# Load sector mapping of emissions species

# %%
ceds_mapping = pd.read_excel(ceds_sector_mapping_file, sheet_name="CEDS Mapping 2024")
ceds_map = get_map(ceds_mapping, "59_Sectors_2024")  # note; with 7BC now added it is actually 60 sectors, not 59?!
ceds_map.to_frame(index=False)

# %% [markdown]
# Read CEDS emissions data

# %%
ceds = pd.concat(
    read_CEDS(Path(ceds_data_folder) / f"{s}_CEDS_emissions_by_country_sector_v{ceds_release}.csv") for s in species
).rename_axis(index={"region": "country"})
ceds = ceds.pix.semijoin(ceds_map, how="outer")
ceds.loc[isna].pix.unique(["sector_59", "sector"])  # print sectors with NAs

# %%
ceds = ceds.pix.dropna(subset=["units"]).pix.format(unit="{units}/yr", drop=True)  # adjust units
ceds = pix.units.convert_unit(ceds, lambda x: "Mt " + x.removeprefix("kt").strip())  # adjust units

# %%
ceds = ceds.groupby(["em", "country", "unit", "sector"]).sum().pix.fixna()  # group and fix NAs

# %%
# aggregate countries where this is necessary, e.g. because of specific other data (like SSP socioeconomic driver data)
# TODO: check this based on the new SSP data (because old SSP data only had the sum); so recheck.
country_combinations = {"isr_pse": ["isr", "pse"], "sdn_ssd": ["ssd", "sdn"], "srb_ksv": ["srb", "srb (kosovo)"]}
ceds = ceds.pix.aggregate(country=country_combinations)

# %%
# add global
ceds = add_global(ceds)

# %%
ceds_reformatted = ceds.rename_axis(index={"em": "variable", "country": "region"})
ceds_reformatted

# %%
# rename to IAMC-style variable names including standard index order
ceds_reformatted_iamc = (
    ceds_reformatted.rename(index={"SO2": "Sulfur"}, level="variable")
    .pix.format(variable="Emissions|{variable}|{sector}", drop=True)
    .pix.assign(model="History", scenario=f"CEDSv{ceds_release}")
    .reorder_levels(["model", "scenario", "region", "variable", "unit"])
).sort_values(by=["region", "variable"])
ceds_reformatted_iamc

# %%
ceds_reformatted_iamc

# %%
unit_wishes = (
    # ("variable", "unit")
    ("BC", "Mt BC/yr"),
    ("CH4", "Mt CH4/yr"),
    ("N2O", "Mt N2O/yr"),
    ("CO", "Mt CO/yr"),
    ("CO2", "Mt CO2/yr"),
    ("NH3", "Mt NH3/yr"),
    ("NMVOC", "Mt NMVOC/yr"),
    ("NOx", "kt N2O/yr"),
    ("OC", "Mt OC/yr"),
    ("Sulfur", "Mt SO2/yr"),
)


# %%
def get_conv_factor(species, start_unit, target_unit):
    """
    Get conversion factor
    """
    if species == "OC" and (start_unit == "Mt C/yr" and target_unit == "Mt OC/yr"):
        return 1.0

    if species == "BC" and (start_unit == "Mt C/yr" and target_unit == "Mt BC/yr"):
        return 1.0

    if species == "NOx" and (start_unit == "Mt NO2/yr" and target_unit == "kt N2O/yr"):
        with ur.context("NOx_conversions"):
            interim = Q(1, start_unit).to("kt N / yr")

        with ur.context("N2O_conversions"):
            return interim.to(target_unit).m

    return Q(1, start_unit).to(target_unit).m


# %%
# I have no idea how I'm meant to navigate this better, anyway
ceds_reformatted_iamc_desired_units = ceds_reformatted_iamc.copy()
for species, target_unit in unit_wishes:
    locator = pix.ismatch(variable=f"Emissions|{species}|**")
    current_unit = ceds_reformatted_iamc_desired_units.loc[locator].index.get_level_values("unit").unique()
    if len(current_unit) != 1:
        raise AssertionError(current_unit)

    current_unit = current_unit[0]

    if current_unit == target_unit:
        continue

    conversion_factor = get_conv_factor(species=species, start_unit=current_unit, target_unit=target_unit)
    print(f"{species=} {current_unit=} {target_unit=} {conversion_factor=}")

    tmp = ceds_reformatted_iamc_desired_units.loc[locator].copy()
    ceds_reformatted_iamc_desired_units = ceds_reformatted_iamc_desired_units.loc[~locator]

    tmp *= conversion_factor
    tmp = pix.assignlevel(tmp, unit=target_unit)

    ceds_reformatted_iamc_desired_units = pd.concat([tmp, ceds_reformatted_iamc_desired_units])
    # break

ceds_reformatted_iamc_desired_units = ceds_reformatted_iamc_desired_units.sort_index()
ceds_reformatted_iamc_desired_units

# %% [markdown]
# Save formatted CEDS data

# %%
out_global = ceds_reformatted_iamc_desired_units.loc[pix.isin(region="World")]
out_national = ceds_reformatted_iamc_desired_units.loc[~pix.isin(region="World")]
out_global

# %%
assert out_national.shape[0] + out_global.shape[0] == ceds_reformatted_iamc_desired_units.shape[0]

# %%
ceds_processed_output_file_national.parent.mkdir(exist_ok=True, parents=True)
out_national.to_csv(ceds_processed_output_file_national)
ceds_processed_output_file_national

# %%
ceds_processed_output_file_global.parent.mkdir(exist_ok=True, parents=True)
out_global.to_csv(ceds_processed_output_file_global)
ceds_processed_output_file_global
