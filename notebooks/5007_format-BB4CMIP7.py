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
# # Format BB4CMIP7 data
#
# Format the processed biomass burning emissions.

# %% [markdown]
# ## Imports

# %%

import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pint
import tqdm.auto
import xarray as xr
from gcages.index_manipulation import set_new_single_value_levels

from emissions_harmonization_historical.constants_5000 import (
    BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR,
    BB4CMIP7_PROCESSED_DB,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.units import assert_units_match_wishes

# %% [markdown]
# ## Setup

# %%
# set unit registry
pix.units.set_openscm_registry_as_default()

# %%
pandas_openscm.register_pandas_accessor()

# %%
# mapping from GFED variables to longer IAMC-style naming
sector_mapping = {
    "AGRI": "Agricultural Waste Burning",
    "BORF": "Forest Burning",
    "DEFO": "Forest Burning",
    "PEAT": "Peat Burning",
    "SAVA": "Grassland Burning",
    "TEMF": "Forest Burning",
}
species_mapping = {
    "BC": "BC",
    "CH4": "CH4",
    "CO": "CO",
    "CO2": "CO2",
    "N2O": "N2O",
    "NH3": "NH3",
    "NMVOC": "VOC",
    "NOx": "NOx",
    "OC": "OC",
    "SO2": "Sulfur",
}

# %% [markdown]
# ## Load data

# %%
raw_files = tuple(BB4CMIP7_ANNUAL_SECTORAL_COUNTRY_OUTPUT_DIR.glob("*.nc"))
res_l = []
for f in tqdm.auto.tqdm(raw_files):
    species, sector = f.stem.split("_")

    sector_iamc = sector_mapping[sector]
    species_iamc = species_mapping[species]
    unit = f"kg {species}/yr"
    variable = f"Emissions|{species_iamc}|{sector_iamc}"

    tmp = xr.open_dataarray(f)
    tmp = tmp.to_dataframe(name="value").set_index("sector", append=True)["value"].unstack("year")
    tmp.index = tmp.index.rename({"iso": "region"})
    #     display(tmp)
    tmp = set_new_single_value_levels(
        tmp, {"scenario": HISTORY_SCENARIO_NAME, "model": "BB4CMIP7", "unit": unit, "variable": variable}
    )

    if species in ["NOx"]:
        with pint.get_application_registry().context("NOx_conversions"):
            tmp = tmp.pix.convert_unit({unit: "Mt NO2/yr"})

    elif species in ["N2O"]:
        tmp = tmp.pix.convert_unit({unit: unit.replace("kg", "kt")})

    elif species in ["NMVOC"]:
        tmp = tmp.pix.convert_unit({unit: unit.replace("kg", "Mt").replace("NMVOC", "VOC")})

    else:
        tmp = tmp.pix.convert_unit({unit: unit.replace("kg", "Mt")})

    res_l.append(tmp)

res = pix.concat(res_l)
# res

# %% [markdown]
# ## Check aggregates

# %%
res_aggregate = res.openscm.groupby_except("sector").sum()
# res_aggregate

# %%
# Created this by looking at the expected outputs
# in netCDF files/the notebooks we've run
exp_sums = pd.Series(
    [
        # 1.91,
        # 16.45,
        # 362.89,
        # 7059.48,
        # 1.04,
        # 4.92,
        # 52.87,
        # 13.11,
        20.38,
        2.72,
    ],
    pd.MultiIndex.from_tuples(
        [
            # ("historical", "BB4CMIP7", "Mt BC/yr"),
            # ("historical", "BB4CMIP7", "Mt CH4/yr"),
            # ("historical", "BB4CMIP7", "Mt CO/yr"),
            # ("historical", "BB4CMIP7", "Mt CO2/yr"),
            # ("historical", "BB4CMIP7", "Mt N2O/yr"),
            # ("historical", "BB4CMIP7", "Mt NH3/yr"),
            # ("historical", "BB4CMIP7", "Mt VOC/yr"),
            # ("historical", "BB4CMIP7", "Mt NOx/yr"),
            ("historical", "BB4CMIP7", "Mt OC/yr"),
            ("historical", "BB4CMIP7", "Mt SO2/yr"),
        ],
        names=["scenario", "model", "unit"],
    ),
    name=2021,
)

exp_sums

# %%
pd.testing.assert_series_equal(
    res_aggregate.openscm.groupby_except(["region", "variable"]).sum()[2021].round(2), exp_sums
)

# %% [markdown]
# ## Process

# %%
res_aggregate

# %%
assert_units_match_wishes(res_aggregate)

# %% [markdown]
# ## Save formatted BB4CMIP7 data

# %%
BB4CMIP7_PROCESSED_DB.save(res_aggregate.pix.assign(stage="iso3c"), allow_overwrite=True)
