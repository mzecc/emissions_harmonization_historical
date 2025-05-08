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
# # Process BB4CMIP7 data
#
# Process the biomass burning emissions
# from the BB4CMIP7 data on ESGF.
# In theory, these are the same as GFED4.
# In practice, they're not so we have to do it this way.
# This is extremely painful, we have to deal with a stupid amount of data.
# We download the data, process it, then delete it
# to avoid filling up our disks with 50GB of (mostly) zeros.
# We also run this species by species,
# again to avoid overloading our disks.

# %% [markdown]
# ## Imports

# %%
from functools import partial

import numpy as np
import pandas_indexing as pix
import pooch
import seaborn as sns
import xarray as xr
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    CMIP7_GHG_PROCESSED_DB,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Setup

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
species: str = "BC"


# %% [markdown]
# ## Download data


# %%
def get_download_urls(species: str) -> tuple[str, ...]:
    total_files = (
        f"https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/mon/{species}smoothed/gn/v20250403/{species}smoothed_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_gn_{tp}.nc"
        for tp in [
            "175001-189912",
            "190001-202112",
        ]
    )
    sector_files = (
        f"https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/mon/{species}{suffix}/gn/v20250403/{species}{suffix}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_gn_175001-202112.nc"
        for suffix in [
            "smoothedpercentageAGRI",
            "smoothedpercentageBORF",
            "smoothedpercentageDEFO",
            "smoothedpercentagePEAT",
            "smoothedpercentageSAVA",
            "smoothedpercentageTEMF",
        ]
    )

    res = tuple(
        (
            *total_files,
            *sector_files,
            "https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/fx/gridcellarea/gn/v20250227/gridcellarea_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_gn.nc",
        )
    )

    return res


# %%
downloaded_files_l = []
for download_url in get_download_urls(species):
    downloaded_files_l.append(
        pooch.retrieve(
            download_url,
            known_hash=None,  # from ESGF, assume safe
            fname=download_url.split("/")[-1],
            progressbar=True,
        )
    )

# downloaded_files_l

# %%
alldat = xr.open_mfdataset(downloaded_files_l)
cell_area = alldat["gridcellarea"]


# %%
def to_annual_global_sum(ds: xr.Dataset, variable_of_interest: str, cell_area: xr.DataArray) -> xr.DataArray:
    # Much easier with cf-python
    seconds_per_step = (ds["time_bnds"].sel(bound=1) - ds["time_bnds"].sel(bound=0)).compute() / 1e9

    return (ds[variable_of_interest] * cell_area * seconds_per_step.astype(int)).sum(["latitude", "longitude", "time"])


# %%
# Make sure we can sum correctly
np.testing.assert_allclose(
    to_annual_global_sum(
        alldat.isel(time=range(12)), variable_of_interest=f"{species}smoothed", cell_area=cell_area
    ).compute(),
    float(alldat.attrs["annual_total_first_year_Tg_yr"]) * 1e9,
    atol=0.01 * 1e9,
)

# %%
assert False

# %% [markdown]
# ## Extrapolate

# %%
pdf = inverse_info.loc[pix.ismatch(model="**smooth")].openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    col="variable",
    col_order=sorted(pdf["variable"].unique()),
    col_wrap=3,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %%
inverse_extrapolated = (
    inverse_info.loc[pix.ismatch(model="**smooth")].copy().pix.assign(model="CR-CMIP-1-0-0-inverse-smooth-extrapolated")
)
for y in range(inverse_extrapolated.columns.max(), HARMONISATION_YEAR + 1):
    # Very basic linear extrapolation
    inverse_extrapolated[y] = 2 * inverse_extrapolated[y - 1] - inverse_extrapolated[y - 2]

# Just in case
inverse_extrapolated[inverse_extrapolated < 0.0] = 0.0

# inverse_extrapolated

# %%
pdf = inverse_extrapolated.openscm.to_long_data()
sns.relplot(
    data=pdf,
    x="time",
    y="value",
    col="variable",
    col_order=sorted(pdf["variable"].unique()),
    col_wrap=3,
    kind="line",
    facet_kws=dict(sharey=False),
)

# %% [markdown]
# ## Save

# %% [markdown]
# ### Pre-industrial emissions

# %% [markdown]
# Make sure we can convert the variable names

# %%
update_index_levels_func(
    inverse_extrapolated,
    {
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
CMIP7_GHG_PROCESSED_DB.save(inverse_extrapolated, groupby=["model"], allow_overwrite=True)
