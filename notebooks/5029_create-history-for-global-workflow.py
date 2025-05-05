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
# # Create history for global workflow

# %% [markdown]
# ## Imports

# %%
from functools import partial

import pandas_indexing as pix
import pandas_openscm
from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    ADAM_ET_AL_2024_PROCESSED_DB,
    CMIP7_GHG_PROCESSED_DB,
    GCB_PROCESSED_DB,
    HISTORY_HARMONISATION_DB,
    VELDERS_ET_AL_2022_PROCESSED_DB,
    WMO_2022_PROCESSED_DB,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Setup

# %%
pandas_openscm.register_pandas_accessor()

# %% [markdown]
# ## Load data

# %% [markdown]
# ### History data

# %%
history_for_gridding_harmonisation = HISTORY_HARMONISATION_DB.load(
    pix.ismatch(purpose="gridding_emissions")
).rename_axis("year", axis="columns")
# history_for_gridding_harmonisation

# %% [markdown]
# ## Compile history for global workflow

# %% [markdown]
# ### Aggregate gridding history

# %%
# TODO: check whether we get the same global sum for all IAM region groupings
model_regions = [r for r in history_for_gridding_harmonisation.pix.unique("region") if "REMIND-MAgPIE 3.5-4.10" in r]
model_regions

# %%
to_gcages_names = partial(
    update_index_levels_func,
    updates={
        "variable": partial(
            convert_variable_name,
            to_convention=SupportedNamingConventions.GCAGES,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
history_for_gridding_harmonisation_aggregated = to_global_workflow_emissions(
    history_for_gridding_harmonisation.loc[pix.isin(region=["World", *model_regions])]
    .pix.assign(model="gridding-emissions")
    .reset_index("purpose", drop=True),
    global_workflow_co2_fossil_sector="Energy and Industrial Processes",
    global_workflow_co2_biosphere_sector="AFOLU",
)
history_for_gridding_harmonisation_aggregated = to_gcages_names(history_for_gridding_harmonisation_aggregated)
history_for_gridding_harmonisation_aggregated

# %% [markdown]
# ### GCB

# %%
gcb = GCB_PROCESSED_DB.load()
# gcb

# %% [markdown]
# ### WMO 2022

# %%
wmo_2022 = WMO_2022_PROCESSED_DB.load()
# wmo_2022

# %% [markdown]
# ### Velders et al., 2022

# %%
velders_et_al_2022 = VELDERS_ET_AL_2022_PROCESSED_DB.load()
# velders_et_al_2022

# %% [markdown]
# ### Adam et al., 2024

# %%
adam_et_al_2024 = ADAM_ET_AL_2024_PROCESSED_DB.load()
# adam_et_al_2024

# %% [markdown]
# ## CMIP inversions

# %%
cmip_inversions = CMIP7_GHG_PROCESSED_DB.load()
# cmip_inversions

# %% [markdown]
# ### Compile

# %%
all_sources = pix.concat(
    [
        history_for_gridding_harmonisation_aggregated,
        adam_et_al_2024,
        cmip_inversions,
        gcb,
        velders_et_al_2022,
        wmo_2022,
    ]
)
# all_sources

# %%
global_variable_sources = {
    "Emissions|BC": "gridding-emissions",
    "Emissions|CF4": "WMO 2022 AGAGE inversions",
    "Emissions|C2F6": "WMO 2022 AGAGE inversions",
    "Emissions|C3F8": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|cC4F8": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C4F10": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C5F12": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C7F16": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C8F18": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|C6F14": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|CH4": "gridding-emissions",
    "Emissions|CO": "gridding-emissions",
    "Emissions|CO2|Biosphere": "Global Carbon Budget",
    "Emissions|CO2|Fossil": "gridding-emissions",
    "Emissions|HFC125": "Velders et al., 2022",
    "Emissions|HFC134a": "Velders et al., 2022",
    "Emissions|HFC143a": "Velders et al., 2022",
    "Emissions|HFC152a": "Velders et al., 2022",
    "Emissions|HFC227ea": "Velders et al., 2022",
    "Emissions|HFC23": "Adam et al., 2024",
    "Emissions|HFC236fa": "Velders et al., 2022",
    "Emissions|HFC245fa": "Velders et al., 2022",
    "Emissions|HFC32": "Velders et al., 2022",
    "Emissions|HFC365mfc": "Velders et al., 2022",
    "Emissions|HFC4310mee": "Velders et al., 2022",
    "Emissions|CCl4": "WMO 2022 projections v20250129 smoothed",
    "Emissions|CFC11": "WMO 2022 projections v20250129 smoothed",
    "Emissions|CFC113": "WMO 2022 projections v20250129 smoothed",
    "Emissions|CFC114": "WMO 2022 projections v20250129 smoothed",
    "Emissions|CFC115": "WMO 2022 projections v20250129 smoothed",
    "Emissions|CFC12": "WMO 2022 projections v20250129 smoothed",
    "Emissions|CH2Cl2": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|CH3Br": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|CH3CCl3": "WMO 2022 projections v20250129 smoothed",
    "Emissions|CH3Cl": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|CHCl3": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|HCFC141b": "WMO 2022 projections v20250129 smoothed",
    "Emissions|HCFC142b": "WMO 2022 projections v20250129 smoothed",
    "Emissions|HCFC22": "WMO 2022 projections v20250129 smoothed",
    "Emissions|Halon1202": "WMO 2022 projections v20250129 smoothed",
    "Emissions|Halon1211": "WMO 2022 projections v20250129 smoothed",
    "Emissions|Halon1301": "WMO 2022 projections v20250129 smoothed",
    "Emissions|Halon2402": "WMO 2022 projections v20250129 smoothed",
    "Emissions|N2O": "gridding-emissions",
    "Emissions|NF3": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|NH3": "gridding-emissions",
    "Emissions|NOx": "gridding-emissions",
    "Emissions|OC": "gridding-emissions",
    "Emissions|SF6": "WMO 2022 AGAGE inversions",
    "Emissions|SO2F2": "CR-CMIP-1-0-0-inverse-smooth-extrapolated",
    "Emissions|SOx": "gridding-emissions",
    "Emissions|NMVOC": "gridding-emissions",
}

# %%
to_reporting_names = partial(
    update_index_levels_func,
    updates={
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.GCAGES,
            to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        )
    },
)

# %%
global_workflow_harmonisation_emissions_l = []
for variable, source in global_variable_sources.items():
    source_loc = pix.isin(variable=[variable]) & pix.isin(model=[source])
    to_keep = all_sources.loc[source_loc]
    if to_keep.empty:
        msg = f"{variable} not available from {source}"
        raise AssertionError(msg)

    global_workflow_harmonisation_emissions_l.append(to_keep)

global_workflow_harmonisation_emissions = (
    pix.concat(global_workflow_harmonisation_emissions_l).sort_index().sort_index(axis="columns")
)

global_workflow_harmonisation_emissions_reporting_names = to_reporting_names(global_workflow_harmonisation_emissions)
global_workflow_harmonisation_emissions_reporting_names = update_index_levels_func(
    global_workflow_harmonisation_emissions_reporting_names, {"unit": lambda x: x.replace("HFC4310mee", "HFC4310")}
)

# TODO: turn back on
# exp_n_timeseries = 52
# if global_workflow_harmonisation_emissions.shape[0] != exp_n_timeseries:
#     raise AssertionError

global_workflow_harmonisation_emissions_reporting_names

# %% [markdown]
# ## Last checks

# %%
if global_workflow_harmonisation_emissions_reporting_names[HARMONISATION_YEAR].isnull().any():
    missing = global_workflow_harmonisation_emissions_reporting_names.loc[
        global_workflow_harmonisation_emissions_reporting_names[HARMONISATION_YEAR].isnull()
    ]

    display(missing)  # noqa: F821
    raise AssertionError

# %% [markdown]
# ## Save

# %%
HISTORY_HARMONISATION_DB.save(
    global_workflow_harmonisation_emissions_reporting_names.pix.assign(purpose="global_workflow_emissions"),
    allow_overwrite=True,
)
