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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Infilling
#
# Here we do the infilling.
# This is done model by model.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from gcages.aneris_helpers import harmonise_all
from gcages.completeness import assert_all_groups_are_complete
from gcages.index_manipulation import split_sectors
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.io import load_timeseries_csv
from functools import partial
from pandas_openscm.index_manipulation import update_index_levels_func
from gcages.renaming import convert_variable_name, SupportedNamingConventions
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
    CMIP7_SCENARIOMIP_HARMONISATION_ID,
    COMBINED_HISTORY_ID,
    DATA_ROOT,
    IAMC_REGION_PROCESSING_ID,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.harmonisation import HARMONISATION_YEAR

# %% [markdown]
# ## Set up

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "REMIND-MAgPIE 3.5-4.10"

# %%
model_clean = model.replace(" ", "_").replace(".", "p")
model_clean

# %%
pandas_openscm.register_pandas_accessor()

# %%
OUT_ID_KEY = "_".join(
    [model_clean, COMBINED_HISTORY_ID, IAMC_REGION_PROCESSING_ID, SCENARIO_TIME_ID, CMIP7_SCENARIOMIP_PRE_PROCESSING_ID, CMIP7_SCENARIOMIP_HARMONISATION_ID]
)

# %%
HARMONISATION_EMISSIONS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonisation-emissions_gcages-conventions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
HARMONISATION_EMISSIONS_FILE

# %%
override_files = tuple((
    DATA_ROOT / "cmip7-scenariomip-workflow" / "harmonisation"
).glob("harmonisation-overrides*.csv"))
override_files

# %%
HARMONISED_FILE_GLOBAL_WORKFLOW = (
    DATA_ROOT / "cmip7-scenariomip-workflow" / "harmonisation" / f"harmonised-for-global-workflow_{OUT_ID_KEY}.csv"
)
HARMONISED_FILE_GLOBAL_WORKFLOW

# %% [markdown]
# ## Load data

# %%
harmonisation_emissions = load_timeseries_csv(
    HARMONISATION_EMISSIONS_FILE,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if harmonisation_emissions.empty:
    raise AssertionError
    
# harmonisation_emissions

# %%
harmonised = load_timeseries_csv(
    HARMONISED_FILE_GLOBAL_WORKFLOW,
    index_columns=["model", "scenario", "region", "variable", "unit"],
    out_column_type=int,
)
if harmonised.empty:
    raise AssertionError
    
# harmonised

# %% [markdown]
# ### Infilling database

# %%
# TODO: freeze infilling database based on something

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

# %%
# TODO: hard-code this in gcages somewhere
harmonise_silicone_gcages = [
    'Emissions|C2F6',
    'Emissions|C6F14',
    'Emissions|CF4',
    'Emissions|SF6',
    'Emissions|BC',
    'Emissions|CH4',
    'Emissions|CO',
    'Emissions|CO2|Biosphere',
    'Emissions|CO2|Fossil',
    'Emissions|HFC125',
    'Emissions|HFC134a',
    'Emissions|HFC143a',
    'Emissions|HFC227ea',
    'Emissions|HFC23',
    'Emissions|HFC245fa',
    'Emissions|HFC32',
    'Emissions|HFC4310mee',
    'Emissions|N2O',
    'Emissions|NH3',
    'Emissions|NMVOC',
    'Emissions|NOx',
    'Emissions|OC',
    'Emissions|SOx'
]

# %%
harmonise_silicone_reporting = [
    convert_variable_name(
        v, 
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        from_convention=SupportedNamingConventions.GCAGES,
    )
    for v in harmonise_silicone_gcages
]

# %%
infilling_db = SCENARIO_DB.load(
    pix.isin(variable=harmonise_silicone_reporting, region="World"), 
    progress=True,
)
if infilling_db.empty:
    raise AssertionError

infilling_db = infilling_db.sort_index(axis="columns")
infilling_db.columns.name = "year"

infilling_db = update_index_levels_func(
    infilling_db,
    {
        "unit": strip_pint_incompatible_characters_from_unit_string,
        "variable": partial(
            convert_variable_name,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        ),
    },
)
# len(infilling_db.pix.unique("variable"))

# %%
overrides = pd.concat([pd.read_csv(f) for f in override_files]).set_index(
    ["model", "scenario", "region", "variable"]
)["method"]
# overrides

# %%
# %pdb

# %%
harmonise_all(
    scenarios=infilling_db,
    history=harmonisation_emissions,
    year=HARMONISATION_YEAR,
    overrides=overrides,
)

# %% [markdown]
# ## Infill

# %%

# %% [markdown]
# ## Save

# %%
infilled.to_csv(OUT_FILE_INFILLED)
