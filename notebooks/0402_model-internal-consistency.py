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
# # Model internal consistency
#
# Here we check the internal consistency of the reporting of a given model.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pandas_openscm
from gcages.cmip7_scenariomip.pre_processing import (
    get_independent_index_input,
)
from gcages.index_manipulation import combine_species, split_sectors
from gcages.testing import compare_close
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.indexing import multi_index_lookup

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
pandas_openscm.register_pandas_accessor()

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
model_df_considered = multi_index_lookup(
    model_df,
    get_independent_index_input(model_regions),
)
# model_df_considered

# %%
totals_exp = combine_species(
    split_sectors(model_df_considered).openscm.groupby_except(["region", "sectors"]).sum()
).pix.assign(region="World")
# totals_exp

# %%
totals_reported = multi_index_lookup(model_df, totals_exp.index)
# totals_reported

# %%
# [print(f"'{v}': dict(rtol=1e-3, atol=1e-6),") for v in totals_exp.pix.unique("variable")]
tols = {
    "Emissions|BC": dict(rtol=1e-3, atol=1e-6),
    "Emissions|CH4": dict(rtol=1e-3, atol=1e-6),
    "Emissions|CO": dict(rtol=1e-3, atol=1e-6),
    # Higher absolute tolerance because of reporting units
    "Emissions|CO2": dict(rtol=1e-3, atol=1.0),
    "Emissions|NH3": dict(rtol=1e-3, atol=1e-6),
    "Emissions|NOx": dict(rtol=1e-3, atol=1e-6),
    "Emissions|OC": dict(rtol=1e-3, atol=1e-6),
    "Emissions|Sulfur": dict(rtol=1e-3, atol=1e-6),
    "Emissions|VOC": dict(rtol=1e-3, atol=1e-6),
    "Emissions|N2O": dict(rtol=1e-3, atol=1e-6),
}

# %%
inconsistencies_l = []
for variable in totals_exp.pix.unique("variable"):
    totals_exp_variable = totals_exp.loc[pix.isin(variable=variable)]
    totals_reported_variable = totals_reported.loc[pix.isin(variable=variable)]

    comparison_variable = compare_close(
        left=totals_exp_variable,
        right=totals_reported_variable.reorder_levels(totals_exp_variable.index.names),
        left_name="derived_from_input",
        right_name="reported_total",
        **tols[variable],
    )

    if comparison_variable.empty:
        continue

    inconsistencies_l.append(comparison_variable)
    display(comparison_variable)  # noqa: F821

if not inconsistencies_l:
    print("Data is internally consistent")

else:
    print("The following internal consistency issues were found")
    inconsistencies = pix.concat(inconsistencies_l)
    display(inconsistencies)  # noqa: F821
    # TODO: save this out by variable
    # inconsistencies.to_csv(output_dir_model / "internal-inconsistencies.csv", index=True)
    # also include the components we considered
    # also see if we can figure out whether there is anything obvious missing
    # model_df_considered.loc[pix.ismatch(variable=f"{variable}**")]
    # variable.split("|")[-1]
