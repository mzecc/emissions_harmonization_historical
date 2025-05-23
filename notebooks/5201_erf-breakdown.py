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
# # ERF breakdown
#
# An ERF breakdown over time.
# Very custom.
# If you want to edit, probably best to duplicate
# then edit rather than trying to make this do multiple things.

# %% [markdown]
# ## Imports

# %%
import multiprocessing

import pandas as pd
import pandas_indexing as pix
import pandas_openscm

from emissions_harmonization_historical.constants_5000 import (
    SCM_OUTPUT_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
pix.set_openscm_registry_as_default()

# %%
pd.set_option("display.max_rows", 100)

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Categories

# %%
model = "REMIND"
scenario = "SSP1**Very Low**_c"

# %%
raw_scm_output = SCM_OUTPUT_DB.load(
    pix.ismatch(
        variable=[
            "Effective Radiative Forcing**",
        ],
        model=f"**{model}**",
        scenario=f"**{scenario}**",
        climate_model="**MAGICCv7.6**",
    ),
    progress=True,
    max_workers=multiprocessing.cpu_count(),
)
raw_scm_output

# %%
median = raw_scm_output.openscm.groupby_except("run_id").median()
median

# %%
pdf = (
    median.loc[pix.ismatch(variable=[f"**{v}" for v in ["Greenhouse Gases", "Aerosols", "CO2"]]), 2000:]
    .pix.project("variable")
    .T
)
ax = pdf.plot()

for variable, loc in [*pdf.idxmax().items(), *pdf.idxmin().items()]:
    if loc == pdf.index.max() or (loc == pdf.index.min()):
        continue

    ax.axvline(loc, linestyle="--", label=f"{variable} 'peak'")

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
