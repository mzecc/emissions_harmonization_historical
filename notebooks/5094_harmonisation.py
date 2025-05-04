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
# # Harmonisation
#
# Here we harmonise each model's data.

# %% [markdown]
# ## Imports

# %%
import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
import numpy as np
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor

from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    PRE_PROCESSED_SCENARIO_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "COFFEE"

# %% [markdown]
# ## Harmonise for gridding emissions

# %% [markdown]
# ### Load data

# %% [markdown]
# #### Scenarios

# %%
model_pre_processed_for_gridding = PRE_PROCESSED_SCENARIO_DB.load(
    pix.ismatch(model=f"*{model}*", stage="gridding_emissions"), progress=True
)
if model_pre_processed_for_gridding.empty:
    raise AssertionError

# model_pre_processed_for_gridding

# %% [markdown]
# #### History to use for harmonisation

# %%
history_for_gridding_harmonisation = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="gridding_emissions"))
history_for_gridding_harmonisation

# %% [markdown]
# ### Harmonise

# %%
# Harmonise needs to:
# - be separate so we can use it for the infilling db too
# - harmonise
# - return harmonised stuff
# - return the overrides we used, so we can pass them to teams and get tweaks

# %% [markdown]
# 1. harmonise gridded stuff
# 2. aggregate
# 3. harmonise stuff not covered by gridding
# 4. save

# %% [markdown]
# ## Harmonise

# %% [markdown]
# ### Figure out the model-specific regions

# %%
model_regions = [r for r in model_pre_processed.pix.unique("region") if r.startswith(model.split(" ")[0])]
if not model_regions:
    raise AssertionError
# model_regions

# %%
assert False

# %% [markdown]
# ### Do the pre-processing
#
# The hard part here could be deciding on your re-aggregator.
# For now, we assume all models use the same basic reporting
# hence we can use the same re-aggregation strategy for all.

# %%
reaggregator = gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.ReaggregatorBasic(
    model_regions=model_regions
)
pre_processor = CMIP7ScenarioMIPPreProcessor(
    reaggregator=reaggregator,
    n_processes=None,  # run serially
)

# %% [markdown]
# #### Temporary hack: turn off checks
#
# So even if the model's reporting is inconsistent,
# it should still pass pre-processing.

# %%
pre_processor.run_checks = False

# %%
pre_processing_res = pre_processor(model_df)

# %%
pdf = pre_processing_res.global_workflow_emissions.loc[
    pix.ismatch(variable=["**CO2**", "**CH4", "**N2O", "**SOx"])
].openscm.to_long_data()
fg = sns.relplot(
    data=pdf,
    x="time",
    y="value",
    col="variable",
    col_order=sorted(pdf["variable"].unique()),
    col_wrap=2,
    hue="scenario",
    hue_order=sorted(pdf["scenario"].unique()),
    kind="line",
    linewidth=2.0,
    alpha=0.7,
    facet_kws=dict(sharey=False),
)
for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="gray")
    else:
        ax.set_ylim(ymin=0.0)

    ax.set_xticks(np.arange(2020, 2101, 10))
    ax.grid()

# %%
sorted(pre_processing_res.assumed_zero_emissions.pix.unique("variable"))

# %%
for stage, df in (
    ("gridding_emissions", pre_processing_res.gridding_workflow_emissions),
    ("global_workflow_emissions", pre_processing_res.global_workflow_emissions),
    ("global_workflow_emissions_raw_names", pre_processing_res.global_workflow_emissions_raw_names),
    ("assumed_zero", pre_processing_res.assumed_zero_emissions),
):
    PRE_PROCESSED_SCENARIO_DB.save(df.pix.assign(stage=stage), allow_overwrite=True)
    print(f"Saved {stage}")
