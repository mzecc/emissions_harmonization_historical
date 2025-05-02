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
# # Harmonisation
#
# Here we do the harmonisation.
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
from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.completeness import assert_all_groups_are_complete
from gcages.index_manipulation import split_sectors
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)
from pandas_openscm.io import load_timeseries_csv

from emissions_harmonization_historical.constants import (
    CMIP7_SCENARIOMIP_PRE_PROCESSING_ID,
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

# %% editable=true slideshow={"slide_type": ""}
HISTORICAL_GRIDDING_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"gridding-harmonisation-emissions_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}.csv"
)
HISTORICAL_GRIDDING_FILE

# %%
HARMONISATION_DEFAULT_METHODS_FILE = (
    DATA_ROOT
    / "cmip7-scenariomip-workflow"
    / "harmonisation"
    / f"harmonisation-default-methods_{COMBINED_HISTORY_ID}_{IAMC_REGION_PROCESSING_ID}_{SCENARIO_TIME_ID}_{CMIP7_SCENARIOMIP_PRE_PROCESSING_ID}.csv"
)
# HARMONISATION_DEFAULT_METHODS_FILE

# %%
# TODO: load historical non-gridding data
# TODO: split out io function to just load all default historical data
#       so we can use it here and in the actual harmonisation

# %%
IN_DIR = DATA_ROOT / "cmip7-scenariomip-workflow" / "pre-processing" / CMIP7_SCENARIOMIP_PRE_PROCESSING_ID
in_db = OpenSCMDB(
    db_dir=IN_DIR,
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

in_db.load_metadata().shape

# %%
OUT_ID_KEY = "_".join(
    [model_clean, COMBINED_HISTORY_ID, IAMC_REGION_PROCESSING_ID, SCENARIO_TIME_ID, CMIP7_SCENARIOMIP_PRE_PROCESSING_ID]
)

# %%
OUT_FILE = DATA_ROOT / "cmip7-scenariomip-workflow" / "harmonisation" / f"harmonised_{OUT_ID_KEY}.csv"
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)
OUT_FILE

# %%
OUT_FILE_OVERRIDES = (
    DATA_ROOT / "cmip7-scenariomip-workflow" / "harmonisation" / f"harmonisation-overrides_{OUT_ID_KEY}.csv"
)
OUT_FILE_OVERRIDES.parent.mkdir(exist_ok=True, parents=True)
OUT_FILE_OVERRIDES

# %% [markdown]
# ## Load data

# %%
in_db.load_metadata().get_level_values("stage").unique()

# %%
pre_processed_model = in_db.load(pix.isin(model=model, stage="gridding_emissions"), progress=True).reset_index(
    "stage", drop=True
)
if pre_processed_model.empty:
    raise AssertionError

pre_processed_model

# %%
model_variables_regions = pre_processed_model.pix.unique(["variable", "region"])
# model_variables_regions

# %%
historical_emissions_gridding_model = (
    load_timeseries_csv(
        HISTORICAL_GRIDDING_FILE,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_column_type=int,
    )
).openscm.mi_loc(model_variables_regions)
historical_emissions_gridding_model.columns.name = "year"

if historical_emissions_gridding_model.empty:
    raise AssertionError

assert_all_groups_are_complete(historical_emissions_gridding_model, model_variables_regions)

historical_emissions_gridding_model

# %%
harmonisation_defaults = pd.read_csv(HARMONISATION_DEFAULT_METHODS_FILE).set_index(
    ["model", "scenario", "region", "variable"]
)["method"]

harmonisation_defaults

# %% [markdown]
# ## Apply harmonisation overrides

# %%
# # If you're trying to figure out issues with a particular timeseries,
# # filter first
# pre_processed_model = pre_processed_model.loc[pix.ismatch(variable="**Sulfur**Industr**", region=[
#     "REMIND-MAgPIE 3.5-4.10|Canada, Australia, New Zealand",
#     "REMIND-MAgPIE 3.5-4.10|India",
#     "REMIND-MAgPIE 3.5-4.10|Middle East and North Africa",
# ])]
# pre_processed_model

# %% [markdown]
# The default harmonisation gives unwanted negative values
# where the difference offset between the model and history
# is larger than the historical emissions themselves
# and the default choice is reduce offset.
# Hence we override these here
# for all species except CO<sub>2</sub>.

# %%
overrides = harmonisation_defaults

# %%
offsets = pre_processed_model[HARMONISATION_YEAR].subtract(
    historical_emissions_gridding_model[HARMONISATION_YEAR].reset_index(["model", "scenario"], drop=True)
)
pre_processed_model_plus_offset = pre_processed_model.subtract(offsets, axis="rows")
negative_with_offset = pre_processed_model_plus_offset[pre_processed_model_plus_offset.min(axis="columns") < 0.0]
# Drop out CO2
negative_with_offset = negative_with_offset.loc[~pix.ismatch(variable="**CO2**")]
reduce_offset_despite_negative_with_offset = negative_with_offset[
    negative_with_offset == "reduce_offset_2150_cov"
].index
# Override with ratio instead to avoid unintended negatives
overrides.loc[reduce_offset_despite_negative_with_offset] = "constant_ratio"

# %%
overrides.loc["REMIND-MAgPIE 3.5-4.10", "SSP1 - Low Emissions", slice(None), "Emissions|BC|Transportation Sector"]

# %%
# model specific stuff could go in here
# (or be loaded from a file)
# overrides.loc["REMIND-MAgPIE 3.5-4.10", "SSP1 - Low Emissions", slice(None), "Emissions|BC|Transportation Sector"]

# %% [markdown]
# ## Harmonise

# %%
pre_processed_model

# %%
harmonised = harmonise_all(
    scenarios=pre_processed_model,
    history=historical_emissions_gridding_model,
    year=HARMONISATION_YEAR,
    overrides=overrides,
)
# harmonised

# %%
combo = pix.concat(
    [
        pre_processed_model.pix.assign(stage="pre-processed"),
        harmonised.pix.assign(stage="harmonised"),
        historical_emissions_gridding_model.pix.assign(stage="history"),
    ]
)

# %%
combo_global_workflow = to_global_workflow_emissions(combo)
pdf_global_workflow = combo_global_workflow.openscm.to_long_data().dropna()
# Drop out CO2 biosphere as we don't harmonise this via the gridding sectors
# TODO: fix
pdf_global_workflow = pdf_global_workflow[~pdf_global_workflow["variable"].str.endswith("CO2|Biosphere")]

# %%
fg = sns.relplot(
    data=pdf_global_workflow[pdf_global_workflow["time"] > 1990],
    x="time",
    y="value",
    hue="scenario",
    hue_order=sorted(pdf_global_workflow["scenario"].unique()),
    style="stage",
    dashes={
        "history": "",
        "harmonised": "",
        "pre-processed": (3, 3),
    },
    col="variable",
    col_order=sorted(pdf_global_workflow["variable"].unique()),
    col_wrap=3,
    facet_kws=dict(sharey=False),
    kind="line",
)
for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="tab:gray")

    else:
        ax.set_ylim(ymin=0.0)

# %%
pdf = (
    combo.loc[
        pix.isin(
            variable=harmonised.index.get_level_values("variable").unique(),
            region=harmonised.index.get_level_values("region").unique(),
        )
    ]
    .sort_index(axis="columns")
    .loc[:, 1950:]
)

# %%
# # If you need to look at negative values only, use this
# variable_regions_to_plot = harmonised[harmonised.min(axis="columns") < 0].index.droplevel(harmonised.index.names.difference(["variable", "region"])).drop_duplicates()
# # variable_regions_to_plot
# pdf = multi_index_lookup(pdf, variable_regions_to_plot)

# %%
if pdf.empty:
    raise AssertionError

# pdf

# %%
pdf.pix.unique("stage")

# %%
pdf_sectors = split_sectors(pdf)
pdf_sectors

# %%
for region in ["World", *sorted([r for r in pdf_sectors.index.get_level_values("region").unique() if r != "World"])]:
    pdf_r = pdf_sectors.loc[pix.isin(region=region)]
    for species, sdf in pdf_r.groupby("species"):
        snsdf = sdf.openscm.to_long_data().dropna()
        fg = sns.relplot(
            data=snsdf,
            x="time",
            y="value",
            hue="scenario",
            hue_order=sorted(snsdf["scenario"].unique()),
            style="stage",
            dashes={
                "history": "",
                "harmonised": "",
                "pre-processed": (3, 3),
            },
            col="sectors",
            col_wrap=3,
            kind="line",
            facet_kws=dict(sharey=False),
        )
        fg.fig.suptitle(f"{species} - {region}", y=1.02)
        for ax in fg.axes.flatten():
            ax.axvline(HARMONISATION_YEAR, linestyle="--", color="gray", alpha=0.3, zorder=1.2)

            if species == "CO2":
                ax.axhline(0.0, linestyle="--", color="tab:gray")
            else:
                ax.set_ylim(ymin=0.0)

        plt.show()

    # Don't plot all for now
    if region != "World":
        break

# %%
assert False, "Save harmonised and overrides"

# %%
