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
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
import tqdm.auto
from gcages.index_manipulation import split_sectors
from matplotlib.backends.backend_pdf import PdfPages

from emissions_harmonization_historical.constants_5000 import (
    HARMONISED_OUT_DIR,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    PRE_PROCESSED_SCENARIO_DB,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
    harmonise,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "COFFEE"
output_to_pdf: bool = True

# %%
output_dir_model = HARMONISED_OUT_DIR / model
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

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

# %%
model_pre_processed = pix.concat([model_pre_processed_for_gridding]).reset_index("stage", drop=True)

# model_pre_processed

# %% [markdown]
# ##### Temporary hack: interpolate model data to annual to allow harmonisation

# %%
if HARMONISATION_YEAR not in model_pre_processed:
    model_pre_processed[HARMONISATION_YEAR] = np.nan
    model_pre_processed = model_pre_processed.sort_index(axis="columns")
    model_pre_processed = model_pre_processed.T.interpolate(method="index").T

model_pre_processed.sort_values(by=HARMONISATION_YEAR)

# %% [markdown]
# #### History to use for harmonisation

# %%
history_for_gridding_harmonisation = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="gridding_emissions"))
# history_for_gridding_harmonisation

# %%
# TODO: expand this to include global workflow emissions too
history_for_harmonisation = pix.concat([history_for_gridding_harmonisation]).reset_index("purpose", drop=True)

# Only keep data relevant to the model we're harmonising
history_for_harmonisation = history_for_harmonisation.openscm.mi_loc(
    model_pre_processed.index.droplevel(["model", "scenario"])
)
if history_for_harmonisation.empty:
    raise AssertionError

# aneris explodes if any history year is Nan,
# even ones we don't use
history_for_harmonisation = history_for_harmonisation.dropna(axis="columns")

if HARMONISATION_YEAR not in history_for_harmonisation:
    raise AssertionError

# history_for_harmonisation

# %% [markdown]
# ### Harmonise

# %%
# Could load in user overrides from elsewhere here
user_overrides = None

# %%
harmonise_res = harmonise(
    scenarios=model_pre_processed,
    history=history_for_harmonisation,
    harmonisation_year=HARMONISATION_YEAR,
    user_overrides=user_overrides,
    do_not_update_negative_after_offset=("**CO2**",),
    offset_methods_to_update=("reduce_offset_2150_cov",),
    offset_methods_replacement="constant_ratio",
)
# harmonise_res

# %% [markdown]
# ## Examine results

# %%
combo = pix.concat(
    [
        model_pre_processed.pix.assign(stage="pre-processed"),
        harmonise_res.timeseries.pix.assign(stage="harmonised"),
        history_for_harmonisation.pix.assign(stage="history"),
    ]
).sort_index(axis="columns")

# %% [markdown]
# ### Single variable

# %%
pdf = (
    combo.loc[
        pix.isin(variable="Emissions|CO2|International Shipping"),
        1990:2100,
    ]
    .openscm.to_long_data()
    .dropna()
)
# pdf

# %%
fg = sns.relplot(
    data=pdf,
    x="time",
    y="value",
    hue="scenario",
    hue_order=sorted(pdf["scenario"].unique()),
    style="stage",
    dashes={
        "history": "",
        "harmonised": "",
        "pre-processed": (3, 3),
    },
    col="variable",
    col_wrap=1,
    facet_kws=dict(sharey=False),
    kind="line",
)
for ax in fg.axes.flatten():
    if "CO2" in ax.get_title():
        ax.axhline(0.0, linestyle="--", color="tab:gray")

    else:
        ax.set_ylim(ymin=0.0)

# %% [markdown]
# ### Global harmonisation

# %%
# pdf_global_total = (
#     combo.loc[
#         (
#             pix.isin(region="World")
#             & (
#                 ~pix.ismatch(variable="Emissions|*|*")
#                 | pix.isin(variable=["Emissions|CO2|Fossil", "Emissions|CO2|Biosphere"])
#             )
#         ),
#         1990:2100,
#     ]
#     .openscm.to_long_data()
#     .dropna()
# )
# pdf_global_total

# %%
# fg = sns.relplot(
#     data=pdf_global_total,
#     x="time",
#     y="value",
#     hue="scenario",
#     hue_order=sorted(pdf_global_total["scenario"].unique()),
#     style="stage",
#     dashes={
#         "history": "",
#         "harmonised": "",
#         "pre-processed": (3, 3),
#     },
#     col="variable",
#     col_order=sorted(pdf_global_total["variable"].unique()),
#     col_wrap=3,
#     facet_kws=dict(sharey=False),
#     kind="line",
# )
# for ax in fg.axes.flatten():
#     if "CO2" in ax.get_title():
#         ax.axhline(0.0, linestyle="--", color="tab:gray")

#     else:
#         ax.set_ylim(ymin=0.0)

# %% [markdown]
# ### Gridding emissions

# %%
pdf_gridding = (
    combo.loc[
        pix.ismatch(region=harmonise_res.timeseries.index.get_level_values("region").unique())
        & pix.ismatch(variable="Emissions|*|*")
    ]
    .sort_index(axis="columns")
    .loc[:, 1950:]
)

# %%
# # If you need to look at negative values only, use this
# variable_regions_to_plot = harmonise_res.timeseries[harmonise_res.timeseries.min(axis="columns") < 0].index.droplevel(
#   harmonise_res.timeseries.index.names.difference(["variable", "region"])
# ).drop_duplicates()
# # variable_regions_to_plot
# pdf_gridding = pdf_gridding.openscm.mi_loc(variable_regions_to_plot)
# pdf_gridding.pix.unique("variable")

# %%
if pdf_gridding.empty:
    raise AssertionError

# pdf_gridding

# %%
pdf_sectors = split_sectors(pdf_gridding)
# pdf_sectors

# %%
regions = ["World", *sorted([r for r in pdf_sectors.index.get_level_values("region").unique() if r != "World"])]
species_l = sorted(pdf_sectors.pix.unique("species"))

if output_to_pdf:
    ctx_manager = PdfPages(output_dir_model / f"gridding-results_{model}.pdf")

    pn = 1
    toc_l = ["Table of contents", "=================", ""]

    for region in regions:
        toc_l.append(f"{region}")
        toc_l.append("-" * len(region))
        for species in species_l:
            pad = 10 - len(species)
            toc_l.append(f"    {species}:{' ' * pad}{pn}")
            pn += 1

        toc_l.append("")

    toc = "\n".join(toc_l)
    # toc

else:
    ctx_manager = nullcontext()

with ctx_manager as output_pdf_file:
    for region in tqdm.auto.tqdm(regions, desc="regions"):
        pdf_r = pdf_sectors.loc[pix.isin(region=region)]
        for species in tqdm.auto.tqdm(species_l, desc="species", leave=False):
            sdf = pdf_r.loc[pix.isin(species=species)]
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
                col_wrap=min(3, len(snsdf["sectors"].unique())),
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

            if output_to_pdf:
                output_pdf_file.savefig(bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        # # Don't plot all for now
        # if region != "World":
        #     break

# %% [markdown]
# ## Save

# %%
if output_to_pdf:
    with open(output_dir_model / f"gridding-results_{model}_table-of-contents.txt", "w") as fh:
        fh.write(toc)

# %%
harmonise_res.overrides.to_csv(output_dir_model / f"harmonisation-methods_{model}.csv")

# %%
HARMONISED_SCENARIO_DB.save(harmonise_res.timeseries, allow_overwrite=True)
