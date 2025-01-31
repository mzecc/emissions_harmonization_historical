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

# %% [markdown]
# # Run workflow - AR6-style
#
# Run the climate assessment workflow exactly as it was run in AR6.

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing
import os
import platform
import random

import pandas as pd
import pandas_indexing as pix
import tqdm.autonotebook as tqdman
from gcages.ar6 import run_ar6_workflow
from gcages.ar6.post_processing import get_temperatures_in_line_with_assessment
from gcages.database import GCDB
from gcages.io import load_timeseries_csv
from loguru import logger

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
)

# %%
# Disable logging to avoid a million messages.
logging.disable(logging.CRITICAL)
logger.disable("gcages")

# %% [markdown]
# ## Set up

# %%
SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"
SCENARIO_PATH

# %%
scm_output_variables = ("Surface Air Temperature Change",)

# %%
batch_size_scenarios = 5
n_processes = multiprocessing.cpu_count()
# n_processes = 1

# %%
# Needed for 7.5.3 on a mac
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"

if platform.system() == "Darwin":
    if platform.processor() == "arm":
        magicc_exe = "magicc-darwin-arm64"
    else:
        raise NotImplementedError(platform.processor())
elif platform.system() == "Windows":
        magicc_exe = "magicc.exe"
else:
    raise NotImplementedError(platform.system())

magicc_exe_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "bin" / magicc_exe
magicc_prob_distribution_path = DATA_ROOT.parents[0] / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json"
scm_results_db = GCDB(
    DATA_ROOT / "climate-assessment-workflow" / "scm-output" / "ar6-workflow-magicc" / SCENARIO_TIME_ID
)

# %%
# If you need to re-write.
# scm_results_db.delete()

# %% [markdown]
# ## Load scenario data

# %%
scenario_files = tuple(SCENARIO_PATH.glob(f"{SCENARIO_TIME_ID}__scenarios-scenariomip__*.csv"))
if not scenario_files:
    msg = f"Check your scenario ID. {list(SCENARIO_PATH.glob('*.csv'))=}"
    raise AssertionError(msg)

scenario_files[:5]

# %%
scenarios_raw = pix.concat(
    [
        load_timeseries_csv(
            f,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_column_type=int,
        )
        for f in tqdman.tqdm(scenario_files)
    ]
).sort_index(axis="columns")

scenarios_raw_global = scenarios_raw.loc[
    pix.ismatch(region="World"),
    # TODO: drop this once we have usable scenario data post 2100
    :2100,
]

scenarios_raw_global

# %% [markdown]
# ### Hacky pre-processing

# %%
# TODO: move this somewhere clearer,
# probably as a pre-processing check.

# %%
pre_pre_processed_l = []
all_nan_or_zero_l = []
for (model, scenario), msdf in scenarios_raw_global.groupby(["model", "scenario"]):
    if "Emissions|CO2|Energy and Industrial Processes" not in msdf.pix.unique("variable"):
        print(
            f"Excluding {model=} {scenario=} because there are no CO2 fossil emissions."
            # f"\nAvailable variables: {sorted(msdf.pix.unique('variable').tolist())}.\n"
        )
        continue

    all_nan_or_zero = (msdf.isnull() | (msdf == 0.0)).all(axis=1)
    if all_nan_or_zero.any():
        all_nan_or_zero_l.append(msdf[all_nan_or_zero])

    msdf_use = msdf[~all_nan_or_zero]

    pre_pre_processed_l.append(msdf_use)

all_nan_or_zero = pix.concat(all_nan_or_zero_l)
pre_pre_processed = pix.concat(pre_pre_processed_l)
pre_pre_processed

# %%
all_nan_or_zero.index.to_frame(index=False).groupby(["model"])["scenario"].value_counts().sort_values()

# %% [markdown]
# ### Down-select scenarios

# %%
# Randomly select some scenarios
# (this is how I generated the hard-coded values in the next cell).
base = pre_pre_processed.pix.unique(["model", "scenario"]).to_frame(index=False)
base["scenario_group"] = base["scenario"].apply(lambda x: x.split("-")[-1].split("_")[0].strip())

selected_scenarios_l = []
selected_models = []
for scenario_group, sdf in base.groupby("scenario_group"):
    options = sdf.index.values.tolist()
    random.shuffle(options)

    n_selected = 0
    for option_loc in options:
        selected_model = sdf.loc[option_loc, :].model
        if selected_model not in selected_models:
            selected_scenarios_l.append(sdf.loc[option_loc, :])
            selected_models.append(selected_model)
            n_selected += 1
            if n_selected >= 2:  # noqa: PLR2004
                break

    else:
        if n_selected >= 1:
            selected_scenarios_l.append(sdf.loc[option_loc, :])
            selected_models.append(selected_model)
        else:
            selected_scenarios_l.append(sdf.loc[option_loc, :])
            selected_models.append(selected_model)

            option_loc = options[-2]
            selected_model = sdf.loc[option_loc, :].model
            selected_scenarios_l.append(sdf.loc[option_loc, :])
            selected_models.append(selected_model)

selected_scenarios = pd.concat(selected_scenarios_l, axis="columns").T
selected_scenarios_idx = selected_scenarios.set_index(["model", "scenario"]).index
selected_scenarios

# %%
selected_scenarios_idx = pd.MultiIndex.from_tuples(
    (
        ("MESSAGEix-GLOBIOM 2.1-M-R12", "SSP5 - High Emissions"),
        ("IMAGE 3.4", "SSP5 - High Emissions"),
        ("AIM 3.0", "SSP2 - Medium-Low Emissions"),
        ("WITCH 6.0", "SSP2 - Low Emissions"),
        ("REMIND-MAgPIE 3.4-4.8", "SSP2 - Low Overshoot_b"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP5 - Low Overshoot"),
        ("COFFEE 1.5", "SSP2 - Medium Emissions"),
        ("GCAM 7.1 scenarioMIP", "SSP2 - Medium Emissions"),
        ("IMAGE 3.4", "SSP2 - Very Low Emissions"),
        ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP1 - Very Low Emissions"),
    ),
    name=["model", "scenario"],
)
scenarios_run = pre_pre_processed[pre_pre_processed.index.isin(selected_scenarios_idx)]

# scenarios_run = pre_pre_processed.loc[pix.ismatch(scenario="*Very Low*")]

# %%
# To run all, just uncomment the below
scenarios_run = pre_pre_processed

# %%
scenarios_run.pix.unique(["model", "scenario"]).to_frame(index=False)

# %%
res = run_ar6_workflow(
    input_emissions=scenarios_run,
    magicc_exe_path=magicc_exe_path,
    magicc_prob_distribution_path=magicc_prob_distribution_path,
    batch_size_scenarios=batch_size_scenarios,
    scm_output_variables=scm_output_variables,
    scm_results_db=scm_results_db,
    n_processes=n_processes,
    run_checks=False,  # TODO: turn this back on
)

# %%
res.post_processed_scenario_metadata

# %%
# TODO: add this to the workflow's output
temperature_match_historical_assessment = get_temperatures_in_line_with_assessment(
    res.scm_results_raw.loc[pix.isin(variable=["AR6 climate diagnostics|Raw Surface Temperature (GSAT)"])],
    assessment_median=0.85,
    assessment_time_period=range(1995, 2014 + 1),
    assessment_pre_industrial_period=range(1850, 1900 + 1),
).pix.assign(variable="AR6 climate diagnostics|Surface Temperature (GSAT)")

temperature_match_historical_assessment

# %%
peak_warming_quantiles = (
    temperature_match_historical_assessment.max(axis="columns")
    .groupby(["model", "scenario"])
    .quantile([0.05, 0.17, 0.33, 0.5, 0.67, 0.83, 0.95])
    .unstack()
    .sort_values(by=0.33)
)
peak_warming_quantiles

# %%
eoc_warming_quantiles = (
    temperature_match_historical_assessment[2100]
    .groupby(["model", "scenario"])
    .quantile([0.05, 0.17, 0.5, 0.83, 0.95])
    .unstack()
    .sort_values(by=0.5)
)
eoc_warming_quantiles

# %%
category_join = res.post_processed_scenario_metadata
peak_warming_quantiles_join = peak_warming_quantiles[[0.33, 0.5, 0.67]].copy().round(3)
peak_warming_quantiles_join.columns = peak_warming_quantiles_join.columns.map(lambda x: f"Peak {x * 100:.1f}th")

eoc_quantiles_join = eoc_warming_quantiles[[0.5]].copy().round(3)
eoc_quantiles_join.columns = eoc_quantiles_join.columns.map(lambda x: f"2100 {x * 100:.1f}th")

pd.concat(
    [
        category_join,
        peak_warming_quantiles_join,
        eoc_quantiles_join,
    ],
    axis="columns",
).sort_values(["category", "Peak 50.0th"])

# %%
category_join.groupby(["model"]).value_counts().sort_index()
