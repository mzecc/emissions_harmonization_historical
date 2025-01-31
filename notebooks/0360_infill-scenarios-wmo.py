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
# # Infill - WMO
#
# Here we infill the variables we can using WMO's scenarios.

# %%
import pandas as pd
import pandas_indexing as pix
import seaborn as sns

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    HARMONISATION_ID,
    INFILLING_WMO_ID,
    SCENARIO_TIME_ID,
)
from emissions_harmonization_historical.io import load_csv

# %%
wmo_input_file = DATA_ROOT / "global" / "wmo-2022" / "data_raw" / "Emissions_fromWMO2022.xlsx"
wmo_input_file

# %%
harmonised_file = (
    DATA_ROOT
    / "climate-assessment-workflow"
    / "harmonised"
    / f"harmonised-scenarios_{SCENARIO_TIME_ID}_{HARMONISATION_ID}.csv"
)
harmonised_file

# %%
out_file = (
    DATA_ROOT / "climate-assessment-workflow" / "interim" / f"infilled-wmo_{SCENARIO_TIME_ID}_{INFILLING_WMO_ID}.csv"
)
out_file

# %%
harmonised = load_csv(harmonised_file)

# %%
var_map = {
    # WMO name: (gas name, variable name)
    "CFC-11": ("CFC11", "Emissions|Montreal Gases|CFC|CFC11"),
    "CFC-12": ("CFC12", "Emissions|Montreal Gases|CFC|CFC12"),
    "CFC-113": ("CFC113", "Emissions|Montreal Gases|CFC|CFC113"),
    "CFC-114": ("CFC114", "Emissions|Montreal Gases|CFC|CFC114"),
    "CFC-115": ("CFC115", "Emissions|Montreal Gases|CFC|CFC115"),
    "CCl4": ("CCl4", "Emissions|Montreal Gases|CCl4"),
    "CH3CCl3": ("CH3CCl3", "Emissions|Montreal Gases|CH3CCl3"),
    "HCFC-22": ("HCFC22", "Emissions|Montreal Gases|HCFC22"),
    "HCFC-141b": ("HCFC141b", "Emissions|Montreal Gases|HCFC141b"),
    "HCFC-142b": ("HCFC142b", "Emissions|Montreal Gases|HCFC142b"),
    "halon-1211": ("Halon1211", "Emissions|Montreal Gases|Halon1211"),
    "halon-1202": ("Halon1202", "Emissions|Montreal Gases|Halon1202"),
    "halon-1301": ("Halon1301", "Emissions|Montreal Gases|Halon1301"),
    "halon-2402": ("Halon2402", "Emissions|Montreal Gases|Halon2402"),
}

# %%
wmo_raw = pd.read_excel(wmo_input_file).rename({"Unnamed: 0": "year"}, axis="columns")
wmo_raw

# %%
# Hand woven unit conversion
wmo_clean = (wmo_raw.set_index("year") / 1000).melt(ignore_index=False)
wmo_clean["unit"] = wmo_clean["variable"].map({k: v[0] for k, v in var_map.items()})
wmo_clean["unit"] = "kt " + wmo_clean["unit"] + "/yr"
wmo_clean["variable"] = wmo_clean["variable"].map({k: v[1] for k, v in var_map.items()})
wmo_clean["model"] = "WMO 2022"
wmo_clean["scenario"] = "WMO 2022 projections v20250129"
wmo_clean["region"] = "World"
wmo_clean = wmo_clean.set_index(["variable", "unit", "model", "scenario", "region"], append=True)["value"].unstack(
    "year"
)
# HACK: remove spurious last year
wmo_clean[2100] = wmo_clean[2099]
# HACK: remove negative values
wmo_clean = wmo_clean.where(wmo_clean >= 0, 0.0)
# TODO: apply some smoother

wmo_clean

# %%
RCMIP_PATH = DATA_ROOT / "global/rcmip/data_raw/rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_PATH


# %%
def transform_rcmip_to_iamc_variable(v):
    """Transform RCMIP variables to IAMC variables"""
    res = v

    replacements = (
        ("F-Gases|", ""),
        ("PFC|", ""),
        ("HFC4310mee", "HFC43-10"),
        ("MAGICC AFOLU", "AFOLU"),
        ("MAGICC Fossil and Industrial", "Energy and Industrial Processes"),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


# %%
rcmip = pd.read_csv(RCMIP_PATH)
rcmip_clean = rcmip.copy()
rcmip_clean.columns = rcmip_clean.columns.str.lower()
rcmip_clean = rcmip_clean.set_index(["model", "scenario", "region", "variable", "unit", "mip_era", "activity_id"])
rcmip_clean.columns = rcmip_clean.columns.astype(int)
rcmip_clean = rcmip_clean.pix.assign(
    variable=rcmip_clean.index.get_level_values("variable").map(transform_rcmip_to_iamc_variable)
)
ar6_wmo_scenarios = rcmip_clean.loc[
    pix.ismatch(mip_era="CMIP6")
    & pix.ismatch(scenario="ssp245")
    & pix.ismatch(region="World")
    & pix.ismatch(variable=wmo_clean.pix.unique("variable"))
].reset_index(["mip_era", "activity_id"], drop=True)
ar6_wmo_scenarios = ar6_wmo_scenarios.T.interpolate(method="index").T.loc[:, harmonised.columns]
ar6_wmo_scenarios


# %%
def get_sns_df(indf):
    """
    Get data frame to use with seaborn's plotting
    """
    out = indf.copy()
    out.columns.name = "year"
    out = out.stack().to_frame("value").reset_index()

    return out


# %%
sns.relplot(
    data=pd.concat([get_sns_df(wmo_clean), get_sns_df(ar6_wmo_scenarios)]),
    x="year",
    y="value",
    hue="scenario",
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
    kind="line",
)

# %%
wmo_infilled_l = []
for model, scenario in harmonised.pix.unique(["model", "scenario"]):
    # TODO: use same columns as harmonised once we have WMO data beyond 2100
    tmp = wmo_clean.pix.assign(model=model, scenario=scenario).loc[:, harmonised.columns.min() :]
    wmo_infilled_l.append(tmp)

wmo_infilled = pix.concat(wmo_infilled_l)
wmo_infilled

# %%
sns.relplot(
    data=get_sns_df(
        wmo_infilled.loc[
            pix.isin(model=wmo_infilled.pix.unique("model")[0], scenario=wmo_infilled.pix.unique("scenario")[0])
        ]
    ),
    x="year",
    y="value",
    hue="scenario",
    col="variable",
    col_wrap=3,
    facet_kws=dict(sharey=False),
    kind="line",
)

# %%
out_file.parent.mkdir(exist_ok=True, parents=True)
wmo_infilled.to_csv(out_file)
out_file
