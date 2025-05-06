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
# # Infilling
#
# Here we infill each model's data.

# %% [markdown]
# ## Imports

# %%

from collections.abc import Callable
from functools import partial

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import pyam
import silicone.database_crunchers
import tqdm.auto
from gcages.completeness import assert_all_groups_are_complete
from gcages.harmonisation.common import align_history_to_data_at_time
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import compare_close
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.constants_5000 import (
    CMIP7_GHG_PROCESSED_DB,
    HARMONISED_SCENARIO_DB,
    HISTORY_HARMONISATION_DB,
    INFILLED_OUT_DIR,
    INFILLED_SCENARIOS_DB,
    INFILLING_DB,
)
from emissions_harmonization_historical.harmonisation import (
    HARMONISATION_YEAR,
)

# %% [markdown]
# ## Set up

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %%
pandas_openscm.register_pandas_accessor()

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "COFFEE"
output_to_pdf: bool = False

# %%
output_dir_model = INFILLED_OUT_DIR / model
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Harmonised data for SCMs
#
# We're not interested in infilling the other workflows.

# %%
harmonised = HARMONISED_SCENARIO_DB.load(pix.isin(workflow="for_scms") & pix.ismatch(model=f"*{model}*")).reset_index(
    "workflow", drop=True
)
# harmonised

# %% [markdown]
# ### Infilling database

# %%
infilling_db = INFILLING_DB.load()
# infilling_db

# %% [markdown]
# ### History

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

# history

# %% [markdown]
# ### CMIP7 GHG inversions

# %%
cmip7_ghg_inversions = CMIP7_GHG_PROCESSED_DB.load()
# cmip7_ghg_inversions

# %% [markdown]
# ## Check that the infilling database and scenario data are harmonised the same


# %%
def assert_harmonised(  # noqa: D103 # TODO: move to module
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
    species_tolerances: dict[str, dict[str, float | Q]] | None = None,
) -> None:
    if species_tolerances is None:
        species_tolerances = {
            "BC": dict(rtol=1e-3, atol=Q(1e-3, "Mt BC/yr")),
            "CH4": dict(rtol=1e-3, atol=Q(1e-2, "Mt CH4/yr")),
            "CO": dict(rtol=1e-3, atol=Q(1e-1, "Mt CO/yr")),
            "CO2": dict(rtol=1e-3, atol=Q(1e0, "Mt CO2/yr")),
            "NH3": dict(rtol=1e-3, atol=Q(1e-2, "Mt NH3/yr")),
            "NOx": dict(rtol=1e-3, atol=Q(1e-2, "Mt NO2/yr")),
            "OC": dict(rtol=1e-3, atol=Q(1e-3, "Mt OC/yr")),
            "Sulfur": dict(rtol=1e-3, atol=Q(1e-2, "Mt SO2/yr")),
            "VOC": dict(rtol=1e-3, atol=Q(1e-2, "Mt VOC/yr")),
            "N2O": dict(rtol=1e-3, atol=Q(1e-1, "kt N2O/yr")),
        }

    scenarios_a, history_a = align_history_to_data_at_time(
        scenarios,
        history=history.loc[pix.isin(variable=scenarios.pix.unique("variable"))].reset_index(
            ["model", "scenario"], drop=True
        ),
        time=HARMONISATION_YEAR,
    )
    for variable, scen_a_vdf in scenarios_a.groupby("variable"):
        history_a_vdf = history_a.loc[pix.isin(variable=variable)]
        species = variable.split("|")[1]
        if species in species_tolerances:
            unit_l = scen_a_vdf.pix.unique("unit").tolist()
            if len(unit_l) != 1:
                raise AssertionError(unit_l)
            unit = unit_l[0]

            rtol = species_tolerances[species]["rtol"]
            atol = species_tolerances[species]["atol"].to(unit).m
        else:
            rtol = 1e-4
            atol = 1e-6

        compare_close(
            scen_a_vdf.unstack("region"),
            history_a_vdf.unstack("region"),
            left_name="scenario",
            right_name="history",
            rtol=rtol,
            atol=atol,
        )


# %%
assert_harmonised(harmonised, history)

# %%
assert_harmonised(infilling_db, history)

# %%
wmo_locator = pix.ismatch(model="WMO**")
infilling_db_wmo = infilling_db.loc[wmo_locator]
infilling_db_silicone = infilling_db.loc[~wmo_locator]


# %% [markdown]
# ## Infill

# %% [markdown]
# ### Helper functions


# %%
def infill(indf: pd.DataFrame, infillers) -> pd.DataFrame | None:  # noqa: D103 # TODO: move to module
    infilled_l = []
    for variable in tqdm.auto.tqdm(infillers):
        for (model, scenario), msdf in indf.groupby(["model", "scenario"]):
            if variable not in msdf.index.get_level_values("variable"):
                tmp = infillers[variable](msdf)
                infilled_l.append(tmp)

    if not infilled_l:
        return None

    return pix.concat(infilled_l)


# %%
def get_complete(indf: pd.DataFrame, infilled: pd.DataFrame | None) -> pd.DataFrame:  # noqa: D103 # TODO: move to module
    if infilled is not None:
        complete = pix.concat([indf, infilled])

    else:
        complete = indf

    return complete


# %% [markdown]
# ### Silicone


# %%
def get_silicone_based_infiller(infiller):  # noqa: D103 # TODO: move to module
    def res(inp: pd.DataFrame) -> pd.DataFrame:
        res_h = infiller(pyam.IamDataFrame(inp)).timeseries()
        # The fact that this is needed suggests there's a bug in silicone
        res_h = res_h.loc[:, inp.dropna(axis="columns", how="all").columns]

        return res_h

    return res


# %%
lead = "Emissions|CO2|Energy and Industrial Processes"
infillers_silicone = {}
for variable in tqdm.auto.tqdm([v for v in infilling_db_silicone.pix.unique("variable") if v != lead]):
    infillers_silicone[variable] = get_silicone_based_infiller(
        silicone.database_crunchers.RMSClosest(pyam.IamDataFrame(infilling_db_silicone)).derive_relationship(
            variable_follower=variable,
            variable_leaders=[lead],
        )
    )

# %%
infilled_silicone = infill(
    harmonised,
    # harmonised.loc[~pix.isin(variable=infill_silicone_gcages[:3]) | pix.isin(variable=lead)],
    infillers_silicone,
)
complete_silicone = get_complete(harmonised, infilled_silicone)


# %%
# TODO: some plots here

# %% [markdown]
# ### WMO 2022


# %%
def get_direct_copy_infiller(variable: str, copy_from: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Get an infiller which just copies the scenario from another scenario"""

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        model = inp.pix.unique("model")
        if len(model) != 1:
            raise AssertionError(model)
        model = model[0]

        scenario = inp.pix.unique("scenario")
        if len(scenario) != 1:
            raise AssertionError(scenario)
        scenario = scenario[0]

        res = copy_from.loc[pix.isin(variable=variable)].pix.assign(model=model, scenario=scenario)

        return res

    return infiller


# %%
infillers_wmo = {}
for wmo_var in infilling_db_wmo.pix.unique("variable"):
    infillers_wmo[wmo_var] = get_direct_copy_infiller(
        variable=wmo_var,
        copy_from=infilling_db_wmo,
    )

# %%
infilled_wmo = infill(complete_silicone, infillers_wmo)
complete_wmo = get_complete(complete_silicone, infilled_wmo)


# %% [markdown]
# ### Scale timeseries


# %%
def get_direct_scaling_infiller(  # noqa: PLR0913
    leader: str,
    follower: str,
    scaling_factor: float,
    l_0: float,
    f_0: float,
    f_unit: str,
    calculation_year: int,
    f_calculation_year: float,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an infiller which just scales one set of emissions to create the next set

    This is basically silicone's constant ratio infiller
    with smarter handling of pre-industrial levels.
    """

    def infiller(inp: pd.DataFrame) -> pd.DataFrame:
        lead_df = inp.loc[pix.isin(variable=[leader])]

        follow_df = (scaling_factor * (lead_df - l_0) + f_0).pix.assign(variable=follower, unit=f_unit)
        if not np.isclose(follow_df[calculation_year], f_calculation_year).all():
            raise AssertionError

        return follow_df

    return infiller


# %%
to_reporting_names = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.GCAGES,
    to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
)

# %%
scaling_leaders_gcages = {
    "Emissions|C3F8": "Emissions|C2F6",
    "Emissions|C4F10": "Emissions|C2F6",
    "Emissions|C5F12": "Emissions|C2F6",
    "Emissions|C7F16": "Emissions|C2F6",
    "Emissions|C8F18": "Emissions|C2F6",
    "Emissions|cC4F8": "Emissions|CF4",
    "Emissions|SO2F2": "Emissions|CF4",
    "Emissions|HFC236fa": "Emissions|HFC245fa",
    "Emissions|HFC152a": "Emissions|HFC4310mee",
    "Emissions|HFC365mfc": "Emissions|HFC134a",
    "Emissions|CH2Cl2": "Emissions|HFC134a",
    "Emissions|CHCl3": "Emissions|C2F6",
    "Emissions|CH3Br": "Emissions|C2F6",
    "Emissions|CH3Cl": "Emissions|CF4",
    "Emissions|NF3": "Emissions|SF6",
}
scaling_leaders = {to_reporting_names(k): to_reporting_names(v) for k, v in scaling_leaders_gcages.items()}

# %%
cmip7_ghg_inversions_reporting_names = update_index_levels_func(cmip7_ghg_inversions, {"variable": to_reporting_names})

# %%
PI_YEAR = 1750
infillers_scaling = {}
for follower, leader in tqdm.auto.tqdm(scaling_leaders.items()):
    # Need:
    # - f_harmonisation_year
    # - l_harmonisation_year,
    # - f_0
    # - l_0

    lead_df = history.loc[pix.isin(variable=[leader])]
    follow_df = history.loc[pix.isin(variable=[follower])]
    cmip7_inverse_df = cmip7_ghg_inversions_reporting_names.loc[pix.isin(variable=[leader])]

    f_unit = follow_df.pix.unique("unit")
    if len(f_unit) != 1:
        raise AssertionError
    f_unit = f_unit[0].replace("-", "")

    l_unit = lead_df.pix.unique("unit")
    if len(l_unit) != 1:
        raise AssertionError
    l_unit = l_unit[0].replace("-", "")

    for harmonisation_yr_use in [HARMONISATION_YEAR, 2021]:
        l_harmonisation_year = float(lead_df[harmonisation_yr_use].values.squeeze())
        f_harmonisation_year = float(follow_df[harmonisation_yr_use].values.squeeze())
        if not (pd.isnull(l_harmonisation_year) or pd.isnull(f_harmonisation_year)):
            break
    else:
        raise AssertionError

    f_0 = float(cmip7_inverse_df[PI_YEAR].values.squeeze())
    l_0 = float(cmip7_inverse_df[PI_YEAR].values.squeeze())

    # if (f_harmonisation_year - f_0) == 0.0:
    #     scaling_factor = 0.0
    # else:
    scaling_factor = (f_harmonisation_year - f_0) / (l_harmonisation_year - l_0)

    if np.isnan(scaling_factor):
        msg = f"{f_harmonisation_year=} {l_harmonisation_year=} {f_0=} {l_0=}"
        raise AssertionError(msg)

    # break
    infillers_scaling[follower] = get_direct_scaling_infiller(
        leader=leader,
        follower=follower,
        scaling_factor=scaling_factor,
        l_0=l_0,
        f_0=f_0,
        f_unit=f_unit,
        calculation_year=harmonisation_yr_use,
        f_calculation_year=f_harmonisation_year,
    )

# %%
infilled_scaling = infill(complete_wmo, infillers_scaling)
complete = get_complete(complete_wmo, infilled_scaling)

# %% [markdown]
# ## Check completeness

# %%
complete_variables = [
    "Emissions|CO2|Biosphere",
    "Emissions|CO2|Fossil",
    "Emissions|BC",
    "Emissions|CH4",
    "Emissions|CO",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NMVOC",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|SOx",
    "Emissions|C2F6",
    "Emissions|C6F14",
    "Emissions|CF4",
    "Emissions|SF6",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC4310mee",
    "Emissions|CCl4",
    "Emissions|CFC11",
    "Emissions|CFC113",
    "Emissions|CFC114",
    "Emissions|CFC115",
    "Emissions|CFC12",
    "Emissions|CH3CCl3",
    "Emissions|HCFC141b",
    "Emissions|HCFC142b",
    "Emissions|HCFC22",
    "Emissions|Halon1202",
    "Emissions|Halon1211",
    "Emissions|Halon1301",
    "Emissions|Halon2402",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|cC4F8",
    "Emissions|SO2F2",
    "Emissions|HFC236fa",
    "Emissions|HFC152a",
    "Emissions|HFC365mfc",
    "Emissions|CH2Cl2",
    "Emissions|CHCl3",
    "Emissions|CH3Br",
    "Emissions|CH3Cl",
    "Emissions|NF3",
]
len(complete_variables)

# %%
complete_index_reporting_names = pd.MultiIndex.from_product(
    [[to_reporting_names(v) for v in complete_variables], ["World"]],
    names=["variable", "region"],
)

# %%
assert_all_groups_are_complete(complete, complete_index_reporting_names)

# %% [markdown]
# ## Save

# %%
for ids, df in (
    ("silicone", infilled_silicone),
    ("wmo", infilled_wmo),
    ("scaled", infilled_scaling),
    ("complete", complete),
):
    if df is not None:
        INFILLED_SCENARIOS_DB.save(df.pix.assign(stage=ids), allow_overwrite=True, progress=True)
