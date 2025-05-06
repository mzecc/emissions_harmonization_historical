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
import textwrap
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import gcages.cmip7_scenariomip.pre_processing.reaggregation.basic
import openpyxl
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import tqdm.auto as tqdm
from gcages.index_manipulation import combine_species, split_sectors
from gcages.testing import compare_close
from openpyxl.styles import Font
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
model: str = "IMAGE 3.4"
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
level_separator = "|"

# %%
SCENARIO_DB = OpenSCMDB(
    db_dir=SCENARIO_PATH / SCENARIO_TIME_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

SCENARIO_DB.load_metadata().shape


# %% [markdown]
# ## Helper function(s)


# %%
def set_cell(
    value: str,
    row: int,
    col: int,
    ws: openpyxl.worksheet.worksheet.Worksheet,
    font: openpyxl.styles.fonts.Font | None = None,
):
    """Set the value for an excel cell"""
    cell = ws.cell(row=row, column=col)
    cell.value = value
    if font is not None:
        cell.font = font


# %%
def group_into_levels(v_list: Iterable[str], level_separator: str = "|") -> dict[int, list[str]]:
    """Group variables into common levels"""
    n_sector_groupings = defaultdict(list)
    for v in v_list:
        n_sectors = v.count(level_separator) + 1
        n_sector_groupings[n_sectors].append(v)

    return dict(n_sector_groupings)


def get_higher_lower_detail(level: int, ind: dict[int, str]) -> tuple[list[str], list[str]]:
    """Get variables at higher and lower levels of detail than a given level"""
    higher_detail = []
    lower_detail = []
    for k, v in ind.items():
        if k > level:
            higher_detail.extend(v)
        elif k < level:
            lower_detail.extend(v)

    return higher_detail, lower_detail


# %%
def get_extras_l(  # noqa: PLR0913
    inconsistent_variable: str,
    verbosity: int,
    model_df: pd.DataFrame,
    required_model_region_variables: tuple[str, ...],
    optional_model_region_variables: tuple[str, ...],
    required_world_variables: tuple[str, ...],
    not_relevant: tuple[str, ...] = ("Emissions|CO2|AFOLU [NGHGI]",),
) -> list[str]:
    """
    Get list of extra variables

    In other words, variables that are reported
    but aren't used and we can't see a clear reason
    why this wouldn't be an issue.
    """
    not_relevant_s = set(not_relevant)

    required_variables_grouped = group_into_levels(
        v for v in required_model_region_variables if f"{inconsistent_variable}{level_separator}" in v
    )
    optional_variables_grouped = group_into_levels(
        v for v in optional_model_region_variables if f"{inconsistent_variable}{level_separator}" in v
    )

    extras_l = []
    for n_levels in sorted(required_variables_grouped.keys()):
        exp_variables = required_variables_grouped[n_levels]

        if n_levels in optional_variables_grouped:
            opt_variables = optional_variables_grouped[n_levels]
        else:
            opt_variables = []

        higher_detail_required, lower_detail_required = get_higher_lower_detail(n_levels, required_variables_grouped)
        higher_detail_optional, lower_detail_optional = get_higher_lower_detail(n_levels, optional_variables_grouped)
        higher_detail = [*higher_detail_required, *higher_detail_optional]
        lower_detail = [*lower_detail_required, *lower_detail_optional]

        exp_variables_s = set(exp_variables)
        optional_variables_s = set(opt_variables)
        reported_variables = set(
            model_df.loc[
                pix.ismatch(
                    variable=level_separator.join(
                        (f"{inconsistent_variable}{level_separator}*", *(["*"] * (n_levels - 3)))
                    )
                )
            ].pix.unique("variable")
        )

        not_exp = reported_variables - exp_variables_s - optional_variables_s

        reported_at_higher_detail = {v for v in not_exp if any(v in hd for hd in higher_detail)}
        reported_at_lower_detail = {v for v in not_exp if any(ld in v for ld in lower_detail)}

        reported_at_world_level = not_exp.intersection(
            {
                *required_world_variables,
                # No idea how to express the Bunkers exception well
                level_separator.join([*inconsistent_variable.split(level_separator), "Energy", "Demand", "Bunkers"]),
            }
        )
        extras = (
            not_exp - reported_at_higher_detail - reported_at_lower_detail - reported_at_world_level - not_relevant_s
        )

        if verbosity > 0:
            print(f"Level {n_levels - 2}")
            print("=======")

        def wrap_h(vs: set[str]) -> None:
            return textwrap.indent("\n".join(sorted(vs)), prefix="  - ")

        if verbosity > 1:
            print(f"Reported:\n{wrap_h(exp_variables_s.intersection(reported_variables))}")
            print(f"Reported optional:\n{wrap_h(optional_variables_s.intersection(reported_variables))}")

        if verbosity > 0:
            print(f"Missing:\n\n{wrap_h(exp_variables_s - reported_variables)}\n")

        if verbosity > 1:
            print(f"Missing optional:\n{wrap_h(optional_variables_s - reported_variables)}")
            print(f"Reported at higher detail:\n{wrap_h(reported_at_higher_detail)}")
            print(f"Reported at lower detail:\n{wrap_h(reported_at_lower_detail)}")
            print(f"Reported at world level:\n{wrap_h(reported_at_world_level)}")

        if verbosity > 0:
            if extras:
                print(f"Extras (these could be the cause of internal inconsistencies):\n\n{wrap_h(extras)}\n")
                print("")

        extras_l.extend(extras)

    return extras_l


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

model_df.columns.name = "year"
# model_df

# %% [markdown]
# ### Figure out the model-specific regions

# %%
model_regions = [r for r in model_df.pix.unique("region") if r.startswith(model.split(" ")[0])]
if not model_regions:
    raise AssertionError
# model_regions

# %% [markdown]
# ### Define the internal consistency checking index
#
# If your model reports domestic aviation at the regional level, this will be fine.
# If not, we'll need to add some different code.

# %%
internal_consistency_checking_index = (
    gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.get_internal_consistency_checking_index(model_regions)
)

# %%
required_model_region_variables = (
    gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.REQUIRED_MODEL_REGION_VARIABLES
)
optional_model_region_variables = (
    gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.OPTIONAL_MODEL_REGION_VARIABLES
)
required_world_variables = gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.REQUIRED_WORLD_VARIABLES

# %% [markdown]
# ### Check the internal consistency

# %%
model_df_considered = multi_index_lookup(
    model_df,
    internal_consistency_checking_index,
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
tols = gcages.cmip7_scenariomip.pre_processing.reaggregation.basic.get_default_internal_conistency_checking_tolerances()
tols

# %%
inconsistencies_l = []
for variable in totals_exp.pix.unique("variable"):
    totals_exp_variable = totals_exp.loc[pix.isin(variable=variable)]
    totals_reported_variable = totals_reported.loc[pix.isin(variable=variable)]

    unit_l = totals_exp_variable.index.get_level_values("unit").unique()
    if len(unit_l) > 1:
        raise AssertionError(unit_l)

    unit = unit_l[0]
    tols_use = {k: v.to(unit).m if k == "atol" else v for k, v in tols[variable].items()}

    comparison_variable = compare_close(
        left=totals_exp_variable,
        right=totals_reported_variable.reorder_levels(totals_exp_variable.index.names),
        left_name="derived_from_input",
        right_name="reported_total",
        **tols_use,
    )

    if comparison_variable.empty:
        print(f"o {variable} - consistent")
        continue

    print(f"x {variable} - inconsistent")
    inconsistencies_l.append(comparison_variable)
    # display(comparison_variable)

print()
if not inconsistencies_l:
    print("Data is internally consistent")

else:
    print("Internal consistency issues were found")
    inconsistencies = pix.concat(inconsistencies_l)

# %%
verbose_missing_reporting = "v"
verbosity = verbose_missing_reporting.count("v")

# %%
if inconsistencies_l:
    internal_consistency_issues_filepath = output_dir_model / "internal-consistency-issues.xlsx"
    print(f"Writing {internal_consistency_issues_filepath}")

    with pd.ExcelWriter(internal_consistency_issues_filepath, engine="openpyxl") as writer:
        workbook = writer.book

        for inconsistent_variable, inconsistent_variable_df in tqdm.tqdm(
            inconsistencies.groupby("variable"), desc="species"
        ):
            inconsistent_variable_df.columns.name = "source"
            inconsistent_variable_df_stacked = (
                inconsistent_variable_df.unstack().stack("source", future_stack=True).sort_index()
            )

            species = inconsistent_variable.split(level_separator)[-1]
            species_worksheet = workbook.create_sheet(title=species)

            set_cell_ws = partial(set_cell, ws=species_worksheet)

            set_cell_ws(f"Internal consistency errors for {inconsistent_variable}", row=1, col=1, font=Font(bold=True))
            set_cell_ws(
                "Reported totals compared to the sum of sector-region information is shown below",
                row=3,
                col=1,
                font=Font(bold=True),
            )
            set_cell_ws(
                f"Relative tolerance applied while checking for consistency: {tols[inconsistent_variable]['rtol']}",
                row=4,
                col=1,
            )
            set_cell_ws(
                f"Absolute tolerance applied while checking for consistency: {tols[inconsistent_variable]['atol']}",
                row=5,
                col=1,
            )
            set_cell_ws(
                "NaN indicates that there is not an inconsistency issue for this data point for the given tolerances",
                row=6,
                col=1,
            )
            set_cell_ws(
                f"The scenarios were downloaded from the database at {SCENARIO_TIME_ID}",
                row=7,
                col=1,
            )

            current_row = 7
            extras_l = get_extras_l(
                inconsistent_variable=inconsistent_variable,
                verbosity=verbosity,
                model_df=model_df,
                required_model_region_variables=required_model_region_variables,
                optional_model_region_variables=optional_model_region_variables,
                required_world_variables=required_world_variables,
            )
            if extras_l:
                extras_df = model_df.loc[pix.isin(variable=extras_l, region="World")]
                totals_extras = combine_species(
                    split_sectors(extras_df).openscm.groupby_except(["region", "sectors"]).sum()
                ).pix.assign(region="World")

                compare_incl_missing = compare_close(
                    inconsistent_variable_df_stacked.loc[pix.isin(source="reported_total")].reset_index(
                        "source", drop=True
                    ),
                    inconsistent_variable_df_stacked.loc[pix.isin(source="derived_from_input")]
                    .add(totals_extras)
                    .reset_index("source", drop=True),
                    left_name="reported_total",
                    right_name="derived_from_input_incl_extras",
                )

                current_row += 2
                if compare_incl_missing.empty:
                    set_cell_ws(
                        (
                            "The following variables are reported but not picked up in the hierarchy. "
                            "Including them in the totals fixes the mismatch, "
                            "so they likely just need to be reclassified/renamed."
                        ),
                        row=current_row,
                        col=1,
                        font=Font(bold=True),
                    )

                else:
                    set_cell_ws(
                        (
                            "The following variables are reported but not picked up in the hierarchy. "
                            "They may be the source of the inconsistency."
                        ),
                        row=current_row,
                        col=1,
                        font=Font(bold=True),
                    )

                for extra in extras_l:
                    current_row += 1
                    set_cell_ws(
                        f"- {extra}",
                        row=current_row,
                        col=1,
                    )

            current_row += 2
            inconsistent_variable_df_stacked.to_excel(
                writer,
                sheet_name=species,
                startrow=current_row - 1,  # pandas offset by one
            )

            components_included_in_sum = model_df_considered.loc[pix.ismatch(variable=f"{inconsistent_variable}**")]
            components_included_in_sum = (
                components_included_in_sum.pix.assign(source="model_reported")
                .reorder_levels(inconsistent_variable_df_stacked.index.names)
                .sort_index()
            )

            current_row = current_row + inconsistent_variable_df_stacked.shape[0] + 2
            set_cell_ws(
                "The following rows were used to create the sector-region sum",
                row=current_row,
                col=1,
                font=Font(bold=True),
            )
            components_included_in_sum.to_excel(
                writer,
                sheet_name=species,
                startrow=current_row + 2 - 1,  # pandas offset by one
            )

        print(f"Wrote {internal_consistency_issues_filepath}")
