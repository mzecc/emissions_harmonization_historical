"""
Ad-hoc support for unit handling
"""

from __future__ import annotations

import pandas as pd
import pandas_indexing as pix

# Units we want things to be in.
# As this is for the harmonization of IAMs (in CMIP7 / ScenarioMIP),
# the desired units are defined in
# https://github.com/IAMconsortium/common-definitions/.
UNIT_WISHES = (
    # ("variable", "unit")
    ("BC", "Mt BC/yr"),
    ("C2F6", "kt C2F6/yr"),
    ("C6F14", "kt C6F14/yr"),
    ("CF4", "kt CF4/yr"),
    ("CH4", "Mt CH4/yr"),
    ("CO", "Mt CO/yr"),
    ("CO2", "Mt CO2/yr"),
    ("HFC|HFC125", "kt HFC125/yr"),
    ("HFC|HFC134a", "kt HFC134a/yr"),
    ("HFC|HFC143a", "kt HFC143a/yr"),
    ("HFC|HFC152a", "kt HFC152a/yr"),
    ("HFC|HFC227ea", "kt HFC227ea/yr"),
    ("HFC|HFC23", "kt HFC23/yr"),
    ("HFC|HFC236fa", "kt HFC236fa/yr"),
    ("HFC|HFC245fa", "kt HFC245fa/yr"),
    ("HFC|HFC32", "kt HFC32/yr"),
    ("HFC|HFC365mfc", "kt HFC365mfc/yr"),
    ("HFC|HFC43-10", "kt HFC43-10/yr"),
    ("N2O", "kt N2O/yr"),
    ("NH3", "Mt NH3/yr"),
    ("NOx", "Mt NO2/yr"),
    ("OC", "Mt OC/yr"),
    ("SF6", "kt SF6/yr"),
    ("Sulfur", "Mt SO2/yr"),
    ("VOC", "Mt VOC/yr"),
)


def assert_units_match_wishes(indf: pd.DataFrame) -> None:
    """
    Assert that the units used in a `pd.DataFrame` match the units we want

    Parameters
    ----------
    indf
        `pd.DataFrame` to check

    Raises
    ------
    AssertionError
        Units for emissions of a species don't match the desired units
    """
    mismatches = []
    for species, target_unit in UNIT_WISHES:
        locator = pix.ismatch(variable=[f"**Emissions|{species}|**", f"**Emissions|{species}"])
        df_locs = indf.loc[locator]
        if df_locs.empty:
            print(f"{species} not in df")
            continue

        current_unit = indf.loc[locator].index.get_level_values("unit").unique()
        if len(current_unit) != 1:
            raise AssertionError(current_unit)

        current_unit = current_unit[0]

        if current_unit == target_unit:
            continue

        mismatches.append((species, target_unit, current_unit))

    assert not mismatches, f"Unit mismatches detected:\n{mismatches}"
