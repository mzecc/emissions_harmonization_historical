"""
Ad-hoc support for unit handling
"""

from __future__ import annotations

import openscm_units
import pandas as pd
import pandas_indexing as pix

UR = openscm_units.unit_registry
Q = UR.Quantity

# Units we want things to be in.
# As this is for the harmonization of IAMs (in CMIP7 / ScenarioMIP),
# the desired units are defined in
# https://github.com/IAMconsortium/common-definitions/.
UNIT_WISHES = (
    # ("variable", "unit")
    ("BC", "Mt BC/yr"),
    ("CH4", "Mt CH4/yr"),
    ("N2O", "kt N2O/yr"),
    ("CO", "Mt CO/yr"),
    ("CO2", "Mt CO2/yr"),
    ("NH3", "Mt NH3/yr"),
    ("NMVOC", "Mt NMVOC/yr"),
    ("NOx", "Mt NO2/yr"),
    ("OC", "Mt OC/yr"),
    ("Sulfur", "Mt SO2/yr"),
)


def assert_units_match_wishes(indf: pd.DataFrame) -> None:
    mismatches = []
    for species, target_unit in UNIT_WISHES:
        locator = pix.ismatch(variable=f"**Emissions|{species}|**")
        current_unit = indf.loc[locator].index.get_level_values("unit").unique()
        if len(current_unit) != 1:
            raise AssertionError(current_unit)

        current_unit = current_unit[0]

        if current_unit == target_unit:
            continue

        mismatches.append((species, target_unit, current_unit))

    assert not mismatches, f"Unit mismatches detected:\n{mismatches}"
