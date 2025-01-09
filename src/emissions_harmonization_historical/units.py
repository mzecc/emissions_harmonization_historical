"""
Ad-hoc support for unit handling
"""

from __future__ import annotations

import openscm_units
import pandas as pd
import pandas_indexing as pix

UR = openscm_units.unit_registry
Q = UR.Quantity

# Units we want things to be in
UNIT_WISHES = (
    # ("variable", "unit")
    ("BC", "Mt BC/yr"),
    ("CH4", "Mt CH4/yr"),
    ("N2O", "Mt N2O/yr"),
    ("CO", "Mt CO/yr"),
    ("CO2", "Mt CO2/yr"),
    ("NH3", "Mt NH3/yr"),
    ("NMVOC", "Mt NMVOC/yr"),
    ("NOx", "kt N2O/yr"),
    ("OC", "Mt OC/yr"),
    ("Sulfur", "Mt SO2/yr"),
)


def get_conv_factor(species: str, start_unit: str, target_unit: str) -> float:
    """
    Get conversion factor

    Parameters
    ----------
    species
        Species we are converting

    start_unit
        Starting unit

    target_unit
        Target unit

    Returns
    -------
    :
        Conversion factor from `start_unit` to `target_unit`
    """
    if species == "OC" and (start_unit == "Mt C/yr" and target_unit == "Mt OC/yr"):
        return 1.0

    if species == "BC" and (start_unit == "Mt C/yr" and target_unit == "Mt BC/yr"):
        return 1.0

    if species == "NOx":
        if start_unit == "Mt NO2/yr" and target_unit == "kt N2O/yr":
            with UR.context("NOx_conversions"):
                interim = Q(1, start_unit).to("kt N / yr")

            with UR.context("N2O_conversions"):
                return interim.to(target_unit).m

        if start_unit == "Mt NO / yr" and target_unit == "kt N2O/yr":
            with UR.context("NOx_conversions"):
                interim = Q(1, start_unit).to("kt N / yr")

            with UR.context("N2O_conversions"):
                return interim.to(target_unit).m

    return Q(1, start_unit).to(target_unit).m


def convert_to_desired_units(indf: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to our desired units

    Parameters
    ----------
    indf
        Input `pd.DataFrame` to convert

    Returns
    -------
    :
        `pd.DataFrame` with the data in our desired units
    """
    # I have no idea how I'm meant to navigate this better, anyway
    outdf = indf.copy()
    for species, target_unit in UNIT_WISHES:
        locator = pix.ismatch(variable=f"Emissions|{species}|**")
        current_unit = indf.loc[locator].index.get_level_values("unit").unique()
        if len(current_unit) != 1:
            raise AssertionError(current_unit)

        current_unit = current_unit[0]

        if current_unit == target_unit:
            continue

        conversion_factor = get_conv_factor(species=species, start_unit=current_unit, target_unit=target_unit)
        print(f"{species=} {current_unit=} {target_unit=} {conversion_factor=}")

        tmp = outdf.loc[locator].copy()
        outdf = outdf.loc[~locator]

        tmp *= conversion_factor
        tmp = pix.assignlevel(tmp, unit=target_unit)

        outdf = pd.concat([tmp, outdf])
        # break

    outdf = outdf.sort_index()

    return outdf
