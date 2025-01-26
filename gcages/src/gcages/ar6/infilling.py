"""
Infilling part of the AR6 workflow
"""

from __future__ import annotations

import importlib
import multiprocessing
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Callable

import pandas as pd
import pandas_indexing as pix  # type: ignore
import pyam
import silicone.database_crunchers
import tqdm.autonotebook as tqdman
from attrs import define

from gcages.aneris_helpers import harmonise_all
from gcages.harmonisation.helpers import add_historical_year_based_on_scaling
from gcages.io import load_timeseries_csv
from gcages.parallelisation import (
    assert_only_working_on_variable_unit_variations,
    run_parallel,
)


def load_ar6_infilling_db(cfcs: bool = False) -> pd.DataFrame:
    """
    Load the infilling database that was used in AR6

    Parameters
    ----------
    cfcs
        Load the database for infilling CFCs.

    Returns
    -------
    :
        Infilling database used in AR6
    """
    if cfcs:
        filepath: Path = Path(  # type: ignore
            importlib.resources.files("gcages") / "ar6" / "infilling_db_cfcs_ar6.csv"
        )

    else:
        filepath: Path = Path(  # type: ignore
            importlib.resources.files("gcages")
            / "ar6"
            / "infilling_db_non-cfcs_ar6.csv"
        )

    res = load_timeseries_csv(
        filepath,
        lower_col_names=True,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )

    if not cfcs:
        res = pix.assignlevel(
            res,
            variable=res.index.get_level_values("variable").map(  # type: ignore
                lambda x: x.replace("AR6 climate diagnostics|", "").replace(
                    "Harmonized|", ""
                )
            ),
        )

    return res


def infill_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.DataFrame | None,
    calc_scaling_year: int,
) -> pd.DataFrame:
    """
    Infill a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    history
        History to harmonise to

    year
        Year to use for harmonisation

    overrides
        Overrides to pass to aneris

    calc_scaling_year
        Year to use for calculating scaling if `year` is not in `indf`

    Returns
    -------
    :
        Infilled scenario
    """
    if "Emissions|CO2|Energy and Industrial Processes" not in indf.pix.unique(
        "variable"
    ):
        # TODO: mucking around infilling CO2 fossil from total CO2 (?)
        raise NotImplementedError

    assert_only_working_on_variable_unit_variations(indf)

    # TODO: split this out
    # A bunch of other fix ups that were applied in AR6
    if year not in indf:
        emissions_to_harmonise = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf,
            emissions_history=history,
        )

    elif indf[year].isnull().any():
        null_emms_in_harm_year = indf[year].isnull()

        dont_change = indf[~null_emms_in_harm_year]

        updated = add_historical_year_based_on_scaling(
            year_to_add=year,
            year_calc_scaling=calc_scaling_year,
            emissions=indf[null_emms_in_harm_year].drop(year, axis="columns"),
            emissions_history=history,
        )

        emissions_to_harmonise = pd.concat([dont_change, updated])

    else:
        emissions_to_harmonise = indf

    # In AR6, any emissions with zero in the harmonisation year were dropped
    emissions_to_harmonise = emissions_to_harmonise[
        ~(emissions_to_harmonise[year] == 0.0)
    ]

    ### In AR6, we interpolated before harmonising

    # First, check that there are no nans in the max year.
    # I don't know what happens in that case.
    if emissions_to_harmonise[emissions_to_harmonise.columns.max()].isnull().any():
        raise NotImplementedError

    # Then, interpolate
    out_interp_years = list(range(year, emissions_to_harmonise.columns.max() + 1))
    emissions_to_harmonise = emissions_to_harmonise.reindex(
        columns=out_interp_years
    ).interpolate(method="slinear", axis="columns")

    harmonised = harmonise_all(
        emissions_to_harmonise,
        history=history,
        year=year,
        overrides=overrides,
    )

    return harmonised


@define
class AR6Infiller:
    """
    Infiller that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
    """

    infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
    """
    Functions to use for infilling each variable.

    The keys define the variable that can be infilled.
    The variables define the function which,
    given inputs with the expected lead variables,
    returns the infilled time series.
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    n_processes: int = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to 1 to process in serial.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Infill

        Parameters
        ----------
        in_emissions
            Emissions to infill

        Returns
        -------
        :
            Infilled emissions
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        # TODO: add in the CO2 calculations here (?)
        # i.e. if we have total CO2 and energy, calculate AFOLU as difference
        # i.e. if we have total CO2 and AFOLU, calculate energy as difference

        # Once the relationships are derived,
        # the actual infilling is so cheap
        # that running in parallel is sort of pointless,
        # Nonetheless, we leave the option open.
        infilled_df = pix.concat(
            run_parallel(
                func_to_call=harmonise_scenario,
                iterable_input=(
                    gdf for _, gdf in in_emissions.groupby(["model", "scenario"])
                ),
                input_desc="model-scenario combinations to harmonise",
                n_processes=self.n_processes,
                history=self.historical_emissions,
                year=self.harmonisation_year,
                overrides=self.aneris_overrides,
                calc_scaling_year=self.calc_scaling_year,
            )
        )

        # # Not sure why this is happening, anyway
        # infilled_df.columns = infilled_df.columns.astype(int)

        # Apply AR6 naming scheme
        out: pd.DataFrame = infilled_df.pix.format(
            variable="AR6 climate diagnostics|Infilled|{variable}"
        )

        # TODO:
        #   - enable optional checks for:
        #       - input and output metadata is identical
        #         (except maybe a stage indicator)
        #           - no mangled variable names
        #           - no mangled units
        #           - output timesteps are from harmonisation year onwards,
        #             otherwise identical to input
        #       - output scenarios all have common starting point

        return out

    @classmethod
    @cache
    def from_ar6_like_config(
        cls, run_checks: bool = True, n_processes: int = multiprocessing.cpu_count()
    ) -> AR6Infiller:
        """
        Initialise from config (exactly) like what was used in AR6

        Parameters
        ----------
        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        n_processes
            Number of processes to use for parallel processing.

            Set to 1 to process in serial.

        Returns
        -------
        :
            Initialised harmoniser
        """
        infilling_db_non_cfcs = load_ar6_infilling_db(cfcs=False)
        infilling_db_cfcs = load_ar6_infilling_db(cfcs=True)

        # May have to undo this to handle the infilling with CO2 fossil vs. CO2 total
        lead = ["Emissions|CO2|Energy and Industrial Processes"]
        vars_db_crunchers = {
            # variable_to_infill: (db_to_use, cruncher_to_use)
            "Emissions|BC": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|CH4": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|CO2|AFOLU": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|CO": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|N2O": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|NH3": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|NOx": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|OC": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|Sulfur": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|VOC": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.QuantileRollingWindows,
            ),
            "Emissions|HFC|HFC134a": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC143a": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC227ea": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC23": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC32": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC43-10": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC125": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|SF6": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|CF4": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|C2F6": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|C6F14": (
                infilling_db_non_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CCl4": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC11": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC113": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC114": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC115": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CFC12": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH2Cl2": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH3Br": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH3CCl3": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CH3Cl": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|CHCl3": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HCFC141b": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HCFC142b": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HCFC22": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC152a": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|HFC|HFC236fa": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            # TODO: figure out what went on here
            # 'Emissions|HFC|HFC245fa',
            # "Emissions|HFC|HFC245ca": (
            #     infilling_db_non_cfcs,
            #     silicone.database_crunchers.RMSClosest,
            # ),
            "Emissions|HFC|HFC365mfc": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon1202": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon1211": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon1301": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|Halon2402": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|NF3": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|C3F8": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|C4F10": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|C5F12": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|C7F16": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|C8F18": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|PFC|cC4F8": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
            "Emissions|SO2F2": (
                infilling_db_cfcs,
                silicone.database_crunchers.RMSClosest,
            ),
        }

        infillers = {}
        for v_infill, (db, cruncher) in tqdman.tqdm(
            vars_db_crunchers.items(), desc="Infillers"
        ):
            # Expensive, because you might create infillers for variables
            # you want infill, but this is the cleanest way to set it up
            # without having to add smart caching to silicone
            # or creating the infillers within the infilling routine
            # (which prevents dependency injection so is a bad pattern).
            # Could do our own caching of creating the infillers here,
            # but probably not worth it given we're only going to use this for tests
            # and the class method is cached.
            v_infill_db = db.loc[pix.isin(variable=[v_infill, *lead])]
            infillers[v_infill] = cruncher(
                pyam.IamDataFrame(v_infill_db)
            ).derive_relationship(
                variable_follower=v_infill,
                variable_leaders=lead,
            )

        # TODO: turn checks back on
        return cls(
            infillers=infillers,
            run_checks=run_checks,
            n_processes=n_processes,
        )
