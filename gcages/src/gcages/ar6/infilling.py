"""
Infilling part of the AR6 workflow
"""

from __future__ import annotations

import importlib
import multiprocessing
from collections.abc import Iterable, Mapping
from functools import cache, partial
from pathlib import Path
from typing import Callable

import pandas as pd
import pandas_indexing as pix  # type: ignore
import pyam
import silicone.database_crunchers
from attrs import define

from gcages.io import load_timeseries_csv
from gcages.parallelisation import (
    assert_only_working_on_variable_unit_variations,
    run_parallel,
)


@cache
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

    if cfcs:
        res = pix.assignlevel(
            res,
            # Not sure why this was like this, anyway
            variable=res.index.get_level_values("variable").map(  # type: ignore
                lambda x: x.replace("HFC245fa", "HFC245ca")
            ),
            unit=res.index.get_level_values("unit").map(
                lambda x: x.replace("HFC245fa", "HFC245ca")
            ),
        )

    else:
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
    infillers: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
) -> pd.DataFrame:
    """
    Infill a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    infillers
        Functions to use for infilling each variable.

        The keys define the variable that can be infilled.
        The variables define the function which,
        given inputs with the expected lead variables,
        returns the infilled time series.

    Returns
    -------
    :
        Infilled scenario
    """
    assert_only_working_on_variable_unit_variations(indf)

    indf_variables = indf.pix.unique("variable")

    inferred_l = []
    if "Emissions|CO2" in indf_variables:
        for component_missing, component_other in (
            ("Emissions|CO2|Energy and Industrial Processes", "Emissions|CO2|AFOLU"),
            ("Emissions|CO2|AFOLU", "Emissions|CO2|Energy and Industrial Processes"),
        ):
            if (
                component_missing not in indf_variables
                and component_other in indf_variables
            ):
                inferred = (
                    indf.loc[pix.isin(variable=["Emissions|CO2"])]
                    .subtract(
                        indf.loc[pix.isin(variable=[component_other])].reset_index(
                            "variable", drop=True
                        ),
                        axis="rows",
                    )
                    .pix.assign(variable=component_missing)
                )
                inferred_l.append(inferred)

    if inferred_l:
        inferred = pix.concat(inferred_l)
        inferred_variables = inferred.pix.unique("variable")
    else:
        inferred = None
        inferred_variables = []

    to_infill_silicone = [
        v for v in infillers if v not in indf_variables and v not in inferred_variables
    ]

    # TODO: make progress bar optional
    infilled_silicone_l = []
    for v_to_infill in to_infill_silicone:
        infiller = infillers[v_to_infill]
        tmp = infiller(pyam.IamDataFrame(indf)).timeseries()
        # The fact that this is needed suggests there's a bug in silicone
        tmp = tmp.loc[:, indf.columns]
        infilled_silicone_l.append(tmp)

    # Also add zeros for Emissions|HFC|HFC245ca.
    # The fact that this is effectively hidden in silicone is a little bit
    # the issue with how our stack is set up.
    tmp = (tmp * 0.0).pix.assign(
        variable="Emissions|HFC|HFC245ca", unit="kt HFC245ca/yr"
    )
    infilled_silicone_l.append(tmp)

    infilled_silicone = pix.concat(infilled_silicone_l).sort_index(axis="columns")
    infilled_silicone_vars = infilled_silicone.pix.unique("variable")

    # If we started with total CO2 and infilled fossil CO2,
    # we preserve harmonisation and the total by overwriting AFOLU CO2.
    if (
        "Emissions|CO2" in indf_variables
        and "Emissions|CO2|Energy and Industrial Processes" in infilled_silicone_vars
    ):
        infilled_silicone = pix.concat(
            [
                infilled_silicone.loc[~pix.isin(variable=["Emissions|CO2|AFOLU"])],
                (
                    indf.loc[pix.isin(variable=["Emissions|CO2"])]
                    .subtract(
                        infilled_silicone.loc[
                            pix.isin(
                                variable=[
                                    "Emissions|CO2|Energy and Industrial Processes"
                                ]
                            )
                        ].reset_index("variable", drop=True),
                        axis="rows",
                    )
                    .pix.assign(variable="Emissions|CO2|AFOLU")
                ),
            ]
        )

    if inferred is not None:
        infilled = pix.concat([inferred, infilled_silicone])
    else:
        infilled = infilled_silicone

    return infilled


VARS_DB_CRUNCHERS = {
    "Emissions|BC": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|CH4": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|CO2|AFOLU": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|CO2|Energy and Industrial Processes": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|CO": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|N2O": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|NH3": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|NOx": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|OC": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|Sulfur": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|VOC": (
        False,
        silicone.database_crunchers.QuantileRollingWindows,
    ),
    "Emissions|HFC|HFC134a": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC143a": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC227ea": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC23": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC32": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC43-10": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC125": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|SF6": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|CF4": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|C2F6": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|C6F14": (
        False,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CCl4": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CFC11": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CFC113": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CFC114": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CFC115": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CFC12": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CH2Cl2": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CH3Br": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CH3CCl3": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CH3Cl": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|CHCl3": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HCFC141b": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HCFC142b": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HCFC22": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC152a": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC236fa": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|HFC|HFC365mfc": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|Halon1202": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|Halon1211": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|Halon1301": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|Halon2402": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|NF3": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|C3F8": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|C4F10": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|C5F12": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|C7F16": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|C8F18": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|PFC|cC4F8": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
    "Emissions|SO2F2": (
        True,
        silicone.database_crunchers.RMSClosest,
    ),
}
"""
Definition of our default set of variables to infill and how to infill them

Each key is a variable we can infill.
Each value is a tuple with the following:

- `False` if we should use the 'full' infilling database,
  `True` if we should use the database
  that has information about CFCs and other species not typically modelled by IAMs
- The database cruncher from silicone to use for infilling the variable
"""


@cache
def get_ar6_infiller(
    follower: str,
    lead: tuple[str, ...],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Get an AR6-style infiller

    Parameters
    ----------
    follower
        Variable we are infilling

    lead
        Variable(s) to use as the lead gas for the infilling

    Returns
    -------
    :
        Infiler to use to infill `follower`.
    """
    cfcs, cruncher = VARS_DB_CRUNCHERS[follower]
    db = load_ar6_infilling_db(cfcs=cfcs)

    infiller = cruncher(pyam.IamDataFrame(db)).derive_relationship(
        variable_follower=follower,
        variable_leaders=list(lead),
    )

    return infiller


def do_ar6_like_infilling(
    indf: pyam.IamDataFrame,
    follower: str,
    lead_options: tuple[tuple[str, ...], ...],
) -> pyam.IamDataFrame:
    """
    Infill a variable, AR6-style

    Parameters
    ----------
    indf
        Scenarios to infill

    follower
        Variable we are infilling

    lead_options
        Options to try for variable(s) to use as the lead gas for the infilling.

        The options are tried in the supplied order.

    Returns
    -------
    :
        Infilled timeseries.
    """
    indf_variables = indf.variable
    for leads in lead_options:
        if all(v in indf_variables for v in leads):
            infiller = get_ar6_infiller(follower=follower, lead=leads)

            res = infiller(indf)
            break

    else:
        raise NotImplementedError

    return res


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

        # Strip off any prefixes that might be there
        to_infill = in_emissions.pix.assign(
            variable=in_emissions.index.get_level_values("variable").map(
                lambda x: x.replace("AR6 climate diagnostics|Harmonized|", "")
            )
        ).sort_index(axis="columns")

        # This is not the most efficient parallelisation.
        # It would be better to group by variable,
        # then have an `infill_variable` function,
        # but this would require a bit more thinking
        # to actually achieve.
        infilled_df = pix.concat(
            run_parallel(
                func_to_call=infill_scenario,
                iterable_input=(
                    gdf for _, gdf in to_infill.groupby(["model", "scenario"])
                ),
                input_desc="model-scenario combinations to infill",
                n_processes=self.n_processes,
                infillers=self.infillers,
            )
        )

        # Join input and output emissions (AR6 quirk)
        out: pd.DataFrame = pix.concat(
            [to_infill, infilled_df], axis="index"
        ).sort_index(axis="columns")

        # Apply AR6 naming scheme
        out = out.pix.format(variable="AR6 climate diagnostics|Infilled|{variable}")
        # Apply AR6 filtering
        out = out.loc[~pix.ismatch(variable="**CO2")]
        # Stuff like this is what prevents us from making actually plug and play tools
        out = out.pix.assign(
            unit=out.index.get_level_values("unit").str.replace("HFC4310", "HFC43-10")
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
    def from_ar6_like_config(
        cls,
        run_checks: bool = True,
        n_processes: int = multiprocessing.cpu_count(),
        variables_to_infill: Iterable[str] | None = None,
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

        variables_to_infill
            Variables to infill.

            If not supplied, we support all variables in `VARS_DB_CRUNCHERS`.

        Returns
        -------
        :
            Initialised harmoniser
        """
        if variables_to_infill is None:
            variables_to_infill = VARS_DB_CRUNCHERS.keys()

        lead_options = (
            ("Emissions|CO2",),
            ("Emissions|CO2|Energy and Industrial Processes",),
        )

        infillers = {}

        for v_infill in variables_to_infill:
            infillers[v_infill] = partial(
                do_ar6_like_infilling, follower=v_infill, lead_options=lead_options
            )

        # CO2 Energy and Industrial special case
        infillers["Emissions|CO2|Energy and Industrial Processes"] = partial(
            do_ar6_like_infilling,
            follower="Emissions|CO2|Energy and Industrial Processes",
            lead_options=(("Emissions|CO2",),),
        )

        # TODO: turn checks back on
        return cls(
            infillers=infillers,
            run_checks=run_checks,
            n_processes=n_processes,
        )
