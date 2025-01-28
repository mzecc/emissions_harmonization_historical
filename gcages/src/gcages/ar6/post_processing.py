"""
Post-processing of results
"""

from __future__ import annotations

import multiprocessing

import pandas as pd
import pandas_indexing as pix
from attrs import define


def get_temperatures_in_line_with_assessment(
    raw_temperatures: pd.DataFrame,
    assessment_median: float,
    assessment_time_period: tuple[int, ...],
    assessment_pre_industrial_period: tuple[int, ...],
) -> pd.DataFrame:
    rel_pi_temperatures = raw_temperatures.subtract(
        raw_temperatures.loc[:, list(assessment_pre_industrial_period)].mean(
            axis="columns"
        ),
        axis="rows",
    )
    mod_scen_medians = (
        rel_pi_temperatures.loc[:, list(assessment_time_period)]
        .mean(axis="columns")
        .groupby(["model", "scenario"])
        .median()
    )
    res = (
        rel_pi_temperatures.subtract(mod_scen_medians, axis="rows") + assessment_median
    )
    # Checker:
    # res.loc[:, list(assessment_time_period)].mean(axis="columns").groupby( ["model", "scenario"]).median()  # noqa: E501

    return res


@define
class AR6PostProcessor:
    """
    Post-processor that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
    """

    gsat_assessment_median: float
    """
    Median of the GSAT assessment
    """

    gsat_assessment_time_period: tuple[int, ...]
    """
    Time period over which the GSAT assessment applies
    """

    gsat_assessment_pre_industrial_period: tuple[int, ...]
    """
    Pre-industrial time period used for the GSAT assessment
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

    def __call__(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        Do the post-processing

        Parameters
        ----------
        in_df
            Data to post-process

        Returns
        -------
        :
            Post-processed results
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in the output
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        temperatures_in_line_with_assessment = get_temperatures_in_line_with_assessment(
            in_df.loc[
                pix.isin(
                    variable=["AR6 climate diagnostics|Raw Surface Temperature (GSAT)"]
                )
            ],
            assessment_median=self.gsat_assessment_median,
            assessment_time_period=self.gsat_assessment_time_period,
            assessment_pre_industrial_period=self.gsat_assessment_pre_industrial_period,
        ).pix.assign(variable="AR6 climate diagnostics|Surface Temperature (GSAT)")

        temperatures_in_line_with_assessment_percentiles = (
            temperatures_in_line_with_assessment.groupby(
                ["climate_model", "model", "scenario", "variable", "region", "unit"]
            ).quantile([0.05, 0.1, 0.167, 0.33, 0.5, 0.67, 0.833, 0.9, 0.95])
        )
        temperatures_in_line_with_assessment_percentiles.index.names = [
            *temperatures_in_line_with_assessment_percentiles.index.names[:-1],
            "quantile",
        ]
        temperatures_in_line_with_assessment_percentiles = (
            temperatures_in_line_with_assessment_percentiles.reset_index("quantile")
        )
        temperatures_in_line_with_assessment_percentiles["percentile"] = (
            (100 * temperatures_in_line_with_assessment_percentiles["quantile"])
            .round(1)
            .astype(str)
        )
        temperatures_in_line_with_assessment_percentiles = (
            temperatures_in_line_with_assessment_percentiles.drop(
                "quantile", axis="columns"
            )
            .set_index("percentile", append=True)
            .pix.format(
                variable="{variable}|{climate_model}|{percentile}th Percentile",
                drop=True,
            )
        )

        out_l = [
            # temperatures_in_line_with_assessment,
            temperatures_in_line_with_assessment_percentiles,
        ]

        out = pix.concat(out_l)
        out.columns = out.columns.astype(int)
        # TODO:
        #   - enable optional checks for:
        #       - input and output scenarios are the same

        return out

    @classmethod
    def from_ar6_like_config(
        cls,
        run_checks: bool = True,
        n_processes: int = multiprocessing.cpu_count(),
    ) -> AR6PostProcessor:
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
            Initialised post-processor
        """
        return cls(
            gsat_assessment_median=0.85,
            gsat_assessment_time_period=tuple(range(1995, 2014 + 1)),
            gsat_assessment_pre_industrial_period=tuple(range(1850, 1900 + 1)),
            run_checks=run_checks,
            n_processes=n_processes,
        )
