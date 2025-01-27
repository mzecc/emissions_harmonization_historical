"""
Simple climate model (SCM) running part of the AR6 workflow
"""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import openscm_runner.adapters
import openscm_runner.run
import pandas as pd
import pandas_indexing as pix  # noqa: F401
import pymagicc.definitions
import scmdata
import tqdm.autonotebook as tqdman
from attrs import define

from gcages.units_helpers import strip_pint_incompatible_characters_from_units


def transform_iamc_to_openscm_runner_variable(v):
    """
    Transform IAMC variable to OpenSCM-Runner variable

    Parameters
    ----------
    v
        Variable name to transform

    Returns
    -------
    :
        OpenSCM-Runner equivalent of `v`
    """
    res = v

    replacements = (
        ("CFC|", ""),
        ("HFC|", ""),
        ("PFC|", ""),
        ("|Montreal Gases", ""),
        (
            "HFC43-10",
            "HFC4310mee",
        ),
        (
            "AFOLU",
            "MAGICC AFOLU",
        ),
        (
            "Energy and Industrial Processes",
            "MAGICC Fossil and Industrial",
        ),
    )
    for old, new in replacements:
        res = res.replace(old, new)

    return res


def convert_openscm_runner_output_names_to_magicc_output_names(
    openscm_runner_names: Iterable[str],
) -> tuple[str, ...]:
    """
    Get output names for the call to MAGICC

    Parameters
    ----------
    openscm_runner_names
        OpenSCM-Runner output names

    Returns
    -------
    :
        MAGICC output names
    """
    res_l = []
    for openscm_runner_variable in openscm_runner_names:
        if openscm_runner_variable == "Surface Air Temperature Change":
            # A fun bug in pymagicc
            res_l.append("SURFACE_TEMP")
        elif openscm_runner_variable == "Effective Radiative Forcing|HFC4310mee":
            # Another fun bug in pymagicc
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                "Effective Radiative Forcing|HFC4310",
                inverse=True,
            )
            res_l.append(magicc_var)
        else:
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                openscm_runner_variable,
                inverse=True,
            )
            res_l.append(magicc_var)

    return tuple(res_l)


def run_scm(
    scenarios: pd.DataFrame,
    climate_models_cfgs: dict[str, list[dict[str, Any]]],
    output_variables: tuple[str, ...],
    n_processes: int,
    batch_size_scenarios: int | None = None,
    force_interpolate_to_yearly: bool = True,
) -> pd.DataFrame:
    if force_interpolate_to_yearly:
        # Interpolate to ensure no nans.
        for y in range(scenarios.columns.min(), scenarios.columns.max() + 1):
            if y not in scenarios:
                scenarios[y] = np.nan

        scenarios = scenarios.T.interpolate("index").T

    for climate_model, cfg in tqdman.tqdm(
        climate_models_cfgs.items(), desc="Climate models"
    ):
        if climate_model == "MAGICC7":
            # Avoid MAGICC's last year jump
            magicc_extra_years = 3
            last_year = scenarios.columns.max()
            scenarios[last_year + magicc_extra_years] = scenarios[last_year]
            cfg_use = [{**c, "endyear": scenarios.columns.max()} for c in cfg]
            os.environ["MAGICC_WORKER_NUMBER"] = n_processes

        else:
            cfg_use = cfg

        # Batching here
        if batch_size_scenarios is None:
            scenario_batches = [scenarios]
        else:
            mod_scens = scenarios.pix.unique(["model", "scenario"])
            scenario_batches = []
            for i in range(batch_size_scenarios, mod_scens.size):
                breakpoint()

        for scenario_batch in tqdman.tqdm(scenario_batches, desc="Scenario batch"):
            # Add db checking here
            assert False, "Up to DB implementation"
            mod_scen_in_db = db.metadata.pix.unique(["model", "scenario"])
            not_run = scenario_batch[~scenario_batch.index.isin(mod_scen_in_db)]

            batch_res = openscm_runner.run.run(
                scenarios=scmdata.ScmRun(scenario_batch, copy_data=True),
                climate_models_cfgs={climate_model: cfg_use},
                output_variables=output_variables,
            ).timeseries(time_axis="year")
            if climate_model == "MAGICC7":
                # Chop off the extra years
                batch_res = batch_res.iloc[:, :-magicc_extra_years]

            assert False, "Up to DB implementation"
            # db.save(batch_res)

    res = db.load()

    return res


@define
class AR6SCMRunner:
    """
    Simple climate model runner that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_like_config`][(c)]
    """

    climate_models_cfgs: dict[str, list[dict[str, Any]]]
    """
    Climate models to run and the configuration to use with them
    """

    output_variables: tuple[str, ...]
    """
    Variables to include in the output
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
        Run the simple climate model

        Parameters
        ----------
        in_emissions
            Emissions to run

        Returns
        -------
        :
            Raw results from the simple climate model
        """
        if self.run_checks:
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in in_emissions
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        # Strip off any prefixes that might be there
        to_run = in_emissions.pix.assign(
            variable=in_emissions.index.get_level_values("variable").map(
                lambda x: x.replace("AR6 climate diagnostics|Infilled|", "")
            )
        ).sort_index(axis="columns")

        # Transform variable names
        to_run = to_run.pix.assign(
            variable=to_run.index.get_level_values("variable")
            .map(transform_iamc_to_openscm_runner_variable)
            .values
        )

        # Deal with the HFC245 bug
        to_run = to_run.pix.assign(
            variable=to_run.index.get_level_values("variable")
            .str.replace("HFC245ca", "HFC245fa")
            .values,
            unit=to_run.index.get_level_values("unit")
            .str.replace("HFC245ca", "HFC245fa")
            .values,
        )

        to_run = strip_pint_incompatible_characters_from_units(to_run)

        scm_results = run_scm(
            to_run,
            climate_models_cfgs=self.climate_models_cfgs,
            output_variables=self.output_variables,
            n_processes=self.n_processes,
        )

        # Apply AR6 naming scheme
        out: pd.DataFrame = scm_results.pix.format(
            variable="AR6 climate diagnostics|{variable}"
        )

        # TODO:
        #   - enable optional checks for:
        #       - input and output scenarios are the same

        return out

    @classmethod
    def from_ar6_like_config(
        cls,
        magicc_exe_path: Path,
        magicc_prob_distribution_path: Path,
        run_checks: bool = True,
        n_processes: int = multiprocessing.cpu_count(),
    ) -> AR6SCMRunner:
        """
        Initialise from config (exactly) like what was used in AR6

        Parameters
        ----------
        magicc_exe_path
            Path to the MAGICC executable to use.

            This should be a v7.5.3 executable.

        magicc_prob_distribution_path
            Path to the MAGICC probabilistic distribution.

            This should be the AR6 probabilistic distribution.

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
            Initialised SCM runner
        """
        os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
        if openscm_runner.adapters.MAGICC7.get_version() != "v7.5.3":
            raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())

        # Check MAGICC prob. distribution
        with open(magicc_prob_distribution_path) as fh:
            cfgs_raw = json.load(fh)

        expected_description = (
            "IPCC AR6 config drawn on 9th Feb 2021. "
            "Sub-sample set 0fd0f62 from derived metrics set f023edb. "
            "Drawn in 532_plot_and_save_subsampled_distribution.ipynb."
        )
        if cfgs_raw["description"] != expected_description:
            msg = "This probabilistic config doesn't match what we expect"
            raise AssertionError(msg)

        base_cfgs = [
            {
                "run_id": c["paraset_id"],
                **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
            }
            for c in cfgs_raw["configurations"]
        ]

        startyear = 1750
        output_variables = (
            # GSAT
            "Surface Air Temperature Change",
            # GMST
            "Surface Air Ocean Blended Temperature Change",
            # ERFs
            "Effective Radiative Forcing",
            "Effective Radiative Forcing|Anthropogenic",
            "Effective Radiative Forcing|Aerosols",
            "Effective Radiative Forcing|Aerosols|Direct Effect",
            "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
            "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
            "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
            "Effective Radiative Forcing|Aerosols|Indirect Effect",
            "Effective Radiative Forcing|Greenhouse Gases",
            "Effective Radiative Forcing|CO2",
            "Effective Radiative Forcing|CH4",
            "Effective Radiative Forcing|N2O",
            "Effective Radiative Forcing|F-Gases",
            "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
            "Effective Radiative Forcing|CFC11",
            "Effective Radiative Forcing|CFC12",
            "Effective Radiative Forcing|HCFC22",
            "Effective Radiative Forcing|Ozone",
            "Effective Radiative Forcing|HFC125",
            "Effective Radiative Forcing|HFC134a",
            "Effective Radiative Forcing|HFC143a",
            "Effective Radiative Forcing|HFC227ea",
            "Effective Radiative Forcing|HFC23",
            "Effective Radiative Forcing|HFC245fa",
            "Effective Radiative Forcing|HFC32",
            "Effective Radiative Forcing|HFC4310mee",
            "Effective Radiative Forcing|CF4",
            "Effective Radiative Forcing|C6F14",
            "Effective Radiative Forcing|C2F6",
            "Effective Radiative Forcing|SF6",
            # Heat uptake
            "Heat Uptake",
            # "Heat Uptake|Ocean",
            # Atmospheric concentrations
            "Atmospheric Concentrations|CO2",
            "Atmospheric Concentrations|CH4",
            "Atmospheric Concentrations|N2O",
            # carbon cycle
            "Net Atmosphere to Land Flux|CO2",
            "Net Atmosphere to Ocean Flux|CO2",
            # permafrost
            "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
            "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
        )
        common_cfg = {
            "startyear": startyear,
            "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(  # noqa: E501
                output_variables
            ),
            "out_ascii_binary": "BINARY",
            "out_binary_format": 2,
        }

        run_config = [
            {
                **common_cfg,
                **base_cfg,
            }
            for base_cfg in base_cfgs
        ]
        magicc_full_distribution_n_config = 600
        if len(run_config) != magicc_full_distribution_n_config:
            raise AssertionError(len(run_config))

        openscm_runner_output_variables = (
            convert_openscm_runner_output_names_to_magicc_output_names(
                common_cfg["out_dynamic_vars"]
            )
        )

        return cls(
            climate_models_cfgs={"MAGICC7": run_config},
            output_variables=openscm_runner_output_variables,
            run_checks=run_checks,
            n_processes=n_processes,
        )
