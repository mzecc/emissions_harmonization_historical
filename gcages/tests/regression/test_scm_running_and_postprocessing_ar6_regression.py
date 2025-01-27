"""
Regression tests of SCM running compared to AR6
"""

from __future__ import annotations

import platform
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.ar6 import AR6PostProcessor, AR6SCMRunner
from gcages.testing import (
    AR6_IPS,
    assert_frame_equal,
    create_model_scenario_test_cases,
    get_all_model_scenarios,
    get_ar6_harmonised_emissions,
    get_ar6_infilled_emissions,
)

TEST_DATA_DIR = Path(__file__).parents[1] / "test-data"
MODEL_SCENARIO_COMBOS_FILE = (
    TEST_DATA_DIR / "ar6_scenarios_raw_model_scenario_combinations.csv"
)


infilling_cases = pytest.mark.parametrize(
    "model, scenario",
    create_model_scenario_test_cases(
        get_all_model_scenarios(MODEL_SCENARIO_COMBOS_FILE)
    ),
)


@pytest.mark.slow
@infilling_cases
def test_infilling_single_model_scenario(model, scenario):
    infilled = get_ar6_infilled_emissions(
        model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
    )
    # Drop out some variables that come from post-processing/aren't used
    infilled = (
        infilled.loc[~pix.ismatch(variable="**Kyoto**")]
        .loc[~pix.ismatch(variable="**F-Gases")]
        .loc[~pix.ismatch(variable="**HFC")]
        .loc[~pix.ismatch(variable="**PFC")]
        .loc[~pix.ismatch(variable="**CO2")]
    )

    if infilled.empty:
        msg = f"No test data for {model=} {scenario=}?"
        raise AssertionError(msg)

    if platform.system() == "Darwin":
        if platform.processor() == "arm":
            magicc_exe = TEST_DATA_DIR / "magicc-v7.5.3/bin/magicc-darwin-arm64"

        else:
            raise NotImplementedError(platform.processor())

    else:
        raise NotImplementedError(platform.system())

    scm_runner = AR6SCMRunner.from_ar6_like_config(
        run_checks=False,
        # TODO: try using more processes here
        n_processes=1,
        magicc_exe_path=magicc_exe,
        magicc_prob_distribution_path=TEST_DATA_DIR
        / "magicc-v7.5.3/configs/600-member.json",
    )
    post_processor = AR6PostProcessor.from_ar6_like_config(
        run_checks=False, n_processes=1
    )

    scm_results = scm_runner(infilled)
    post_processed = post_processor(scm_results)

    res_temperature_percentiles = post_processed.loc[
        pix.ismatch(
            variable="AR6 climate diagnostics|Surface Temperature (GSAT)|*|*Percentile"
        )
    ]
    exp_temperature_percentiles = get_ar6_temperature_outputs(
        model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
    )

    assert_frame_equal(res_temperature_percentiles, exp_temperature_percentiles)

    exp_metadata = get_ar6_metadata_outputs(
        model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
    )
    res_metadata = res_temperature_percentiles.index.to_frame(index=False).set_index(
        ["model", "scenario"]
    )

    metadata_compare_cols = [
        "Category",
        "Category_name",
    ]
    assert_frame_equal(
        res_metadata[metadata_compare_cols], exp_metadata[metadata_compare_cols]
    )


@pytest.mark.slow
def test_infilling_ips_simultaneously():
    harmonised = pd.concat(
        [
            get_ar6_harmonised_emissions(
                model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
            )
            for model, scenario in AR6_IPS
        ]
    ).dropna(axis="columns", how="all")
    # Drop out some variables that come from post-processing
    harmonised = (
        harmonised.loc[~pix.ismatch(variable="**Kyoto**")]
        .loc[~pix.ismatch(variable="**F-Gases")]
        .loc[~pix.ismatch(variable="**HFC")]
        .loc[~pix.ismatch(variable="**PFC")]
    )

    infiller = AR6Infiller.from_ar6_like_config(
        run_checks=False,
        n_processes=1,
    )

    res = infiller(harmonised)

    exp = (
        pd.concat(
            [
                get_ar6_infilled_emissions(
                    model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
                )
                for model, scenario in AR6_IPS
            ]
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**CO2")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)


@pytest.mark.slow
def test_infilling_all_simultaneously():
    model_scenarios = get_all_model_scenarios(MODEL_SCENARIO_COMBOS_FILE).values

    harmonised = pd.concat(
        [
            get_ar6_harmonised_emissions(
                model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
            )
            for model, scenario in model_scenarios
        ]
    ).dropna(axis="columns", how="all")
    # Drop out some variables that come from post-processing
    harmonised = (
        harmonised.loc[~pix.ismatch(variable="**Kyoto**")]
        .loc[~pix.ismatch(variable="**F-Gases")]
        .loc[~pix.ismatch(variable="**HFC")]
        .loc[~pix.ismatch(variable="**PFC")]
    )

    infiller = AR6Infiller.from_ar6_like_config(
        run_checks=False,
        n_processes=1,
    )

    res = infiller(harmonised)

    exp = (
        pd.concat(
            [
                get_ar6_infilled_emissions(
                    model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
                )
                for model, scenario in model_scenarios
            ]
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**CO2")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)
