"""
Regression tests of infilling compared to AR6
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.ar6 import AR6Infiller
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


@pytest.mark.superslow
@infilling_cases
def test_infilling_single_model_scenario(model, scenario):
    harmonised = get_ar6_harmonised_emissions(
        model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
    ).dropna(axis="columns", how="all")
    # Drop out some variables that come from post-processing
    harmonised = (
        harmonised.loc[~pix.ismatch(variable="**Kyoto**")]
        .loc[~pix.ismatch(variable="**F-Gases")]
        .loc[~pix.ismatch(variable="**HFC")]
        .loc[~pix.ismatch(variable="**PFC")]
    )
    if harmonised.empty:
        msg = f"No test data for {model=} {scenario=}?"
        raise AssertionError(msg)

    infiller = AR6Infiller.from_ar6_like_config(
        run_checks=False,
        n_processes=1,
    )

    res = infiller(harmonised)

    exp = (
        get_ar6_infilled_emissions(
            model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**CO2")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)


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


@pytest.mark.superslow
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
