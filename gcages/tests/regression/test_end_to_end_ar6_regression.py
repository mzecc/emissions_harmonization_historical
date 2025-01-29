"""
Regression tests of raw emissions to post-processed output compared to AR6
"""

from __future__ import annotations

import multiprocessing
import platform
from pathlib import Path

import pandas as pd
import pandas_indexing as pix
import pytest

from gcages.ar6 import (
    AR6Harmoniser,
    AR6Infiller,
    AR6PostProcessor,
    AR6PreProcessor,
    AR6SCMRunner,
)
from gcages.database import GCDB
from gcages.testing import (
    AR6_IPS,
    assert_frame_equal,
    create_model_scenario_test_cases,
    get_all_model_scenarios,
    get_ar6_harmonised_emissions,
    get_ar6_metadata_outputs,
    get_ar6_raw_emissions,
    get_ar6_temperature_outputs,
)

TEST_DATA_DIR = Path(__file__).parents[1] / "test-data"
RUN_OUTPUT_DB_DIR = TEST_DATA_DIR / "end-to-end-running-ar6-regression"
MODEL_SCENARIO_COMBOS_FILE = (
    TEST_DATA_DIR / "ar6_scenarios_raw_model_scenario_combinations.csv"
)


scenario_cases = pytest.mark.parametrize(
    "model, scenario",
    create_model_scenario_test_cases(
        get_all_model_scenarios(MODEL_SCENARIO_COMBOS_FILE)
    ),
)


@pytest.mark.superslow
@scenario_cases
def test_end_to_end_single_model_scenario(model, scenario):
    raw = get_ar6_raw_emissions(
        model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
    )
    if raw.empty:
        msg = f"No test data for {model=} {scenario=}?"
        raise AssertionError(msg)

    pre_processor = AR6PreProcessor.from_ar6_like_config(
        run_checks=False, n_processes=1
    )
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False, n_processes=1)
    infiller = AR6Infiller.from_ar6_like_config(
        run_checks=False,
        n_processes=1,
    )

    if platform.system() == "Darwin":
        if platform.processor() == "arm":
            magicc_exe = TEST_DATA_DIR / "magicc-v7.5.3/bin/magicc-darwin-arm64"

        else:
            raise NotImplementedError(platform.processor())

    else:
        raise NotImplementedError(platform.system())

    scm_runner = AR6SCMRunner.from_ar6_like_config(
        run_checks=False,
        n_processes=multiprocessing.cpu_count(),
        magicc_exe_path=magicc_exe,
        magicc_prob_distribution_path=TEST_DATA_DIR
        / "magicc-v7.5.3/configs/600-member.json",
        db=GCDB(RUN_OUTPUT_DB_DIR),
    )
    post_processor = AR6PostProcessor.from_ar6_like_config(
        run_checks=False, n_processes=1
    )

    pre_processed = pre_processor(raw)
    harmonised = harmoniser(pre_processed)
    infilled = infiller(harmonised)
    scm_results = scm_runner(infilled, batch_size_scenarios=5)
    post_processed = post_processor(scm_results)

    res_temperature_percentiles = post_processed.loc[
        pix.ismatch(
            variable="AR6 climate diagnostics|Surface Temperature (GSAT)|*|*Percentile"
        )
    ]
    exp_temperature_percentiles = pd.concat(
        get_ar6_temperature_outputs(
            model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
        ).dropna(axis="columns", how="all")
        for model, scenario in infilled.pix.unique(["model", "scenario"])
    )

    assert_frame_equal(
        res_temperature_percentiles.loc[
            :, exp_temperature_percentiles.columns
        ].reset_index(["category", "category_name"], drop=True),
        exp_temperature_percentiles,
        rtol=1e-5,
    )

    all_metadata = get_ar6_metadata_outputs(test_data_dir=TEST_DATA_DIR)
    exp_metadata = all_metadata.loc[
        all_metadata.index.isin(infilled.pix.unique(["model", "scenario"]))
    ]
    if exp_metadata.empty:
        raise AssertionError

    res_metadata = post_processed.index.to_frame(index=False).set_index(
        ["model", "scenario"]
    )
    metadata_compare_cols = ["category", "category_name"]
    exp_metadata_compare = exp_metadata[
        ~exp_metadata["category"].isin(["failed-vetting", "no-climate-assessment"])
    ][metadata_compare_cols]
    if not exp_metadata_compare.empty:
        res_metadata_compare = res_metadata[metadata_compare_cols]
        res_metadata_compare = res_metadata_compare[
            ~res_metadata_compare.index.duplicated()
        ]

        assert_frame_equal(
            res_metadata_compare,
            exp_metadata_compare,
        )


@pytest.mark.superslow
def test_end_to_end_ips_simultaneously():
    raise NotImplementedError
    raw = pd.concat(
        [
            get_ar6_raw_emissions(
                model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
            )
            for model, scenario in AR6_IPS
        ]
    )

    pre_processor = AR6PreProcessor.from_ar6_like_config(run_checks=False)
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False)

    pre_processed = pre_processor(raw)
    res = harmoniser(pre_processed)

    exp = (
        pd.concat(
            [
                get_ar6_harmonised_emissions(
                    model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
                )
                for model, scenario in AR6_IPS
            ]
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)


@pytest.mark.superslow
def test_end_to_end_all_simultaneously():
    raise NotImplementedError
    model_scenarios = get_all_model_scenarios(MODEL_SCENARIO_COMBOS_FILE).values

    raw = pd.concat(
        [
            get_ar6_raw_emissions(
                model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
            )
            for model, scenario in model_scenarios
        ]
    )

    pre_processor = AR6PreProcessor.from_ar6_like_config(run_checks=False)
    harmoniser = AR6Harmoniser.from_ar6_like_config(run_checks=False)

    pre_processed = pre_processor(raw)
    res = harmoniser(pre_processed)

    exp = (
        pd.concat(
            [
                get_ar6_harmonised_emissions(
                    model=model, scenario=scenario, test_data_dir=TEST_DATA_DIR
                )
                for model, scenario in model_scenarios
            ]
        )
        .loc[~pix.ismatch(variable="**Kyoto**")]  # Not used downstream
        .loc[~pix.ismatch(variable="**F-Gases")]  # Not used downstream
        .loc[~pix.ismatch(variable="**HFC")]  # Not used downstream
        .loc[~pix.ismatch(variable="**PFC")]  # Not used downstream
    )

    assert_frame_equal(res, exp)
