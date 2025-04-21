"""
Run the 040x notebooks for all the available models
"""

import papermill as pm
import tqdm
from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)

from emissions_harmonization_historical.constants import (
    DATA_ROOT,
    SCENARIO_TIME_ID,
)


def main():
    """
    Run the notebooks
    """
    SCENARIO_PATH = DATA_ROOT / "scenarios" / "data_raw"

    SCENARIO_DB = OpenSCMDB(
        db_dir=SCENARIO_PATH / SCENARIO_TIME_ID / "db",
        backend_data=FeatherDataBackend(),
        backend_index=FeatherIndexBackend(),
    )

    output_dir = DATA_ROOT / "reporting-checking"

    available_models = sorted(SCENARIO_DB.load_metadata().get_level_values("model").unique())
    for model in tqdm.tqdm(available_models, desc="models"):
        model_clean = model.replace(".", "-").replace(" ", "-")

        for notebook_to_run in ["0401_model-reporting.ipynb", "0402_model-internal-consistency.ipynb"]:
            output_dir_notebook = output_dir / SCENARIO_TIME_ID / model_clean
            output_dir_notebook.mkdir(exist_ok=True, parents=True)
            output_notebook = output_dir_notebook / notebook_to_run

            print(f"Executing {notebook_to_run} for {model=}")
            print(f"Writing to {output_notebook}")
            pm.execute_notebook(
                notebook_to_run,
                output_notebook,
                parameters=dict(model=model, output_dir=str(output_dir)),
            )


if __name__ == "__main__":
    main()
