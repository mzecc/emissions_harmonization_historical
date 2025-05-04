"""
Run the 500x series of notebooks

We use this to avoid having to run every combination
of IAM and simple climate model by hand
"""

from pathlib import Path

import papermill as pm


def get_notebook_parameters(notebook_name: str, iam: str) -> dict[str, str]:
    """
    Get parameters for a given notebook

    A bit yuck that we have to do it like this,
    but the notebooks don't all use the same names.
    """
    if notebook_name == "5090_download-scenarios.ipynb":
        res = {"model_search": iam}

    elif notebook_name in ["5091_check-reporting.ipynb"]:
        res = {"model": iam}

    elif notebook_name in ["5092_check-interal-consistency.ipynb"]:
        res = {"model": iam}

    else:
        raise NotImplementedError(notebook_name)

    return res


def main():
    """
    Run the 500x series of notebooks
    """
    HERE = Path(__file__).parent
    DEFAULT_NOTEBOOKS_DIR = HERE.parent / "notebooks"
    RUN_NOTEBOOKS_DIR = HERE.parent / "notebooks-papermill"

    notebooks_dir = DEFAULT_NOTEBOOKS_DIR

    iams = ["REMIND"]
    iams = ["COFFEE"]
    # iams = ["WITCH", "MESSAGE"]
    iams = [
        "REMIND",
        "WITCH",
        "COFFEE",
        "MESSAGE",
        "AIM",
        "IMAGE",
    ]

    notebook_prefixes = ["5091", "5092"]
    # notebook_prefixes = ["5090", "5091", "5092"]

    all_notebooks = tuple(sorted(notebooks_dir.glob("*.ipynb")))
    for iam in iams:
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                parameters = get_notebook_parameters(notebook.name, iam=iam)
                output_notebook = RUN_NOTEBOOKS_DIR / f"{notebook.stem}_{iam}.ipynb"
                output_notebook.parent.mkdir(exist_ok=True, parents=True)

                print(f"Executing {notebook.name=} for {iam=} with {parameters=}. Writing to {output_notebook=}")
                # Execute to specific directory
                pm.execute_notebook(notebook, output_notebook, parameters=parameters)


if __name__ == "__main__":
    main()
