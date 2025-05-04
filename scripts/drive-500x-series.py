"""
Run the 500x series of notebooks

We use this to avoid having to run every combination
of IAM and simple climate model by hand
"""

from pathlib import Path

import papermill as pm

HERE = Path(__file__).parent
DEFAULT_NOTEBOOKS_DIR = HERE.parent / "notebooks"

# This will become the default below,
# but for now only REMIND has full reporting so just use that
ALL_SCENARIOMIP_IAMS = [
    "REMIND",
    "WITCH",
    "COFFEE",
    "MESSAGE",
    "AIM",
    "IMAGE",
]


def get_notebook_parameters(notebook_name: str, iam: str) -> dict[str, str]:
    """
    Get parameters for a given notebook

    A bit yuck that we have to do it like this,
    but the notebooks don't all use the same names.
    """
    if notebook_name == "5090_download-scenarios.ipynb":
        res = {"model_search": iam}

    else:
        raise NotImplementedError(notebook_name)

    return res


def main():
    """
    Run the 500x series of notebooks
    """
    notebooks_dir = DEFAULT_NOTEBOOKS_DIR

    iams = ["REMIND"]
    iams = ["COFFEE", "AIM"]
    # iams = [
    #     "REMIND",
    #     "WITCH",
    #     "COFFEE",
    #     "MESSAGE",
    #     "AIM",
    #     "IMAGE",
    # ]

    notebook_prefixes = ["5090"]

    all_notebooks = tuple(sorted(notebooks_dir.glob("*.ipynb")))
    for iam in iams:
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                parameters = get_notebook_parameters(notebook.name, iam=iam)
                print(f"Executing {notebook.name=} for {iam=} with {parameters=}")
                # Execute in place
                pm.execute_notebook(notebook, notebook, parameters=parameters)


if __name__ == "__main__":
    main()
