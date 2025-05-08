"""
Run the 500x series of notebooks

We use this to avoid having to run every combination
of IAM and simple climate model by hand
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import jupytext
import papermill as pm


def get_notebook_parameters(notebook_name: str, iam: str, scm: str | None = None) -> dict[str, str]:
    """
    Get parameters for a given notebook

    A bit yuck that we have to do it like this,
    but the notebooks don't all use the same names
    and I can't see a better solution (maybe someone else can).
    """
    if notebook_name == "5090_download-scenarios.py":
        res = {"model_search": iam}

    elif notebook_name in [
        "5091_check-reporting.py",
        "5092_check-interal-consistency.py",
        "5093_pre-processing.py",
    ]:
        res = {"model": iam}

    elif notebook_name in [
        "5094_harmonisation.py",
        "5095_infilling.py",
    ]:
        res = {"model": iam, "output_to_pdf": True}

    elif notebook_name in [
        "5096_run-simple-climate-model.py",
        "5097_post-process.py",
    ]:
        if scm is None:
            raise TypeError(scm)

        res = {"model": iam, "scm": scm}

    else:
        raise NotImplementedError(notebook_name)

    return res


def run_notebook(notebook: Path, run_notebooks_dir: Path, parameters: dict[str, Any], idn: str) -> None:
    """
    Run a notebook
    """
    notebook_jupytext = jupytext.read(notebook)

    # Write the .py file as .ipynb
    in_notebook = run_notebooks_dir / f"{notebook.stem}_{idn}_unexecuted.ipynb"
    in_notebook.parent.mkdir(exist_ok=True, parents=True)
    jupytext.write(notebook_jupytext, in_notebook, fmt="ipynb")

    output_notebook = run_notebooks_dir / f"{notebook.stem}_{idn}.ipynb"
    output_notebook.parent.mkdir(exist_ok=True, parents=True)

    print(f"Executing {notebook.name=} with {parameters=} from {in_notebook=}. " f"Writing to {output_notebook=}")
    # Execute to specific directory
    pm.execute_notebook(in_notebook, output_notebook, parameters=parameters)


def run_notebook_iam(notebook: Path, run_notebooks_dir: Path, iam: str) -> None:
    """
    Run a notebook that only needs IAM information
    """
    parameters = get_notebook_parameters(notebook.name, iam=iam)

    run_notebook(notebook=notebook, run_notebooks_dir=run_notebooks_dir, parameters=parameters, idn=iam)


def run_notebook_with_scm(notebook: Path, run_notebooks_dir: Path, iam: str, scm: str) -> None:
    """
    Run a notebook that needs SCM information
    """
    parameters = get_notebook_parameters(notebook.name, iam=iam, scm=scm)

    run_notebook(notebook=notebook, run_notebooks_dir=run_notebooks_dir, parameters=parameters, idn=f"{iam}_{scm}")


def main():
    """
    Run the 500x series of notebooks
    """
    HERE = Path(__file__).parent
    DEFAULT_NOTEBOOKS_DIR = HERE.parent / "notebooks"
    RUN_NOTEBOOKS_DIR = HERE.parent / "notebooks-papermill"

    notebooks_dir = DEFAULT_NOTEBOOKS_DIR

    # Individual IAMs
    # iams = ["REMIND"]
    iams = ["COFFEE"]
    # iams = ["GCAM"]
    # iams = ["WITCH", "MESSAGE"]
    # iams = ["AIM"]
    # # All
    # iams = [
    #     "WITCH",
    #     "REMIND",
    #     "MESSAGE",
    #     "IMAGE",
    #     "GCAM",
    #     "COFFEE",
    #     "AIM",
    # ]

    # Skip this step
    notebook_prefixes = ["5094"]
    # # Single notebook
    # notebook_prefixes = ["5095"]
    # Everything except downloads and reporting checking
    notebook_prefixes = ["5093", "5094", "5095"]
    notebook_prefixes = ["5090", "5091", "5092"]
    # # Everything
    # notebook_prefixes = ["5090", "5091", "5092", "5093", "5094", "5095"]

    all_notebooks = tuple(sorted(notebooks_dir.glob("*.py")))
    for iam in iams:
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                run_notebook_iam(
                    notebook=notebook,
                    run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                    iam=iam,
                )

    # SCM related notebooks
    notebook_prefixes = ["5096", "5097"]
    notebook_prefixes = []
    # notebook_prefixes = []
    scms = ["MAGICCv7.5.3", "MAGICCv7.6.0a3"]
    for iam, scm in itertools.product(iams, scms):
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                run_notebook_with_scm(
                    notebook=notebook,
                    run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                    iam=iam,
                    scm=scm,
                )


if __name__ == "__main__":
    main()
