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
import tqdm.auto as tqdm


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
    ]:
        res = {"model": iam, "make_region_sector_plots": True, "output_to_pdf": True}
        res = {"model": iam, "make_region_sector_plots": False, "output_to_pdf": False}

    elif notebook_name in [
        "5190_infilling.py",
        "5191_post-process-emissions.py",
    ]:
        res = {"model": iam}

    elif notebook_name in [
        "5195_run-simple-climate-model.py",
        "5196_post-process-simple-climate-model-output.py",
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


def main():  # noqa: PLR0912
    """
    Run the 500x series of notebooks
    """
    HERE = Path(__file__).parent
    DEFAULT_NOTEBOOKS_DIR = HERE.parent / "notebooks"
    RUN_NOTEBOOKS_DIR = HERE.parent / "notebooks-papermill"

    notebooks_dir = DEFAULT_NOTEBOOKS_DIR
    all_notebooks = tuple(sorted(notebooks_dir.glob("*.py")))

    ### Processing of biomass burning (surprise bonus as running this by hand is annoying)
    species = ["CH4"]
    # # All species
    species = [
        ("BC", "BC"),
        ("CH4", "CH4"),
        ("CO", "CO"),
        ("CO2", "CO2"),
        ("N2O", "N2O"),  # new, to have regional, was global in CMIP6
        ("NH3", "NH3"),
        ("NMVOC", "NMVOCbulk"),  # assumed to be equivalent to IAMC-style reported VOC
        ("NOx", "NOx"),
        ("OC", "OC"),
        ("SO2", "SO2"),
    ]

    # Run the notebook
    notebook_prefixes = ["5006"]
    # Skip this step
    notebook_prefixes = []
    for sp, sp_esgf in species[::-1]:
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                run_notebook(
                    notebook=notebook,
                    run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                    parameters={"species": sp, "species_esgf": sp_esgf},
                    idn=sp,
                )

    ### Individual IAM downloading and processing
    # iams = ["REMIND"]
    # iams = ["GCAM"]
    # iams = ["WITCH"]
    # iams = ["AIM"]
    # Combos
    # iams = ["COFFEE", "WITCH"]
    # iams = [
    #     "WITCH",
    #     "REMIND",
    #     "IMAGE",
    #     "AIM",
    #     "MESSAGE",
    #     "COFFEE",
    # ]
    # # Waiting for submission
    # iams = [
    #     "GCAM",
    # ]
    # All
    iams = [
        "WITCH",
        "REMIND",
        "MESSAGE",
        "IMAGE",
        "GCAM",
        "COFFEE",
        "AIM",
    ]
    # iams = ["COFFEE"]
    iams = ["MESSAGE"]

    #### downloading and processing
    # Single notebook
    notebook_prefixes = ["5094"]
    # # Everything except downloads and reporting checking
    # notebook_prefixes = ["5093", "5094"]
    # # # Downloading and reporting checking
    # # notebook_prefixes = ["5090", "5091", "5092"]
    # Everything
    # notebook_prefixes = ["5090", "5091", "5092", "5093", "5094"]
    # # Skip this step
    # notebook_prefixes = []

    for iam in tqdm.tqdm(iams, desc="IAMs pre infilling"):
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                run_notebook_iam(
                    notebook=notebook,
                    run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                    iam=iam,
                )

    ### Infilling database creation
    # This just creates a database based on whatever you have run above.
    # Hence, this can change depend on order of running, which isn't ideal.
    # However, infilling only really matters for some models
    # (and even then only to a limited degree because it is mostly for F-gases)
    # so this shouldn't make such a big impact.
    # Run the notebook
    notebook_prefixes = ["5095"]
    # # Skip this step
    notebook_prefixes = []
    for notebook in all_notebooks:
        if any(notebook.name.startswith(np) for np in notebook_prefixes):
            run_notebook(
                notebook=notebook,
                run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                parameters={},
                idn="only",
            )

    ### Infilling & Post-processing of emissions
    # only infilling
    # notebook_prefixes = ["5190"]
    # only infilling
    # notebook_prefixes = ["5191"]
    # infilling & post-processing emissions
    notebook_prefixes = ["5190", "5191"]
    # Skip this step
    notebook_prefixes = []
    for iam in iams:
        for notebook in all_notebooks:
            if any(notebook.name.startswith(np) for np in notebook_prefixes):
                run_notebook_iam(
                    notebook=notebook,
                    run_notebooks_dir=RUN_NOTEBOOKS_DIR,
                    iam=iam,
                )

    ### Running the SCMs and post-processing climate outputs
    # SCM related notebooks
    notebook_prefixes = ["5195", "5196"]
    # Single notebook: run SCM
    # notebook_prefixes = ["5195"]
    # Single notebook: run post-processing of climate outputs
    # notebook_prefixes = ["5196"]
    # Skip this step
    notebook_prefixes = []
    scms = ["MAGICCv7.6.0a3", "MAGICCv7.5.3"]
    for iam, scm in tqdm.tqdm(itertools.product(iams, scms), desc="IAM SCM runs"):
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
