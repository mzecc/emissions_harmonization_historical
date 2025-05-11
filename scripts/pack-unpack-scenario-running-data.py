"""
Pack and unpack scenario running data
"""

import tempfile
from pathlib import Path

import pandas_indexing as pix
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

from emissions_harmonization_historical.constants_5000 import (
    DOWNLOAD_SCENARIOS_ID,
    HISTORY_FOR_HARMONISATION_ID,
    HISTORY_HARMONISATION_DB,
    INFILLING_DB,
    INFILLING_DB_DIR,
    RAW_SCENARIO_DB,
)


def main(pack: bool = True) -> None:
    """Unpack or pack data"""
    REPO_ROOT = Path(__file__).parents[1]

    if pack:
        model_to_grab = "GCAM"

        raw_scenario_data = RAW_SCENARIO_DB.load(pix.ismatch(model=f"**{model_to_grab}**"))
        harmonisation_history = HISTORY_HARMONISATION_DB.load()
        infilling_db = INFILLING_DB.load()

        for data, gzip in (
            (raw_scenario_data, REPO_ROOT / f"raw-scenarios_{DOWNLOAD_SCENARIOS_ID}.tar.gz"),
            (harmonisation_history, REPO_ROOT / f"harmonisation-history_{HISTORY_FOR_HARMONISATION_ID}.tar.gz"),
            (infilling_db, REPO_ROOT / f"infilling-db_{INFILLING_DB_DIR.name}.tar.gz"),
        ):
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_db_dir = tmp_dir / "db"
            tmp_db = OpenSCMDB(
                db_dir=tmp_db_dir,
                backend_data=FeatherDataBackend(),
                backend_index=FeatherIndexBackend(),
            )
            tmp_db.save(data)

            print(f"Creating {gzip}")
            tmp_db.to_gzipped_tar_archive(gzip)

    else:
        for gzip, dest in (
            # (REPO_ROOT / f"raw-scenarios_{DOWNLOAD_SCENARIOS_ID}.tar.gz", RAW_SCENARIO_DB.db_dir),
            # (
            #     REPO_ROOT / f"harmonisation-history_{HISTORY_FOR_HARMONISATION_ID}.tar.gz",
            #     HISTORY_HARMONISATION_DB.db_dir,
            # ),
            # (
            #     REPO_ROOT / f"infilling-db_{INFILLING_DB_DIR.name}.tar.gz",
            #     INFILLING_DB.db_dir,
            # ),
            (
                REPO_ROOT / "infilling-db_0005_0003_0003_0002.tar.gz",
                INFILLING_DB.db_dir,
            ),
        ):
            OpenSCMDB.from_gzipped_tar_archive(
                tar_archive=gzip,
                db_dir=dest,
            )
            print(dest)


if __name__ == "__main__":
    pack = True
    pack = False

    main(pack=pack)
