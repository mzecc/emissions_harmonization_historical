"""
Constants for use in the 5000 notebooks

These are a way of ensuring that shared ideas are consistent across notebooks
and identifying the processing
that was done to create different data
as we update our processing steps.
If you update the way something is done
and want to avoid overwriting your old data,
just update the ID in here
(because then the data will be written to a different file).
There are much more sophisticated ways this could be done,
but they also require more effort :)
"""

from pathlib import Path

from pandas_openscm.db import (
    FeatherDataBackend,
    FeatherIndexBackend,
    OpenSCMDB,
)

# Chosen to match the CMIP experiment ID
HISTORY_SCENARIO_NAME = "historical"

REPO_ROOT = Path(__file__).parents[2]
DATA_ROOT = REPO_ROOT / "data"

CEDS_VERSION_ID = "v_2025_03_18"

# ID for the CEDS processing step
CEDS_PROCESSING_ID = "0001"

CEDS_TOP_LEVEL_RAW_PATH = DATA_ROOT / "raw" / "ceds"
CEDS_RAW_PATH = CEDS_TOP_LEVEL_RAW_PATH / CEDS_VERSION_ID

# Database into which the processed CEDS data is saved
CEDS_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "ceds" / CEDS_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the GFED4 processing step
GFED4_PROCESSING_ID = "0001"

GFED4_TOP_LEVEL_RAW_PATH = DATA_ROOT / "raw" / "gfed4"
GFED4_RAW_PATH = GFED4_TOP_LEVEL_RAW_PATH

# Database into which the processed GFED4 data is saved
GFED4_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "gfed4" / GFED4_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# Commit from https://github.com/IAMconsortium/common-definitions
# to use
COMMON_DEFINITIONS_COMMIT = "95b5f2c9fb62e32a4d08fe2ffc5b4a6ff246ad2d"
COMMON_DEFINITIONS_PATH = REPO_ROOT / "common-definitions"

# ID for the creation of a historical emissions dataset for gridding
CREATE_HISTORY_FOR_GRIDDING_ID = "0001"

# ID for the created history for harmonisation
HISTORY_FOR_HARMONISATION_ID = "_".join(
    [
        CEDS_PROCESSING_ID,
        GFED4_PROCESSING_ID,
        CREATE_HISTORY_FOR_GRIDDING_ID,
    ]
)

# Directory in which the history for harmonisation information lives
HISTORY_HARMONISATION_DIR = DATA_ROOT / "processed" / "history-for-harmonisation" / HISTORY_FOR_HARMONISATION_ID

REGION_MAPPING_FILE = HISTORY_HARMONISATION_DIR / f"region-mapping_{COMMON_DEFINITIONS_COMMIT}.csv"

# Database to hold historical emissions for harmonisation
HISTORY_HARMONISATION_DB = OpenSCMDB(
    db_dir=HISTORY_HARMONISATION_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)


# ID for the scenario download step
DOWNLOAD_SCENARIOS_ID = "0001"

# Database into which raw scenarios are saved
RAW_SCENARIO_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the pre-processing step
PRE_PROCESSING_ID = "0001"

# Database into which pre-processed scenarios are saved
PRE_PROCESSED_SCENARIO_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "pre-processed" / f"{DOWNLOAD_SCENARIOS_ID}_{PRE_PROCESSING_ID}" / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

# ID for the harmonisation step
HARMONISATION_ID = "0001"

HARMONISED_OUT_DIR = (
    DATA_ROOT
    / "processed"
    / "harmonised"
    / f"{DOWNLOAD_SCENARIOS_ID}_{PRE_PROCESSING_ID}_{HISTORY_FOR_HARMONISATION_ID}_{HARMONISATION_ID}"
)

# Database into which harmonised scenarios are saved
HARMONISED_SCENARIO_DB = OpenSCMDB(
    db_dir=HARMONISED_OUT_DIR / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)
