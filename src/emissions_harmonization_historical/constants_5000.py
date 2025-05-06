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

GCB_VERSION = "2024v1.0"

# ID for the CEDS processing step
GCB_PROCESSING_ID = "0001"

GCB_RAW_PATH = DATA_ROOT / "raw" / "gcb" / GCB_VERSION

# Database into which the processed GCB data is saved
GCB_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "gcb" / GCB_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

WMO_2022_RAW_PATH = DATA_ROOT / "raw" / "wmo-2022"

# ID for the WMO 2022 processing step
WMO_2022_PROCESSING_ID = "0001"

WMO_2022_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "wmo-2022" / WMO_2022_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

VELDERS_ET_AL_2022_RAW_PATH = DATA_ROOT / "raw" / "velders-et-al-2022"

# ID for the Velders et al. 2022 processing step
VELDERS_ET_AL_2022_PROCESSING_ID = "0001"

VELDERS_ET_AL_2022_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "velders-et-al-2022" / VELDERS_ET_AL_2022_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

ADAM_ET_AL_2024_RAW_PATH = DATA_ROOT / "raw" / "adam-et-al-2024"

# ID for the adam et al. 2024 processing step
ADAM_ET_AL_2024_PROCESSING_ID = "0001"

ADAM_ET_AL_2024_PROCESSED_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "processed" / "adam-et-al-2024" / ADAM_ET_AL_2024_PROCESSING_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)

CMIP7_GHG_VERSION_ID = "CR-CMIP-1-0-0"
CMIP7_GHG_PUB_DATE = "v20250228"

CMIP7_GHG_RAW_PATH = DATA_ROOT / "raw" / "cmip7-ghgs" / CMIP7_GHG_VERSION_ID

# ID for the CMIP7 GHG processing step
CMIP7_GHG_PROCESSING_ID = "0001"

CMIP7_GHG_PROCESSED_DIR = DATA_ROOT / "processed" / "cmip7-ghgs" / CMIP7_GHG_PROCESSING_ID
CMIP7_GHG_PROCESSED_DB = OpenSCMDB(
    db_dir=CMIP7_GHG_PROCESSED_DIR / "db",
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
HARMONISATION_ID = "0002"

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
