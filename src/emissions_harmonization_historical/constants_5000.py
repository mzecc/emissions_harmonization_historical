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

DATA_ROOT = Path(__file__).parents[2] / "data"

# ID for the scenario download step
DOWNLOAD_SCENARIOS_ID = "0001"

# Database into which raw scenarios are saved
RAW_SCENARIO_DB = OpenSCMDB(
    db_dir=DATA_ROOT / "raw" / "scenarios" / DOWNLOAD_SCENARIOS_ID / "db",
    backend_data=FeatherDataBackend(),
    backend_index=FeatherIndexBackend(),
)
