"""
Download scenario data from the IIASA database
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import pyam
import tqdm
from pandas_openscm.db import (
    DATA_BACKENDS,
    INDEX_BACKENDS,
    OpenSCMDB,
)


def main():
    """
    Download the scenario data
    """
    HERE = Path(__file__).parent

    time_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    out_dir = HERE / time_id
    db_dir = out_dir / "db"

    db = OpenSCMDB(
        db_dir=db_dir,
        backend_data=DATA_BACKENDS.get_instance("feather"),
        backend_index=INDEX_BACKENDS.get_instance("feather"),
    )

    # Establish connection to the legacy pyam platform 'ssp-submission'
    pyam.iiasa.Connection("ssp_submission")
    conn_ssp = pyam.iiasa.Connection("ssp_submission")
    props = conn_ssp.properties().reset_index()

    # Retrieve the desired scenarios' emissions
    # (download everything;
    # loop over model-scenario combinations to not blow up RAM)
    all_meta_l = []
    for model, scenario in tqdm.tqdm(
        zip(props["model"], props["scenario"]), total=len(props["model"]), desc="Model-scenario combinations"
    ):
        print(f"Grabbing {model=} {scenario=}")
        df = pyam.read_iiasa("ssp_submission", model=model, scenario=scenario, variable="Emissions|*", meta=True)
        if df.empty:
            print(f"No data for {model=} {scenario=}")
            continue

        # Write out the scenario data
        db.save(df.timeseries())

        all_meta_l.append(df.meta)

    all_meta = pd.concat(all_meta_l)
    all_meta.to_csv(out_dir / "all-meta.csv")


if __name__ == "__main__":
    main()
