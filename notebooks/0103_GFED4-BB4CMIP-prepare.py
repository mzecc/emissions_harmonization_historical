# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from collections import defaultdict
from pathlib import Path

import numpy as np
import ptolemy
import scmdata
import xarray as xr
import xarray_regrid  # noqa: F401
from tqdm import tqdm

from emissions_harmonization_historical.constants import DATA_ROOT, GFED_PROCESSING_ID

# %%
raw_data_path = DATA_ROOT / "national/gfed-bb4cmip/data_raw/data"
raw_data_path

# %%
gfed_processed_output_dir = DATA_ROOT / Path("national", "gfed-bb4cmip", "processed")
gfed_processed_output_dir

# %%
gfed_data_aux_folder = DATA_ROOT / Path("national", "gfed", "data_aux")
gfed_isomask = Path(gfed_data_aux_folder, "iso_mask.nc")  # for aggregating to countries

# %% [markdown]
# Group raw data into variable groups which can be used for processing.

# %%
bb4cmip_files = list(raw_data_path.rglob("*.nc"))
bb4cmip_file_groups = defaultdict(list)

for file in bb4cmip_files:
    variable = file.name.split("_")[0]
    bb4cmip_file_groups[variable].append(file)

# %%
species_data = {
    "BC": {
        "unit_label": "Mt BC / yr",
        "filename_label": "BC",
    },
    "NMVOC": {
        "unit_label": "Mt NMVOC / yr",
        "filename_label": "NMVOC_bulk",
    },
    "CO": {
        "unit_label": "Mt CO / yr",
        "filename_label": "CO",
    },
    "CO2": {
        "unit_label": "Mt CO2 / yr",
        "filename_label": "CO2",
    },
    "CH4": {
        "unit_label": "Mt CH4 / yr",
        "filename_label": "CH4",
    },
    "N2O": {
        "unit_label": "Mt N2O / yr",
        "filename_label": "N2O",
    },
    "OC": {
        "unit_label": "Mt OC / yr",
        "filename_label": "OC",
    },
    "NH3": {
        "unit_label": "Mt NH3 / yr",
        "filename_label": "NH3",
    },
    "NOx": {
        "unit_label": "Mt NO / yr",
        "filename_label": "NOx",
    },
    "SO2": {
        "unit_label": "Mt SO2 / yr",
        "filename_label": "SO2",
    },
}

# %%
# cell areas of GFED map
cell_area_gfed = xr.open_mfdataset(bb4cmip_file_groups["gridcellarea"]).rename({"latitude": "lat", "longitude": "lon"})
# Get this into memory so it doesn't load every time
cell_area_gfed = cell_area_gfed.compute()
# not sure needed

# get iso map at half degree
idxr = xr.open_dataarray(gfed_isomask, chunks="auto")  # chunks={"iso": 1})
# Get this into memory so it doesn't load every time
idxr = idxr.compute()

# %%
idxr.sel(iso="aus").plot()


# %%
def gfed_to_scmrun(in_da: xr.DataArray, *, unit_label: str, world: bool = False) -> scmdata.ScmRun:
    """
    Convert an `xr.DataArray` to `scmdata.ScmRun`

    Custom to this notebook

    Parameters
    ----------
    in_da
        Input `xr.DataArray` to convert

    unit_label
        The label to apply to the unit as it will be handled in the output.

        This typically includes the species, e.g. "Mt BC / yr"

    world
        Do global emissions (rather than national)

    Returns
    -------
    :
        `scmdata.ScmRun`
    """
    df = in_da.to_numpy()

    if world:
        region = "World"
    else:
        region = in_da.iso.to_numpy()

    out = scmdata.ScmRun(
        df,
        columns=dict(
            model="GFED-BB4CMIP",
            scenario="historical",
            region=region,
            unit=unit_label,
            variable=f"Emissions|{species}|BB4CMIP",
        ),
        index=in_da.year,
    )

    return out


# %%
gfed_processed_output_dir.mkdir(exist_ok=True, parents=True)
for species in tqdm(species_data):
    species_ds = xr.open_mfdataset(bb4cmip_file_groups[species], combine_attrs="drop_conflicts").rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    # Handy for testing
    # species_ds = species_ds.isel(time=range(12 * 3))
    # Handy for seeing what's going on
    # display(species_ds)

    # First get the emissions on to an annual grid
    emissions_da = species_ds[species].chunk("auto")
    assert emissions_da.units == "kg m-2 s-1"

    # weight by month length
    seconds_per_month = (
        np.tile([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], len(emissions_da.time) // 12) * 24 * 60 * 60
    )
    weights = xr.DataArray(seconds_per_month, dims=("time",), coords=emissions_da.coords["time"].coords)

    # get annual sum emissions in each grid cell
    gridded_annual_emissions_rate = (emissions_da * weights).groupby("time.year").sum().compute()  # kg / m2 / yr

    # regrid to half degree
    regridded_annual_emissions_rate = gridded_annual_emissions_rate.regrid.conservative(
        idxr
    )  # still kg / m2 / yr, now coarser grid

    # get cell areas of new grid
    cell_area = xr.DataArray(
        ptolemy.cell_area(lats=regridded_annual_emissions_rate.lat, lons=regridded_annual_emissions_rate.lon)
    )  # m2

    # emissions in each country, annual time series
    # Mt / yr in each country, the 1e9 is kg to Mt
    assert species_ds[species].units == "kg m-2 s-1"
    country_emissions = ((regridded_annual_emissions_rate * cell_area * idxr).sum(["lat", "lon"])).compute() / 1e9
    world_emissions = (regridded_annual_emissions_rate * cell_area).sum(["lat", "lon"]).compute() / 1e9

    out_world = gfed_to_scmrun(world_emissions, unit_label=species_data[species]["unit_label"], world=True)
    out_country = gfed_to_scmrun(country_emissions, unit_label=species_data[species]["unit_label"])

    # write out temporary files to save RAM and process combined dataset later

    gfed_temp_file_world = gfed_processed_output_dir / f"{species}_world_{GFED_PROCESSING_ID}.csv"
    out_world.timeseries(time_axis="year").to_csv(gfed_temp_file_world)
    gfed_temp_file_country = gfed_processed_output_dir / f"{species}_national_{GFED_PROCESSING_ID}.csv"
    out_country.timeseries(time_axis="year").to_csv(gfed_temp_file_country)
    break
