# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMIP concentration inversions
#
# Invert concentrations from CMIP7.
# We do this for a very limited number of species
# where we have no other obvious option.

# %%
import json

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pooch
import scipy.interpolate
import tqdm.autonotebook as tqdman
import xarray as xr

from emissions_harmonization_historical.constants import (
    CMIP_CONCENTRATION_INVERSION_ID,
    DATA_ROOT,
    HISTORY_SCENARIO_NAME,
)
from emissions_harmonization_historical.infilling_followers import FOLLOW_LEADERS

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %%
source_id = "CR-CMIP-1-0-0"
pub_date = "v20250228"

# %%
DOWNLOAD_PATH = DATA_ROOT / "global" / "esgf" / source_id
DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)

# %%
OUT_PATH = DATA_ROOT / "global" / "esgf" / source_id / f"inverse_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.csv"
OUT_PATH_PI_EMMS = (
    DATA_ROOT / "global" / "esgf" / source_id / f"pre-industrial_emissions_{CMIP_CONCENTRATION_INVERSION_ID}.json"
)

# %%
CMIP7_TO_HERE_VARIABLE_MAP = {
    "hfc134a": "HFC134a",
    "hfc152a": "HFC152a",
    "hfc236fa": "HFC236fa",
    "hfc245fa": "HFC245fa",
    "hfc365mfc": "HFC365mfc",
    "hfc4310mee": "HFC43-10",
    "cf4": "CF4",
    "c2f6": "C2F6",
    "c3f8": "C3F8",
    "c4f10": "C4F10",
    "c5f12": "C5F12",
    "c6f14": "C6F14",
    "c7f16": "C7F16",
    "c8f18": "C8F18",
    "cc4f8": "cC4F8",
    "ch2cl2": "CH2Cl2",
    "ch3br": "CH3Br",
    "ch3cl": "CH3Cl",
    "chcl3": "CHCl3",
    "nf3": "NF3",
    "sf6": "SF6",
    "so2f2": "SO2F2",
}
here_to_cmip7_variable_map = {v: k for k, v in CMIP7_TO_HERE_VARIABLE_MAP.items()}

# %%
# Table A-5 of WMO 2022
# https://csl.noaa.gov/assessments/ozone/2022/downloads/Annex_2022OzoneAssessment.pdf
wmo_lifetimes = {
    # Gas: lifetime years
    "CF4": Q(50000.0, "yr"),
    "C2F6": Q(10000.0, "yr"),
    "C3F8": Q(2600.0, "yr"),
    "C4F10": Q(2600.0, "yr"),
    "C5F12": Q(4100.0, "yr"),
    "C6F14": Q(3100.0, "yr"),
    "C7F16": Q(3000.0, "yr"),
    "C8F18": Q(3000.0, "yr"),
    "CH2Cl2": Q(176.0 / 365.0, "yr"),
    "CH3Br": Q(0.8, "yr"),
    "CH3Cl": Q(0.9, "yr"),
    "CHCl3": Q(178.0 / 365.0, "yr"),
    "HFC134a": Q(13.5, "yr"),
    "HFC152a": Q(1.5, "yr"),
    "HFC236fa": Q(213.0, "yr"),
    "HFC245fa": Q(6.61, "yr"),
    "HFC365mfc": Q(8.86, "yr"),
    "HFC43-10": Q(17.0, "yr"),
    "NF3": Q(569.0, "yr"),
    "SF6": Q((850 + 1280) / 2.0, "yr"),
    "SO2F2": Q(36.0, "yr"),
    "cC4F8": Q(3200.0, "yr"),
}
molecular_masses = {
    # Gas: molecular mass
    "CF4": Q(12.01 + 4 * 19.0, "gCF4 / mole"),
    "C2F6": Q(2 * 12.01 + 6 * 19.0, "gC2F6 / mole"),
    "C3F8": Q(3 * 12.01 + 8 * 19.0, "gC3F8 / mole"),
    "C4F10": Q(4 * 12.01 + 10 * 19.0, "gC4F10 / mole"),
    "C5F12": Q(5 * 12.01 + 12 * 19.0, "gC5F12 / mole"),
    "C6F14": Q(6 * 12.01 + 14 * 19.0, "gC6F14 / mole"),
    "C7F16": Q(7 * 12.01 + 16 * 19.0, "gC7F16 / mole"),
    "C8F18": Q(8 * 12.01 + 18 * 19.0, "gC8F18 / mole"),
    "CH2Cl2": Q(12.01 + 2 * 1.0 + 2 * 35.45, "gCH2Cl2 / mole"),
    "CH3Br": Q(12.01 + 3 * 1.0 + 79.90, "gCH3Br / mole"),
    "CH3Cl": Q(12.01 + 3 * 1.0 + 35.45, "gCH3Cl / mole"),
    "CHCl3": Q(12.01 + 1.0 + 3 * 35.45, "gCHCl3 / mole"),
    "HFC134a": Q(12.01 + 2 * 1.0 + 19.0 + 12.01 + 3 * 19.0, "gHFC134a / mole"),  # CH2FCF3
    "HFC152a": Q(12.01 + 3 * 1.0 + 12.01 + 1.0 + 2 * 19.0, "gHFC152a / mole"),  # CH3CHF2
    "HFC236fa": Q(12.01 + 3 * 19.0 + 12.01 + 2 * 1.0 + 12.01 + 3 * 19.0, "gHFC236fa / mole"),  # CF3CH2CF3
    "HFC245fa": Q(
        12.01 + 2 * 1.0 + 19.0 + 12.01 + 2 * 19.0 + 12.01 + 1.0 + 2 * 19.0, "gHFC245fa / mole"
    ),  # CH2FCF2CHF2
    "HFC365mfc": Q(
        12.01 + 3 * 1.0 + 12.01 + 2 * 19.0 + 12.01 + 2 * 1.0 + 12.01 + 3 * 19.0, "gHFC365mfc / mole"
    ),  # CH3CF2CH2CF3
    "HFC43-10": Q(
        12.01 + 3 * 19.0 + 2 * (12.01 + 1.0 + 19.0) + 12.01 + 2 * 19.0 + 12.01 + 3 * 19.0, "gHFC4310 / mole"
    ),  # CF3CHFCHFCF2CF3
    "NF3": Q(14.01 + 3 * 19.0, "gNF3 / mole"),
    "SF6": Q(32.07 + 6 * 19.0, "gSF6 / mole"),
    "SO2F2": Q(32.07 + 2 * 16.0 + 2 * 19.0, "gSO2F2 / mole"),
    "cC4F8": Q(4 * 12.01 + 8 * 19.0, "gcC4F8 / mole"),
}

# %%
# CDIAC https://web.archive.org/web/20170118004650/http://cdiac.ornl.gov/pns/convert.html
ATMOSPHERE_MASS = Q(5.137 * 10**18, "kg")
# https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
MOLAR_MASS_DRY_AIR = Q(28.9, "g / mol")
atm_moles = (ATMOSPHERE_MASS / MOLAR_MASS_DRY_AIR).to("mole")
# Lines up with CDIAC: https://web.archive.org/web/20170118004650/http://cdiac.ornl.gov/pns/convert.html
fraction_factor = Q(1e-6, "1 / ppm")
mass_one_ppm_co2 = atm_moles * fraction_factor * Q(12.01, "gC / mole")
cdiac_expected = 2.13
if np.round(mass_one_ppm_co2.to("GtC / ppm").m, 2) != cdiac_expected:
    raise AssertionError

# %%
assumed_unit = "ppt"
background_emissions_d = {}
inverse_info_l = []
out_ts_vars = [*FOLLOW_LEADERS.keys(), "Emissions|C6F14"]
out_pi_vars = [
    "Emissions|CF4",
    "Emissions|C2F6",
    "Emissions|HFC|HFC152a",
    "Emissions|HFC|HFC236fa",
    "Emissions|HFC|HFC365mfc",
    "Emissions|HFC|HFC134a",
    "Emissions|HFC|HFC245fa",
    "Emissions|HFC|HFC43-10",
    "Emissions|SF6",
]
pi_year = 1750
found_hashes = {}
for emissions_var in tqdman.tqdm(sorted({*out_ts_vars, *out_pi_vars})):
    if "HFC" in emissions_var and emissions_var not in out_pi_vars:
        # Use Guus' data instead
        continue

    gas = emissions_var.split("|")[-1]
    variable_id_cmip = here_to_cmip7_variable_map[gas]
    download_url = f"https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP7/CMIP/CR/{source_id}/atmos/yr/{variable_id_cmip}/gm/{pub_date}/{variable_id_cmip}_input4MIPs_GHGConcentrations_CMIP_{source_id}_gm_1750-2022.nc"

    pooch.retrieve(
        download_url,
        known_hash=None,  # trust ESGF
        path=DOWNLOAD_PATH,
    )

    gas = emissions_var.split("|")[-1]
    variable_id_cmip = here_to_cmip7_variable_map[gas]
    target_unit = f"kt {gas}/yr".replace("-", "")

    lifetime = wmo_lifetimes[gas]
    molecular_mass = molecular_masses[gas]

    to_load = list(DOWNLOAD_PATH.glob(f"*{variable_id_cmip}*.nc"))
    if len(to_load) != 1:
        raise AssertionError

    ds = xr.open_dataset(to_load[0])
    units = ds[variable_id_cmip].attrs["units"]
    concs = Q(ds[variable_id_cmip].values, units)

    if units == "ppt":
        fraction_factor = Q(1e-12, "1 / ppt")
    else:
        raise NotImplementedError(units)

    emm_factor = 1 / (atm_moles * fraction_factor * molecular_mass)

    if emissions_var in out_pi_vars:
        background_emissions = concs[list(ds["time"].dt.year.values).index(pi_year)] / lifetime / emm_factor
        background_emissions_d[emissions_var] = (float(background_emissions.to(target_unit).m), target_unit)
        if emissions_var not in out_ts_vars:
            continue

    elif emissions_var in out_ts_vars:
        pass

    else:
        raise NotImplementedError(emissions_var)

    # Implicitly assume that first year values are flat
    if concs[1] != concs[0]:
        raise AssertionError

    # Probably a better way to do this, whatever.
    concs_start_year = Q(np.zeros(concs.size + 1), concs.u)
    concs_start_year[0] = concs[0]
    for i in range(1, concs_start_year.size):
        concs_start_year[i] = 2 * concs[i - 1] - concs_start_year[i - 1]
    # plt.plot(np.arange(concs_start_year.size), concs_start_year)
    # plt.step(np.arange(concs.size) + 1.0, concs)
    # a = 140
    # plt.xlim([a, a + 10])
    # plt.ylim(ymax=0.0001, ymin=0.0)
    # plt.show()

    time_step = Q(1, "yr")
    inverse_emissions = (
        1 / emm_factor * (np.diff(concs_start_year) / time_step + concs_start_year[:-1] / lifetime)
    ).to(target_unit)

    # Double check with a forward model
    concs_check = Q(np.zeros(concs.size + 1), concs.u)
    for i in range(1, concs_check.size):
        concs_check[i] = (
            emm_factor * inverse_emissions[i - 1] * time_step
            - concs_check[i - 1] / lifetime * time_step
            + concs_check[i - 1]
        )

    if lifetime.to("yr").m > 1.0:
        np.testing.assert_allclose(
            concs.to(units).m,
            ((concs_check[1:] + concs_check[:-1]) / 2.0).to(units).m,
        )
    # The check doesn't work otherwise

    x = ds[variable_id_cmip]["time"].dt.year.values.squeeze()
    y = inverse_emissions.to(target_unit).m.squeeze()

    y_smooth = scipy.interpolate.make_smoothing_spline(x=x, y=y, lam=100.0)(x)
    y_smooth[np.where(y_smooth < 0.0)] = 0.0
    y_smooth[np.where(y_smooth < y_smooth.max() * 1e-6)] = 0.0

    df = pd.DataFrame(
        np.vstack([y, y_smooth, concs.to(units).m]),
        columns=pd.Index(x, name="year"),
        index=pd.MultiIndex.from_tuples(
            (
                (f"{source_id}-inverse", HISTORY_SCENARIO_NAME, "World", emissions_var, target_unit),
                (f"{source_id}-inverse-smooth", HISTORY_SCENARIO_NAME, "World", emissions_var, target_unit),
                ("CMIP-concs", HISTORY_SCENARIO_NAME, "World", f"Atmospheric Concentrations|{gas}", units),
            ),
            names=["model", "scenario", "region", "variable", "unit"],
        ),
    )
    inverse_info_l.append(df)

# %%
background_emissions_d

# %%
inverse_info = pix.concat(inverse_info_l)
inverse_info

# %%
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
inverse_info.loc[pix.ismatch(variable="Emissions**", model=f"{source_id}-inverse-smooth")].to_csv(OUT_PATH)
OUT_PATH

# %%
OUT_PATH_PI_EMMS

# %%
OUT_PATH_PI_EMMS.parent.mkdir(exist_ok=True, parents=True)
with open(OUT_PATH_PI_EMMS, "w") as fh:
    json.dump(background_emissions_d, fh, indent=2)

OUT_PATH_PI_EMMS
