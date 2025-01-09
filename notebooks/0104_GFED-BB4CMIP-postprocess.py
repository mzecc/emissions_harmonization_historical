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

# %% [markdown]
# The 0103 script dumped the files into separate CSVs; here we combine and make consistenct with other emissions files
# and IAMC format

# %%
import pandas as pd
import pandas_indexing as pix

from emissions_harmonization_historical.constants import DATA_ROOT, GFED_PROCESSING_ID
from emissions_harmonization_historical.units import convert_to_desired_units

# %%
data_path = DATA_ROOT / "national/gfed-bb4cmip/processed"

# %%
out_path_global = data_path / f"gfed-bb4cmip_cmip7_global_{GFED_PROCESSING_ID}.csv"
out_path_national = data_path / f"gfed-bb4cmip_cmip7_national_{GFED_PROCESSING_ID}.csv"

# %%
species = [
    "BC",
    "NMVOC",
    "CO",
    "CO2",
    "CH4",
    "N2O",
    "OC",
    "NH3",
    "NOx",
    "SO2",
]

# %%
df_list = []
# Rename variable in place
for s in species:
    for suffix in [f"_world_{GFED_PROCESSING_ID}", f"_national_{GFED_PROCESSING_ID}"]:
        df_in = pd.read_csv(data_path / f"{s}{suffix}.csv")
        df_in.variable = f"Emissions|{s}|Biomass Burning"
        df_list.append(df_in)
        # display(df_in)

# %%
df = pd.concat(df_list)

# %%
df["model"] = "BB4CMIP"
df

# %%
# sort order: region, variable
df_sorted = df.sort_values(["region", "variable"])

# %%
df_sorted

# %%
# fix column order
df_reordered = df_sorted.set_index(["model", "scenario", "region", "variable", "unit"])

# %%
df_renamed = df_reordered.rename(
    index={"Emissions|SO2|Biomass Burning": "Emissions|Sulfur|Biomass Burning"}, level="variable"
)
df_renamed

# %%
df_renamed_desired_units = convert_to_desired_units(df_renamed)

# %%
out_global = df_renamed_desired_units.loc[pix.ismatch(region="World")]
out_global

# %%
out_national = df_reordered.loc[~pix.ismatch(region="World")]
out_national

# %%
assert out_national.shape[0] + out_global.shape[0] == df_renamed_desired_units.shape[0]

# %%
out_global.to_csv(out_path_global)
out_path_global

# %%
out_national.to_csv(out_path_national)
out_path_national
