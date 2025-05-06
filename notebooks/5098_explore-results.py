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
# # Explore results
#
# Here we post process the results.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas_indexing as pix
import pandas_openscm

from emissions_harmonization_historical.constants_5000 import (
    POST_PROCESSED_METADATA_CATEGORIES_DB,
    POST_PROCESSED_METADATA_QUANTILE_DB,
    POST_PROCESSED_TIMESERIES_RUN_ID_DB,
)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Categories

# %%
categories = POST_PROCESSED_METADATA_CATEGORIES_DB.load()["value"]
# categories

# %%
metadata_quantile = POST_PROCESSED_METADATA_QUANTILE_DB.load()["value"]
# metadata_quantile

# %%
temperatures_in_line_with_assessment = POST_PROCESSED_TIMESERIES_RUN_ID_DB.load(
    pix.isin(variable="Surface Temperature (GSAT)")
)
# temperatures_in_line_with_assessment

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Are any scenarios close to C1?

# %%
categories.unstack(["metric", "climate_model"])["category_name"].sort_values(by=["MAGICCv7.6.0a3"])

# %%
metadata_quantile.unstack(["climate_model", "metric", "unit", "quantile"]).loc[
    :,
    [
        (cm, metric, "K", p)
        for cm in [v for v in ["MAGICCv7.6.0a3", "MAGICCv7.5.3"] if v in metadata_quantile.pix.unique("climate_model")]
        for (metric, p) in [
            ("max", 0.33),
            ("max", 0.5),
            ("max", 0.67),
            ("2100", 0.5),
        ]
    ],
].sort_values(by=[("MAGICCv7.6.0a3", "max", "K", 0.33)]).round(3)

# %% [markdown]
# ### How much difference is the MAGICC update making?

# %%
iam = "COFFEE"
tmp = temperatures_in_line_with_assessment.loc[pix.ismatch(model=f"{iam}**"), 2000:]
ax = tmp.openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id", style_var="climate_model", quantiles_plumes=[(0.5, 0.9), (0.33, 0.9)]
)
ax.grid()
plt.show()

magicc_diff = (
    tmp.stack().unstack("climate_model")["MAGICCv7.6.0a3"] - tmp.stack().unstack("climate_model")["MAGICCv7.5.3"]
).unstack("time")
ax = magicc_diff.openscm.plot_plume_after_calculating_quantiles(
    quantile_over="run_id",
)
ax.grid()
plt.show()

# %%
tmp = metadata_quantile.unstack("climate_model")
magicc_diff = tmp["MAGICCv7.6.0a3"] - tmp["MAGICCv7.5.3"]
magicc_diff.unstack(["metric", "unit", "quantile"])[
    [(metric, "K", percentile) for metric in ["max", "2100"] for percentile in [0.33, 0.5, 0.67, 0.95]]
].sort_values(by=("max", "K", 0.5)).round(3)
