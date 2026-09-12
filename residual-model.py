# This code calculates the residuals of two models, one for LST and one for NDVI, using canopy cover, grass cover, and elevation as predictors.
# The residuals are then saved to CSV files for further analysis. This code accesses processed csv files provided in repository. 
# The residuals of these models are meant to represent be used as operational proxies for canopy quality of an area.
# Residual values are then validated against field data in the statistical-modeling.py scrip 

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILE
# ============================================================

input_file = (
    DATA_DIR
    / "remote_sensing_analysis"
    / "moore_tree_quality_blocks.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_file)


# ============================================================
# RESIDUAL OF LST MODEL
# ============================================================

# ============================================================
# NUMERIC VARIABLES
# ============================================================

numeric_columns = [
    "lst_avg",
    "canopy_per",
    "grass_per",
    "elev_m_mea",
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

df = df.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# TRANSFORMATION
# ============================================================

df["elev_sq"] = df["elev_m_mea"] ** 2


# ============================================================
# MODEL
# ============================================================

X = df[
    [
        "canopy_per",
        "grass_per",
        "elev_sq",
    ]
]

y = df["lst_avg"]

X = sm.add_constant(
    X,
    has_constant="add",
)

model = sm.OLS(
    y,
    X,
).fit()


# ============================================================
# MODEL SUMMARY TABLE
# ============================================================

conf_int = model.conf_int()

model_results = pd.DataFrame(
    {
        "term": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_lower_95": conf_int[0].values,
        "ci_upper_95": conf_int[1].values,
    }
)

model_results["n"] = int(model.nobs)
model_results["r_squared"] = model.rsquared
model_results["adj_r_squared"] = model.rsquared_adj
model_results["aic"] = model.aic
model_results["bic"] = model.bic


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

output_file = (
    MODEL_OUTPUT_DIR
    / "lst_residual_model_summary.csv"
)

model_results.to_csv(
    output_file,
    index=False,
    float_format="%.6f",
)


print(
    f"Model summary saved to:\n"
    f"{output_file}"
)

print()
print(model.summary())


# ============================================================
# RESIDUAL OF NDVI MODEL
# ============================================================

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_file)


# ============================================================
# NUMERIC VARIABLES
# ============================================================

numeric_columns = [
    "ndvi_avg",
    "canopy_per",
    "grass_per"
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

df = df.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# TRANSFORMATIONS
# ============================================================

df["canopy_sqrt"] = np.sqrt(
    df["canopy_per"]
)

df["grass_sqrt"] = np.sqrt(
    df["grass_per"]
)


# ============================================================
# MODEL
# ============================================================

X = df[
    [
        "canopy_sqrt",
        "grass_sqrt",
    ]
]

y = df["ndvi_avg"]

X = sm.add_constant(
    X,
    has_constant="add",
)

model = sm.OLS(
    y,
    X,
).fit()


# ============================================================
# MODEL SUMMARY TABLE
# ============================================================

conf_int = model.conf_int()

model_results = pd.DataFrame(
    {
        "term": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_lower_95": conf_int[0].values,
        "ci_upper_95": conf_int[1].values,
    }
)

model_results["n"] = int(model.nobs)
model_results["r_squared"] = model.rsquared
model_results["adj_r_squared"] = model.rsquared_adj
model_results["aic"] = model.aic
model_results["bic"] = model.bic


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

output_file = (
    MODEL_OUTPUT_DIR
    / "ndvi_residual_model_summary.csv"
)

model_results.to_csv(
    output_file,
    index=False,
    float_format="%.6f",
)


print(
    f"NDVI model summary saved to:\n"
    f"{output_file}"
)

print()
print(model.summary())

# Residuals from these two models are already added to main data 
