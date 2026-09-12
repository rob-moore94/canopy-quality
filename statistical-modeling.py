# Statistical modeling procedures used for primary results presented in:
# Robert Moore
# 9/7/26

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm, colors
from scipy.stats import pearsonr
import itertools

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "Data"
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"

# Create output directories if they do not already exist
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE PATHS
# ============================================================

GROUND_TRUTHING_FILE = (
    DATA_DIR
    / "ground_truthing_analysis"
    / "ground_truthing_analysis.csv"
)
REMOTE_SENSING_FILE = (
    DATA_DIR
    / "remote_sensing_analysis"
    / "moore_tree_quality_blocks.csv"
)

# ============================================================
# CLEAN VARIABLE NAMES
# ============================================================

name_key = {
    "genera_richness_per_hectare": "Genus richness",
    "tax_richness_per_hectare": "Tax. richness",
    "species_true_evenness": "Tax. evenness",
    "genera_true_evenness": "Genus evenness",
    "mean_height_log": "Mean Height",
    "health_color_height_prop_cubed": "Canopy Health Index(height-wieghted)",
    "health_color_squared": "Color. & Intact. (% canopy)",
    "stressors_sum_per_tree_count": "Damage score",
    "tree_health_mean": "Intactness (% canopy)",
    "soil_access_per_tree_count_sqrt": "Soil restrictions",
    "street_tree_height_sum_prop": "Street trees (% height)",
    "street_tree_percent": "Street trees (% of trees)",
    "arbuscular_height_prop": "AMF abund. (% height)",
    "arbuscular_prop": "AMF abund. (% of trees)",
    "arbuscular_height_prop_squared": "AMF abund. (% height)",
    "percent_angio_cubed": "Angio. abund. (% of trees)",
    "diffuse_prop": "D-porous abund. (% trees)",
    "diffuse_height_prop": "D-porous abund. (% height)",
    "acer_count_per_tree_count": "Maple abund. (% of trees)",
    "acer_height_sum_prop": "Maple abund. (% height)",
    "quercus_count_per_tree_sqrt": "Oak abund. (% of trees)",
    "quercus_height_sum_prop_sqrt": "Oak abund. (% height)",
    "ndvi_res": "Residuals of NDVI",
    "lst_res": "Residuals of LST",
    "ndvi_cvA_O": "CV of NDVI (Apr-Oct)",
    "ndvi_cvJ_A": "CV of NDVI (Jun-Aug)",
    "t_quality": "Integrated Tree Quality",
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(GROUND_TRUTHING_FILE)

print(f"Loaded: {GROUND_TRUTHING_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# VARIABLE TRANSFORMATIONS
# ============================================================

# Tree size
df["mean_height_log"] = np.log(df["mean_height"])

# Site / damage variables
df["stressors_sum_per_tree_count_sqrt"] = np.sqrt(
    df["stressors_sum_per_tree_count"]
)

df["soil_access_per_tree_count_sqrt"] = np.sqrt(
    df["soil_access_per_tree_count"]
)

# Health
df["health_color_height_prop_cubed"] = (
    df["health_color_height_prop"] ** 3
)
# Functional
df["arbuscular_height_prop_squared"] = (
    df["arbuscular_height_prop"] ** 2
)

df["quercus_height_sum_prop_sqrt"] = np.sqrt(
    df["quercus_height_sum_prop"]
)
# Gap-fill variables where absence represents zero
df[
    [
        "street_tree_percent",
    ]
].fillna(0)
# ============================================================
# MODEL VARIABLES
# ============================================================

dependent_vars = [
    "ndvi_avg",
    "lst_avg",
    "ndvi_res",
    "lst_res",
    "ndvi_cvA_O",
    "ndvi_cvJ_A",
    "t_quality",
]


independent_vars = [
    "genera_richness_per_hectare",
    "tax_richness_per_hectare",
    "species_true_evenness",
    "genera_true_evenness",
    "mean_height_log",
    "health_color_height_prop_cubed",
    "stressors_sum_per_tree_count",
    "tree_health_mean",
    "health_color_squared",
    "soil_access_per_tree_count_sqrt",
    "street_tree_percent",
    "arbuscular_prop",
    "arbuscular_height_prop_squared",
    "percent_angio_cubed",
    "diffuse_prop",
    "diffuse_height_prop",
    "acer_count_per_tree_count",
    "acer_height_sum_prop",
    "quercus_count_per_tree_sqrt",
    "quercus_height_sum_prop_sqrt"
]


# ============================================================
# SIMPLE OLS REGRESSIONS
# ============================================================

model_results = []

for dependent in dependent_vars:

    for independent in independent_vars:

        valid_data = (
            df[[dependent, independent]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        # Need enough observations to estimate model
        if len(valid_data) < 3:
            continue

        X = sm.add_constant(
            valid_data[independent]
        )

        y = valid_data[dependent]

        model = sm.OLS(y, X).fit()

        # Confidence interval for independent variable
        conf_int = model.conf_int().loc[independent]

        model_results.append(
            {
                "dependent_variable": dependent,
                "dependent_label": name_key.get(
                    dependent,
                    dependent,
                ),

                "independent_variable": independent,
                "independent_label": name_key.get(
                    independent,
                    independent,
                ),

                "n": int(model.nobs),

                "intercept": model.params["const"],

                "coefficient": model.params[independent],

                "coefficient_direction": (
                    "+"
                    if model.params[independent] > 0
                    else "-"
                ),

                "std_error": model.bse[independent],

                "t_value": model.tvalues[independent],

                "p_value": model.pvalues[independent],

                "ci_lower_95": conf_int.iloc[0],

                "ci_upper_95": conf_int.iloc[1],

                "r_squared": model.rsquared,

                "adj_r_squared": model.rsquared_adj,

                "f_statistic": model.fvalue,

                "f_p_value": model.f_pvalue,

                "aic": model.aic,

                "bic": model.bic,
            }
        )


# ============================================================
# MODEL RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(model_results)


# Sort results for consistent output
results_df = results_df.sort_values(
    [
        "dependent_variable",
        "p_value",
        "independent_variable",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE MODEL OUTPUT
# ============================================================

model_output_file = (
    MODEL_OUTPUT_DIR
    / "survey_simple_regressions.csv"
)

results_df.to_csv(
    model_output_file,
    index=False,
    float_format="%.6f",
)

print()
print(
    f"Simple regression results saved to:\n"
    f"{model_output_file}"
)

print(
    f"Models fitted: {len(results_df):,}"
)
 
# ============================================================
# REMOTE SENSING CORRELATION MATRIX
# ============================================================

variables = [
    "canopy_per",
    "grass_per",
    "lst_res",
    "ndvi_res",
    "ndvi_cvA_O",
    "ndvi_cvJ_A",
]

# Load only variables needed for correlation analysis
df_rs_raw = pd.read_csv(
    REMOTE_SENSING_FILE,
    usecols=variables,
    low_memory=False,
)

print()
print(f"Loaded remote sensing data: {REMOTE_SENSING_FILE}")
print(f"Rows: {len(df_rs_raw):,}")


# ============================================================
# COERCE VARIABLES TO NUMERIC
# ============================================================

for variable in variables:
    df_rs_raw[variable] = pd.to_numeric(
        df_rs_raw[variable],
        errors="coerce",
    )


# ============================================================
# CLEAN VARIABLE LABELS
# ============================================================

label_mapping = {
    "canopy_per": "Canopy\nCover",
    "grass_per": "Other Plant\nCover",
    "lst_res": "Residual\n of LST",
    "ndvi_res": "Residual\n of NDVI",
    "ndvi_cvA_O": "NDVI CV\n(Apr–Oct)",
    "ndvi_cvJ_A": "NDVI CV\n(Jun–Aug)",
}

labels = [
    label_mapping[variable]
    for variable in variables
]

df_rs = df_rs_raw.rename(
    columns=label_mapping
)


# ============================================================
# PEARSON CORRELATIONS
# ============================================================

correlation_results = []

for i in range(len(variables)):
    for j in range(i):

        var_y = variables[i]
        var_x = variables[j]

        paired_data = (
            df_rs_raw[[var_x, var_y]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(paired_data) < 3:
            continue

        r, p = pearsonr(
            paired_data[var_x],
            paired_data[var_y],
        )

        correlation_results.append(
            {
                "variable_1": var_x,
                "variable_1_label": label_mapping[var_x].replace("\n", " "),
                "variable_2": var_y,
                "variable_2_label": label_mapping[var_y].replace("\n", " "),
                "n": len(paired_data),
                "pearson_r": r,
                "p_value": p,
            }
        )


correlation_results_df = pd.DataFrame(
    correlation_results
)

correlation_output_file = (
    MODEL_OUTPUT_DIR
    / "remote_sensing_correlations.csv"
)

correlation_results_df.to_csv(
    correlation_output_file,
    index=False,
    float_format="%.6f",
)

print(
    f"Correlation results saved to:\n"
    f"{correlation_output_file}"
)


# ============================================================
# CLIP EXTREMES FOR KDE VISUALIZATION ONLY
# ============================================================

# Important:
# Correlations above use the original un-clipped values.
# Clipping here affects only the visualization.

clip_quantiles = (1, 99)

for label in labels:

    series = df_rs[label]

    valid_values = series.dropna()

    if valid_values.empty:
        continue

    lo, hi = np.nanpercentile(
        valid_values,
        clip_quantiles,
    )

    df_rs[label] = series.clip(
        lower=lo,
        upper=hi,
    )


# ============================================================
# CORRELATION MATRIX FIGURE
# ============================================================

sns.set_context(
    "talk",
    font_scale=1.3,
)

n = len(variables)

fig, axes = plt.subplots(
    n,
    n,
    figsize=(n * 4.2, n * 4.2),
)

plt.subplots_adjust(
    wspace=0.45,
    hspace=0.45,
)

cmap = cm.Blues
contourf_handles = []


# ============================================================
# BUILD LOWER-TRIANGLE MATRIX
# ============================================================

for i in range(n):

    for j in range(n):

        ax = axes[i, j]

        # Hide upper triangle
        if j > i:
            ax.axis("off")
            continue

        x_var = labels[j]
        y_var = labels[i]

        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=4)
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4)
        )

        # ----------------------------------------------------
        # Diagonal: 1D KDE
        # ----------------------------------------------------

        if i == j:

            sns.kdeplot(
                data=df_rs,
                x=x_var,
                ax=ax,
                fill=True,
                linewidth=1,
                color="gray",
            )

        # ----------------------------------------------------
        # Lower triangle: 2D KDE
        # ----------------------------------------------------

        else:

            plot_data = (
                df_rs[[x_var, y_var]]
                .dropna()
            )

            if len(plot_data) < 3:
                ax.axis("off")
                continue

            cset = sns.kdeplot(
                data=plot_data,
                x=x_var,
                y=y_var,
                ax=ax,
                fill=True,
                levels=30,
                thresh=0.05,
                cmap=cmap,
            )

            contourf_handles.append(
                cset
            )

            sns.kdeplot(
                data=plot_data,
                x=x_var,
                y=y_var,
                ax=ax,
                fill=False,
                levels=10,
                linewidths=0.7,
                color="black",
            )


            # -----------------------------------------------
            # Pearson correlation
            #
            # Use ORIGINAL, non-clipped data for statistics
            # -----------------------------------------------

            original_x = variables[j]
            original_y = variables[i]

            stat_data = (
                df_rs_raw[
                    [original_x, original_y]
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            if len(stat_data) >= 3:

                r, p = pearsonr(
                    stat_data[original_x],
                    stat_data[original_y],
                )

                stats_text = (
                    f"r = {r:.2f}\n"
                    f"p = {p:.3g}"
                )

                ax.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=19,
                    va="top",
                    ha="left",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        boxstyle="round,pad=0.3",
                    ),
                )


        # ----------------------------------------------------
        # Edge labels only
        # ----------------------------------------------------

        if i < n - 1:

            ax.set_xlabel("")
            ax.set_xticklabels([])

        else:

            ax.set_xlabel(
                x_var,
                fontsize=26,
                rotation=0,
            )


        if j > 0:

            ax.set_ylabel("")
            ax.set_yticklabels([])

        else:

            ax.set_ylabel(
                y_var,
                fontsize=26,
                rotation=0,
                labelpad=75,
                ha="center",
            )


# ============================================================
# SHARED KDE COLORBAR
# ============================================================

vmin = 0
vmax = 0

for cset in contourf_handles:

    if hasattr(cset, "collections"):

        for collection in cset.collections:

            if collection.get_array() is not None:

                values = collection.get_array()

                vmin = min(
                    vmin,
                    np.nanmin(values),
                )

                vmax = max(
                    vmax,
                    np.nanmax(values),
                )


if vmax == 0:
    vmax = 1


density_map = cm.ScalarMappable(
    cmap=cmap,
    norm=colors.Normalize(
        vmin=vmin,
        vmax=vmax,
    ),
)

density_map.set_array([])


cbar_ax = fig.add_axes(
    [0.55, 0.93, 0.4, 0.025]
)

cbar = fig.colorbar(
    density_map,
    cax=cbar_ax,
    orientation="horizontal",
)

cbar.set_label(
    "Density",
    fontsize=24,
    labelpad=5,
)

cbar.ax.tick_params(
    labelsize=20
)

cbar.ax.ticklabel_format(
    style="sci",
    scilimits=(-2, 2),
)


# ============================================================
# SAVE FIGURE
# ============================================================

figure_output_file = (
    FIGURE_OUTPUT_DIR
    / "remote_sensing_density_correlation_matrix.png"
)

plt.tight_layout()

plt.savefig(
    figure_output_file,
    dpi=600,
    bbox_inches="tight",
)

plt.close()

print(
    f"Correlation matrix saved to:\n"
    f"{figure_output_file}"
)

# ============================================================
# CORRELATIONS AMONG INDEPENDENT VARIABLES
# ============================================================

independent_correlation_results = []

for i in range(len(independent_vars)):
    for j in range(i):

        variable_1 = independent_vars[j]
        variable_2 = independent_vars[i]

        # Pairwise complete observations
        paired_data = (
            df[[variable_1, variable_2]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(paired_data) < 3:
            continue

        r, p = pearsonr(
            paired_data[variable_1],
            paired_data[variable_2],
        )

        independent_correlation_results.append(
            {
                "variable_1": variable_1,
                "variable_1_label": name_key.get(
                    variable_1,
                    variable_1,
                ),
                "variable_2": variable_2,
                "variable_2_label": name_key.get(
                    variable_2,
                    variable_2,
                ),
                "n": len(paired_data),
                "pearson_r": r,
                "p_value": p,
            }
        )


independent_correlation_df = pd.DataFrame(
    independent_correlation_results
)


# Sort strongest absolute correlations first
independent_correlation_df["abs_r"] = (
    independent_correlation_df["pearson_r"].abs()
)

independent_correlation_df = (
    independent_correlation_df
    .sort_values(
        "abs_r",
        ascending=False,
    )
    .drop(columns="abs_r")
    .reset_index(drop=True)
)


# Save results
independent_correlation_output_file = (
    MODEL_OUTPUT_DIR
    / "independent_variable_correlations.csv"
)

independent_correlation_df.to_csv(
    independent_correlation_output_file,
    index=False,
    float_format="%.6f",
)

print()
print(
    f"Independent-variable correlations saved to:\n"
    f"{independent_correlation_output_file}"
)

# ============================================================
# BEST-SUBSETS REGRESSION
# ============================================================

# Best-subsets settings
TOP_N_PER_K = 3
K_MIN = 2
K_MAX = 5
VIF_THRESHOLD = 4.0
P_VALUE_THRESHOLD = 0.10
SIGNIFICANCE_STAR = 0.01


# ============================================================
# BEST-SUBSETS HELPER FUNCTIONS
# ============================================================

def max_vif(X_data, feature_set):
    """
    Calculate maximum VIF among predictors in a model.
    Intercept is excluded from the reported maximum VIF.
    """

    X_subset = sm.add_constant(
        X_data[list(feature_set)],
        has_constant="add",
    )

    vifs = []

    for i in range(X_subset.shape[1]):
        vif = variance_inflation_factor(
            X_subset.values,
            i,
        )

        vifs.append(vif)

    vif_table = pd.DataFrame(
        {
            "feature": X_subset.columns,
            "vif": vifs,
        }
    )

    vif_table = vif_table[
        vif_table["feature"] != "const"
    ]

    return vif_table["vif"].max()


def process_subset(
    y,
    X,
    dependent,
    feature_set,
    vif_threshold=4.0,
    pval_max=0.10,
    sig_star=0.01,
):
    """
    Fit and evaluate one candidate OLS model.

    Model is rejected if:
      - maximum VIF > vif_threshold
      - any predictor p-value > pval_max

    Standardized beta coefficients are also calculated.
    """

    # --------------------------------------------------------
    # VIF FILTER
    # --------------------------------------------------------

    this_max_vif = max_vif(
        X,
        feature_set,
    )

    if (
        not np.isfinite(this_max_vif)
        or this_max_vif > vif_threshold
    ):
        return None


    # --------------------------------------------------------
    # FIT OLS MODEL
    # --------------------------------------------------------

    X_sub = sm.add_constant(
        X[list(feature_set)],
        has_constant="add",
    )

    model = sm.OLS(
        y,
        X_sub,
    ).fit()


    # --------------------------------------------------------
    # P-VALUE FILTER
    # --------------------------------------------------------

    p_values = model.pvalues.drop(
        labels="const",
        errors="ignore",
    )

    if (p_values > pval_max).any():
        return None


    # --------------------------------------------------------
    # STANDARDIZED BETAS
    # --------------------------------------------------------

    y_std = y.std(ddof=0)

    X_raw = X[list(feature_set)]
    X_std = X_raw.std(ddof=0)

    # Reject model if a variable has no variation
    if y_std == 0 or (X_std == 0).any():
        return None

    y_z = (
        (y - y.mean())
        / y_std
    )

    X_z = (
        (X_raw - X_raw.mean())
        / X_std
    )

    X_z = sm.add_constant(
        X_z,
        has_constant="add",
    )

    standardized_model = sm.OLS(
        y_z,
        X_z,
    ).fit()

    standardized_betas = (
        standardized_model.params.drop(
            labels="const",
            errors="ignore",
        )
    )


    # --------------------------------------------------------
    # PREDICTOR LABELS
    # --------------------------------------------------------

    predictor_labels = []

    for variable in feature_set:

        clean_name = name_key.get(
            variable,
            variable,
        )

        beta = standardized_betas[
            variable
        ]

        label = (
            f"{clean_name} "
            f"({beta:.2f})"
        )

        if model.pvalues[variable] < sig_star:
            label += "*"

        predictor_labels.append(label)


    # --------------------------------------------------------
    # RETURN MODEL RESULTS
    # --------------------------------------------------------

    return {
        "dependent_variable": dependent,

        "dependent_label": name_key.get(
            dependent,
            dependent,
        ),

        "n": int(model.nobs),

        "k": len(feature_set),

        "r_squared": model.rsquared,

        "adj_r_squared": model.rsquared_adj,

        "aic": model.aic,

        "bic": model.bic,

        "max_vif": this_max_vif,

        "predictors": predictor_labels,
    }


# ============================================================
# RUN BEST-SUBSETS FOR EACH DEPENDENT VARIABLE
# ============================================================

all_best_models = []
best_subsets_dependent_vars = [
    dependent
    for dependent in dependent_vars
    if dependent not in [
        "ndvi_avg",
        "lst_avg",
    ]
]
for dependent in best_subsets_dependent_vars:

    print()
    print(
        f"Running best-subsets models for: "
        f"{dependent}"
    )

    # --------------------------------------------------------
    # Use the SAME independent variables as simple regressions
    # --------------------------------------------------------

    model_variables = (
        [dependent]
        + independent_vars
    )

    # Replace infinite values and use complete cases.
    #
    # Using one common dataset for all candidate models within
    # this dependent variable keeps AIC values comparable.
    model_df = (
        df[model_variables]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    if model_df.empty:
        print(
            f"No complete observations for {dependent}. "
            f"Skipping."
        )
        continue

    y = model_df[dependent]

    X = model_df[independent_vars]

    print(
        f"Rows used: {len(model_df)}"
    )


    # --------------------------------------------------------
    # TEST MODELS WITH K = 2 THROUGH 5 PREDICTORS
    # --------------------------------------------------------

    dependent_models = []

    for k in range(
        K_MIN,
        K_MAX + 1,
    ):

        results_k = []

        for combination in itertools.combinations(
            independent_vars,
            k,
        ):

            result = process_subset(
                y=y,
                X=X,
                dependent=dependent,
                feature_set=combination,
                vif_threshold=VIF_THRESHOLD,
                pval_max=P_VALUE_THRESHOLD,
                sig_star=SIGNIFICANCE_STAR,
            )

            if result is not None:
                results_k.append(result)


        # ----------------------------------------------------
        # KEEP TOP 3 MODELS FOR THIS NUMBER OF PREDICTORS
        # ----------------------------------------------------

        results_k = sorted(
            results_k,
            key=lambda result: result["aic"],
        )[:TOP_N_PER_K]

        dependent_models.extend(
            results_k
        )

        print(
            f"  k = {k}: "
            f"{len(results_k)} models retained"
        )


    # --------------------------------------------------------
    # DELTA AIC WITHIN DEPENDENT VARIABLE
    # --------------------------------------------------------

    if dependent_models:

        minimum_aic = min(
            model["aic"]
            for model in dependent_models
        )

        for model in dependent_models:

            model["delta_aic"] = (
                model["aic"]
                - minimum_aic
            )

        all_best_models.extend(
            dependent_models
        )

    else:

        print(
            f"No candidate models survived "
            f"for {dependent}."
        )


# ============================================================
# FORMAT BEST-SUBSETS OUTPUT
# ============================================================

if not all_best_models:

    raise ValueError(
        "No best-subsets models survived the "
        "VIF and p-value filters."
    )


best_subsets_df = pd.DataFrame(
    all_best_models
)


# Find largest model retained
max_predictors = (
    best_subsets_df["predictors"]
    .apply(len)
    .max()
)


# Expand predictor list into separate columns
predictor_columns = [
    f"predictor_{i + 1}"
    for i in range(max_predictors)
]


expanded_predictors = (
    best_subsets_df["predictors"]
    .apply(
        lambda predictors: pd.Series(
            predictors
            + [""] * (
                max_predictors
                - len(predictors)
            )
        )
    )
)

expanded_predictors.columns = (
    predictor_columns
)


# Combine model statistics and predictors
best_subsets_final = pd.concat(
    [
        best_subsets_df[
            [
                "dependent_variable",
                "dependent_label",
                "n",
                "k",
                "r_squared",
                "adj_r_squared",
                "aic",
                "delta_aic",
                "bic",
                "max_vif",
            ]
        ],

        expanded_predictors,
    ],

    axis=1,
)


# Sort within dependent variables by Delta AIC
best_subsets_final = (
    best_subsets_final
    .sort_values(
        [
            "dependent_variable",
            "k",
            "delta_aic",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE BEST-SUBSETS RESULTS
# ============================================================

best_subsets_output_file = (
    MODEL_OUTPUT_DIR
    / "survey_best_subsets.csv"
)


best_subsets_final.to_csv(
    best_subsets_output_file,
    index=False,
    float_format="%.6f",
)


print()
print(
    f"Best-subsets results saved to:\n"
    f"{best_subsets_output_file}"
)

print(
    f"Total models saved: "
    f"{len(best_subsets_final):,}"
)