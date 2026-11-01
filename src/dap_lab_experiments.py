"""
Fitness and Workout Progress Analyzer
Module: dap_lab_experiments.py
Description: Full implementation of DAP Lab Syllabus Experiments (EXP 1 to EXP 8)
             using pandas, numpy, scipy, statsmodels, and matplotlib/seaborn.
"""

import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
# Only set Agg backend if not in an interactive IPython/Jupyter environment
if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

# Output plots directory
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output_plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 12
plt.rcParams["axes.titlesize"] = 11

def print_section_header(title: str):
    width = 75
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width + "\n")

# ==============================================================================
# EXP 1 : DATA HANDLING
# ==============================================================================
def run_exp1_data_handling(df: pd.DataFrame, plots_dir: str = PLOTS_DIR) -> dict:
    """
    EXP 1 : Data Handling
    1. Load a dataset into a pandas Data Frame.
    2. Classify and tabulate data types using pandas.
    3. Visualize data with histograms, scatter plots, and box plots.
    """
    print_section_header("EXP 1 : Data Handling & Type Classification")
    
    # 1. Dataset Shape & Overview
    print(f"[1] Dataset successfully loaded into pandas DataFrame.")
    print(f"    - Total Observations (Rows): {df.shape[0]}")
    print(f"    - Total Attributes (Columns): {df.shape[1]}\n")

    # 2. Classify and Tabulate Data Types
    type_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        category = "Numerical (Continuous)" if dtype in ["float64", "float32"] else (
            "Numerical (Discrete)" if dtype in ["int64", "int32"] else "Categorical (Nominal/Ordinal)"
        )
        non_null = df[col].notnull().sum()
        unique = df[col].nunique()
        sample_val = str(df[col].iloc[0])
        type_info.append([col, dtype, category, non_null, unique, sample_val])
    
    headers = ["Column Name", "Data Type", "Classification", "Non-Null", "Unique", "Sample Value"]
    print("[2] Classified Data Types Table:")
    print(tabulate(type_info, headers=headers, tablefmt="grid"))
    
    # 3. Visualizations: Histograms, Scatter Plots, Box Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EXP 1: Exploratory Data Handling & Visualizations", fontsize=15, weight="bold")
    
    # Histogram of Calories Burned with KDE
    sns.histplot(df["Calories_Burned"], kde=True, color="#2b5c8f", bins=25, ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Calories Burned (Histogram + KDE)")
    axes[0, 0].set_xlabel("Calories Burned (kcal)")
    axes[0, 0].set_ylabel("Frequency")
    
    # Histogram of Session Duration
    sns.histplot(df["Session_Duration (hours)"], kde=True, color="#e67e22", bins=20, ax=axes[0, 1])
    axes[0, 1].set_title("Distribution of Session Duration (Histogram + KDE)")
    axes[0, 1].set_xlabel("Session Duration (hours)")
    axes[0, 1].set_ylabel("Frequency")
    
    # Scatter Plot: Session Duration vs Calories Burned by Workout Type
    sns.scatterplot(
        data=df,
        x="Session_Duration (hours)",
        y="Calories_Burned",
        hue="Workout_Type",
        style="Gender",
        palette="Set1",
        alpha=0.85,
        s=60,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Scatter Plot: Duration vs. Calories Burned by Workout Type")
    axes[1, 0].set_xlabel("Session Duration (hours)")
    axes[1, 0].set_ylabel("Calories Burned (kcal)")
    
    # Box Plot of Calories Burned by Workout Type
    sns.boxplot(
        data=df,
        x="Workout_Type",
        y="Calories_Burned",
        hue="Workout_Type",
        palette="Pastel1",
        legend=False,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Box Plot: Calories Burned Across Workout Categories")
    axes[1, 1].set_xlabel("Workout Type")
    axes[1, 1].set_ylabel("Calories Burned (kcal)")
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "exp1_data_handling.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[3] Visualizations saved to: {plot_path}")
    return {"classified_types": type_info, "plot_path": plot_path}

# ==============================================================================
# EXP 2 : CENTRAL TENDENCY AND VARIABILITY
# ==============================================================================
def run_exp2_central_tendency_variability(df: pd.DataFrame, plots_dir: str = PLOTS_DIR) -> dict:
    """
    EXP 2 : Central Tendency and Variability
    1. Calculate mean, median, mode, quartiles, interquartile range (IQR), variance, standard deviation.
    2. Display data distribution with an annotated box plot.
    """
    print_section_header("EXP 2 : Measures of Central Tendency and Variability")
    
    target_cols = ["Calories_Burned", "Session_Duration (hours)", "Avg_BPM", "Weight (kg)"]
    stats_data = []
    
    for col in target_cols:
        series = df[col]
        mean_val = series.mean()
        median_val = series.median()
        mode_val = series.mode().iloc[0]
        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50)  # Median
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        variance_val = series.var()
        std_val = series.std()
        
        # Outlier boundaries using 1.5 * IQR rule
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = ((series < lower_bound) | (series > upper_bound)).sum()
        
        stats_data.append([
            col,
            f"{mean_val:.2f}",
            f"{median_val:.2f}",
            f"{mode_val:.2f}",
            f"{q1:.2f}",
            f"{q3:.2f}",
            f"{iqr:.2f}",
            f"{variance_val:.2f}",
            f"{std_val:.2f}",
            outliers_count
        ])
    
    headers = ["Feature", "Mean", "Median", "Mode", "Q1 (25%)", "Q3 (75%)", "IQR", "Variance", "Std Dev", "Outliers"]
    print("[1] Parametric & Non-Parametric Summary Statistics Table:")
    print(tabulate(stats_data, headers=headers, tablefmt="grid"))
    
    # 2. Display annotated distribution box plot for Calories Burned
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EXP 2: Central Tendency & Dispersion Analysis", fontsize=15, weight="bold")
    
    # Annotated Box Plot for Calories Burned
    bp = axes[0].boxplot(
        df["Calories_Burned"],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#3498db", alpha=0.6, color="#2980b9"),
        medianprops=dict(color="#e74c3c", linewidth=2.5),
        whiskerprops=dict(color="#2c3e50", linewidth=1.5),
        capprops=dict(color="#2c3e50", linewidth=1.5),
        flierprops=dict(marker="o", color="#e74c3c", markersize=6)
    )
    
    q1 = df["Calories_Burned"].quantile(0.25)
    med = df["Calories_Burned"].median()
    q3 = df["Calories_Burned"].quantile(0.75)
    mean_cb = df["Calories_Burned"].mean()
    iqr = q3 - q1
    
    axes[0].text(1.15, q1, f"Q1 = {q1:.1f}", color="#2c3e50", weight="bold", va="center")
    axes[0].text(1.15, med, f"Median = {med:.1f}", color="#c0392b", weight="bold", va="center")
    axes[0].text(1.15, q3, f"Q3 = {q3:.1f}", color="#2c3e50", weight="bold", va="center")
    axes[0].text(1.15, mean_cb, f"Mean = {mean_cb:.1f}", color="#27ae60", weight="bold", va="center")
    axes[0].text(1.15, q3 + 1.5 * iqr, f"Upper Fence = {q3 + 1.5*iqr:.1f}", color="#7f8c8d", va="center")
    axes[0].text(1.15, max(q1 - 1.5 * iqr, df["Calories_Burned"].min()), f"Lower Fence = {max(q1 - 1.5*iqr, df['Calories_Burned'].min()):.1f}", color="#7f8c8d", va="center")
    
    axes[0].set_title("Annotated Box Plot: Calories Burned")
    axes[0].set_ylabel("Calories Burned (kcal)")
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["Gym Cohort"])
    
    # Kernel Density + Mean & Median Lines
    sns.kdeplot(df["Calories_Burned"], fill=True, color="#3498db", ax=axes[1], alpha=0.3)
    axes[1].axvline(mean_cb, color="#27ae60", linestyle="--", linewidth=2, label=f"Mean: {mean_cb:.1f}")
    axes[1].axvline(med, color="#e74c3c", linestyle="-", linewidth=2, label=f"Median: {med:.1f}")
    axes[1].set_title("Density Distribution with Central Tendency Landmarks")
    axes[1].set_xlabel("Calories Burned (kcal)")
    axes[1].set_ylabel("Probability Density")
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "exp2_variability.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[2] Annotated Box Plot saved to: {plot_path}")
    return {"stats_summary": stats_data, "plot_path": plot_path}

# ==============================================================================
# EXP 3 : TRANSFORMATIONS AND WEIGHTED AVERAGES
# ==============================================================================
def run_exp3_transformations_weighted_averages(df: pd.DataFrame, plots_dir: str = PLOTS_DIR) -> dict:
    """
    EXP 3 : Transformations and Weighted Averages
    1. Apply log transformation and calculate the geometric mean.
    2. Compute a weighted mean.
    3. Analyze transformation effects on statistical measures.
    """
    print_section_header("EXP 3 : Data Transformations & Weighted Averages")
    
    cb = df["Calories_Burned"]
    
    # 1. Log Transformation & Geometric Mean
    log_cb = np.log(cb)
    
    # Geometric mean: exp(mean(log(x)))
    geom_mean_manual = np.exp(np.mean(log_cb))
    geom_mean_scipy = stats.gmean(cb)
    
    print(f"[1] Logarithmic Transformation Applied:")
    print(f"    - Arithmetic Mean of Calories Burned: {cb.mean():.3f} kcal")
    print(f"    - Geometric Mean (Formula: exp(E[ln(X)])): {geom_mean_manual:.3f} kcal")
    print(f"    - Geometric Mean (scipy.stats verification): {geom_mean_scipy:.3f} kcal")
    print(f"    - Note: Arithmetic Mean >= Geometric Mean holds (AM-GM Inequality: {cb.mean():.2f} >= {geom_mean_manual:.2f})\n")
    
    # 2. Weighted Mean
    # Compute calories burned weighted by Session Duration
    weights_duration = df["Session_Duration (hours)"]
    weighted_mean_duration = np.average(cb, weights=weights_duration)
    
    # Compute calories burned weighted by Workout Frequency
    weights_freq = df["Workout_Frequency (days/week)"]
    weighted_mean_freq = np.average(cb, weights=weights_freq)
    
    print(f"[2] Weighted Averages:")
    print(f"    - Unweighted Arithmetic Mean: {cb.mean():.3f} kcal")
    print(f"    - Mean Calories Weighted by Session Duration: {weighted_mean_duration:.3f} kcal")
    print(f"      (Heavier weighting on longer sessions shifts mean upwards by {weighted_mean_duration - cb.mean():+.2f} kcal)")
    print(f"    - Mean Calories Weighted by Workout Frequency: {weighted_mean_freq:.3f} kcal\n")
    
    # 3. Analyze Transformation Effects on Distribution
    pre_skew = stats.skew(cb)
    post_skew = stats.skew(log_cb)
    pre_kurt = stats.kurtosis(cb)
    post_kurt = stats.kurtosis(log_cb)
    pre_std = np.std(cb)
    post_std = np.std(log_cb)
    
    comparison_table = [
        ["Metric", "Raw Scale (Calories_Burned)", "Log Transformed ln(Calories_Burned)", "Shift Effect"],
        ["Sample Size (N)", len(cb), len(log_cb), "Invariant"],
        ["Mean", f"{cb.mean():.2f}", f"{log_cb.mean():.4f}", f"Stabilized scale"],
        ["Standard Deviation", f"{pre_std:.2f}", f"{post_std:.4f}", "Variance compressed"],
        ["Skewness (Asymmetry)", f"{pre_skew:.4f}", f"{post_skew:.4f}", f"Reduced absolute skew by {abs(pre_skew) - abs(post_skew):.4f}"],
        ["Kurtosis (Tails)", f"{pre_kurt:.4f}", f"{post_kurt:.4f}", "Approaches Gaussian profile"]
    ]
    print("[3] Statistical Impact of Log Transformation:")
    print(tabulate(comparison_table[1:], headers=comparison_table[0], tablefmt="grid"))
    
    # Visualization: Before vs After Transformation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EXP 3: Distributional Effects of Logarithmic Transformation", fontsize=15, weight="bold")
    
    # Raw Histogram + Normal Fit
    sns.histplot(cb, kde=True, color="#34495e", stat="density", ax=axes[0, 0])
    axes[0, 0].set_title(f"Original Calories Burned (Skewness = {pre_skew:.2f})")
    axes[0, 0].set_xlabel("Calories Burned (kcal)")
    
    # Log Transformed Histogram + Normal Fit
    sns.histplot(log_cb, kde=True, color="#16a085", stat="density", ax=axes[0, 1])
    axes[0, 1].set_title(f"Log-Transformed ln(Calories) (Skewness = {post_skew:.2f})")
    axes[0, 1].set_xlabel("ln(Calories Burned)")
    
    # Q-Q Plot Original
    stats.probplot(cb, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot: Original Data vs. Normal")
    
    # Q-Q Plot Log Transformed
    stats.probplot(log_cb, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot: Log-Transformed Data vs. Normal")
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "exp3_transformations.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[4] Transformation Q-Q and Distribution Plots saved to: {plot_path}")
    return {
        "geometric_mean": geom_mean_manual,
        "weighted_mean_duration": weighted_mean_duration,
        "pre_skew": pre_skew,
        "post_skew": post_skew,
        "plot_path": plot_path
    }

# ==============================================================================
# EXP 4 : CORRELATION ANALYSIS
# ==============================================================================
def run_exp4_correlation_analysis(df: pd.DataFrame, plots_dir: str = PLOTS_DIR) -> dict:
    """
    EXP 4 : Correlation Analysis
    1. Compute Pearson and Spearman correlation coefficients.
    2. Visualize relationships with scatter plots and heatmaps.
    3. Interpret correlations.
    """
    print_section_header("EXP 4 : Correlation Analysis (Pearson vs. Spearman)")
    
    numeric_df = df.select_dtypes(include=[np.number])
    target = "Calories_Burned"
    
    # 1. Pearson (Linear) and Spearman (Rank/Monotonic) Correlation
    pearson_corr = numeric_df.corr(method="pearson")
    spearman_corr = numeric_df.corr(method="spearman")
    
    # Extract correlations with Calories_Burned
    corrs = []
    for col in numeric_df.columns:
        if col == target:
            continue
        p_val = pearson_corr.loc[col, target]
        s_val = spearman_corr.loc[col, target]
        
        # P-values for significance testing
        p_sig = stats.pearsonr(df[col], df[target])[1]
        
        strength = "Very Strong" if abs(p_val) > 0.8 else (
            "Strong" if abs(p_val) > 0.6 else (
                "Moderate" if abs(p_val) > 0.4 else (
                    "Weak" if abs(p_val) > 0.2 else "Very Weak / Negligible"
                )
            )
        )
        nature = "Linear" if abs(p_val - s_val) < 0.05 else "Monotonic/Non-linear"
        corrs.append([col, f"{p_val:+.4f}", f"{s_val:+.4f}", f"{p_sig:.2e}", strength, nature])
    
    # Sort by absolute Pearson correlation
    corrs.sort(key=lambda x: abs(float(x[1])), reverse=True)
    
    headers = ["Feature", "Pearson (r)", "Spearman (rho)", "p-value", "Strength", "Association Nature"]
    print(f"[1] Correlation Coefficients with Target ({target}):")
    print(tabulate(corrs, headers=headers, tablefmt="grid"))
    
    # 2. Visualize with Heatmap & Dual Scatter Plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle("EXP 4: Correlation Matrix & Bivariate Relationships", fontsize=15, weight="bold")
    
    # Correlation Heatmap
    ax_heat = fig.add_subplot(gs[0, :])
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    sns.heatmap(
        pearson_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax_heat
    )
    ax_heat.set_title("Pearson Correlation Heatmap (Lower Triangle)")
    
    # Top Correlated Feature 1: Session_Duration
    ax_scat1 = fig.add_subplot(gs[1, 0])
    sns.regplot(
        data=df,
        x="Session_Duration (hours)",
        y=target,
        scatter_kws={"alpha": 0.5, "color": "#2980b9"},
        line_kws={"color": "#c0392b", "linewidth": 2},
        ax=ax_scat1
    )
    r_dur = pearson_corr.loc["Session_Duration (hours)", target]
    ax_scat1.set_title(f"Session Duration vs. Calories (Pearson r = {r_dur:.3f})")
    ax_scat1.set_xlabel("Session Duration (hours)")
    ax_scat1.set_ylabel("Calories Burned (kcal)")
    
    # Top Correlated Feature 2: Avg_BPM or Weight
    top_col = "Avg_BPM" if "Avg_BPM" in numeric_df.columns else "Weight (kg)"
    ax_scat2 = fig.add_subplot(gs[1, 1])
    sns.regplot(
        data=df,
        x=top_col,
        y=target,
        scatter_kws={"alpha": 0.5, "color": "#27ae60"},
        line_kws={"color": "#e67e22", "linewidth": 2},
        ax=ax_scat2
    )
    r_col = pearson_corr.loc[top_col, target]
    ax_scat2.set_title(f"{top_col} vs. Calories (Pearson r = {r_col:.3f})")
    ax_scat2.set_xlabel(top_col)
    ax_scat2.set_ylabel("Calories Burned (kcal)")
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "exp4_correlation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[2] Correlation Heatmap and Scatter Plots saved to: {plot_path}")
    print("[3] Correlation Interpretation:")
    print("    - Session Duration exhibits the dominant positive linear correlation with Calories Burned.")
    print("    - Avg BPM and Weight exhibit secondary positive contributions to energy expenditure.")
    print("    - Spearman rho closely mirrors Pearson r, indicating consistent monotonic trends.")
    
    return {"correlations": corrs, "plot_path": plot_path}

# ==============================================================================
# EXP 5 : REGRESSION
# ==============================================================================
def run_exp5_regression(df: pd.DataFrame, plots_dir: str = PLOTS_DIR) -> dict:
    """
    EXP 5 : Regression
    1. Build and visualize a simple linear regression model.
    2. Compare correlation and regression.
    """
    print_section_header("EXP 5 : Simple Linear Regression & Correlation Comparison")
    
    x_col = "Session_Duration (hours)"
    y_col = "Calories_Burned"
    
    X = df[[x_col]].values
    y = df[y_col].values
    
    # 1. Fit Ordinary Least Squares (OLS) Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    
    slope = lr.coef_[0]
    intercept = lr.intercept_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # 2. Mathematical Comparison between Correlation and Regression
    r_pearson, p_val = stats.pearsonr(df[x_col], df[y_col])
    sx = df[x_col].std()
    sy = df[y_col].std()
    derived_slope = r_pearson * (sy / sx)
    
    print(f"[1] Fitted Simple Linear Regression Model:")
    print(f"    Equation : Calories_Burned = {intercept:.2f} + {slope:.2f} * Session_Duration")
    print(f"    - Slope (beta_1) : {slope:.4f} kcal per hour")
    print(f"    - Intercept (beta_0) : {intercept:.4f} kcal")
    print(f"    - R-squared (R^2) : {r2:.4f} ({r2*100:.2f}% of calorie variance explained by duration)")
    print(f"    - RMSE : {rmse:.2f} kcal\n")
    
    comparison_table = [
        ["Attribute", "Correlation (Pearson r)", "Regression (OLS Model)"],
        ["Metric Value", f"r = {r_pearson:.4f}", f"Slope = {slope:.4f}, Intercept = {intercept:.2f}"],
        ["Symmetry", "Symmetric: corr(X, Y) == corr(Y, X)", "Asymmetric: Predicts Y given X (directional)"],
        ["Units", "Dimensionless [-1.0, +1.0]", "Units of Y per unit of X (kcal/hour)"],
        ["Relationship Identity", f"r^2 = ({r_pearson:.4f})^2 = {r_pearson**2:.4f}", f"R^2 = {r2:.4f} (Strictly Identical for simple regression!)"],
        ["Slope Derivation", f"beta_1 = r * (sy / sx)", f"{r_pearson:.4f} * ({sy:.2f} / {sx:.2f}) = {derived_slope:.4f} (Exact Match)"],
        ["Utility", "Measures degree & direction of association", "Enables quantitative forecasting and estimation"]
    ]
    print("[2] Formal Mathematical Comparison: Correlation vs. Regression:")
    print(tabulate(comparison_table[1:], headers=comparison_table[0], tablefmt="grid"))
    
    # 3. Visualizations: Regression Fit + Residual Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EXP 5: Simple Linear Regression Analysis & Residual Diagnostic", fontsize=15, weight="bold")
    
    # Scatter + Fitted Regression Line
    axes[0].scatter(X, y, alpha=0.5, color="#2980b9", label="Observed Gym Sessions")
    axes[0].plot(X, y_pred, color="#c0392b", linewidth=2.5, label=f"Fit: y = {intercept:.1f} + {slope:.1f}x (R²={r2:.2f})")
    axes[0].set_title(f"Linear Fit: {x_col} vs. {y_col}")
    axes[0].set_xlabel("Session Duration (hours)")
    axes[0].set_ylabel("Calories Burned (kcal)")
    axes[0].legend()
    
    # Residual Plot
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color="#8e44ad")
    axes[1].axhline(0, color="#c0392b", linestyle="--", linewidth=2)
    axes[1].set_title("Residual Plot (Errors vs. Predicted Values)")
    axes[1].set_xlabel("Fitted Values (kcal)")
    axes[1].set_ylabel("Residuals (Observed - Predicted)")
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "exp5_linear_regression.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[3] Regression Fit and Residual plots saved to: {plot_path}")
    return {"slope": slope, "intercept": intercept, "r2": r2, "rmse": rmse, "plot_path": plot_path}

# ==============================================================================
# EXP 6 : SAMPLING AND HYPOTHESIS TESTING
# ==============================================================================
def run_exp6_sampling_hypothesis_testing(df: pd.DataFrame, plots_dir: str = PLOTS_DIR) -> dict:
    """
    EXP 6 : Sampling and Hypothesis Testing
    1. Execute simple random sampling.
    2. Perform a t-test for mean comparison.
    3. Interpret hypothesis testing outcomes.
    """
    print_section_header("EXP 6 : Sampling & Hypothesis Testing (Two-Sample t-Test)")
    
    target = "Calories_Burned"
    pop_mean = df[target].mean()
    pop_std = df[target].std()
    
    print(f"[1] Population Parameters (Full Dataset, N = {len(df)}):")
    print(f"    - True Population Mean (mu) : {pop_mean:.2f} kcal")
    print(f"    - True Population Std (sigma): {pop_std:.2f} kcal\n")
    
    # Simple Random Sampling Demonstration with various sample sizes
    sample_sizes = [30, 75, 150, 300]
    sampling_results = []
    
    np.random.seed(42)
    for n in sample_sizes:
        sample = df.sample(n=n, random_state=42)[target]
        x_bar = sample.mean()
        s = sample.std()
        se = s / np.sqrt(n)
        error = abs(x_bar - pop_mean)
        sampling_results.append([n, f"{x_bar:.2f}", f"{s:.2f}", f"{se:.2f}", f"{error:.2f}"])
    
    s_headers = ["Sample Size (n)", "Sample Mean (x_bar)", "Sample Std (s)", "Std Error (SE)", "Abs Error (|x_bar - mu|)"]
    print("[2] Simple Random Sampling (SRS) Convergence Table (Central Limit Theorem):")
    print(tabulate(sampling_results, headers=s_headers, tablefmt="grid"))
    
    # 2b. Secondary Large-Scale Dataset Sampling (calories.csv - 15,000 observations)
    try:
        try:
            from src.data_loader import load_calories_dataset
        except ModuleNotFoundError:
            from data_loader import load_calories_dataset
        cal_large = load_calories_dataset()
        cal_pop_mean = cal_large["Calories"].mean()
        print(f"\n[2b] Large-Scale Population Sampling (calories.csv, N = {len(cal_large)}):")
        print(f"     - True Population Mean (mu) = {cal_pop_mean:.2f} kcal")
        large_sample_res = []
        for n_lg in [100, 500, 1000, 2500]:
            sm = cal_large.sample(n=n_lg, random_state=42)["Calories"]
            large_sample_res.append([n_lg, f"{sm.mean():.2f}", f"{sm.std():.2f}", f"{sm.std()/np.sqrt(n_lg):.2f}", f"{abs(sm.mean() - cal_pop_mean):.2f}"])
        print(tabulate(large_sample_res, headers=["Sample Size (n)", "Sample Mean", "Sample Std", "SE", "Abs Error"], tablefmt="grid"))
    except Exception as e:
        pass
    
    # 2. Hypothesis Testing: Independent Two-Sample t-Test
    # Test 1: Male vs Female Mean Calories Burned
    male_cals = df[df["Gender"] == "Male"][target]
    female_cals = df[df["Gender"] == "Female"][target]
    
    t_stat_gender, p_val_gender = stats.ttest_ind(male_cals, female_cals, equal_var=False)
    
    # Calculate Cohen's d effect size
    pooled_std_g = np.sqrt(((len(male_cals) - 1) * male_cals.var() + (len(female_cals) - 1) * female_cals.var()) / (len(male_cals) + len(female_cals) - 2))
    cohen_d_g = (male_cals.mean() - female_cals.mean()) / pooled_std_g
    
    print("\n[3] Hypothesis Test 1: Gender Comparison in Caloric Expenditure")
    print("    - Null Hypothesis (H0)        : mu_Male = mu_Female (No significant difference)")
    print("    - Alternative Hypothesis (H1) : mu_Male != mu_Female (Significant difference exists)")
    print(f"    - Male Sample Mean (N={len(male_cals)})   : {male_cals.mean():.2f} kcal (Std: {male_cals.std():.2f})")
    print(f"    - Female Sample Mean (N={len(female_cals)}) : {female_cals.mean():.2f} kcal (Std: {female_cals.std():.2f})")
    print(f"    - Welch's t-statistic         : {t_stat_gender:.4f}")
    print(f"    - Two-Tailed p-value          : {p_val_gender:.4e}")
    print(f"    - Cohen's d Effect Size       : {cohen_d_g:.4f}")
    
    alpha = 0.05
    if p_val_gender < alpha:
        conclusion_g = f"REJECT H0 at alpha = {alpha}. Statistically significant difference between genders."
    else:
        conclusion_g = f"FAIL TO REJECT H0 at alpha = {alpha}. No statistically significant difference detected."
    print(f"    - Statistical Decision        : {conclusion_g}\n")
    
    # Test 2: HIIT vs Yoga Mean Calories Burned
    hiit_cals = df[df["Workout_Type"] == "HIIT"][target]
    yoga_cals = df[df["Workout_Type"] == "Yoga"][target]
    t_stat_wy, p_val_wy = stats.ttest_ind(hiit_cals, yoga_cals, equal_var=False)
    
    print("[4] Hypothesis Test 2: HIIT vs. Yoga Caloric Expenditure")
    print("    - Null Hypothesis (H0)        : mu_HIIT = mu_Yoga")
    print("    - Alternative Hypothesis (H1) : mu_HIIT != mu_Yoga")
    print(f"    - HIIT Mean (N={len(hiit_cals)})         : {hiit_cals.mean():.2f} kcal")
    print(f"    - Yoga Mean (N={len(yoga_cals)})         : {yoga_cals.mean():.2f} kcal")
    print(f"    - Welch's t-statistic         : {t_stat_wy:.4f}")
    print(f"    - Two-Tailed p-value          : {p_val_wy:.4e}")
    decision_wy = "REJECT H0 (Significant difference)" if p_val_wy < alpha else "FAIL TO REJECT H0 (Difference not significant)"
    print(f"    - Statistical Decision        : {decision_wy}\n")
    
    # Visualization: Hypothesis distributions & t-test critical regions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EXP 6: Sampling & Hypothesis Testing Diagnostics", fontsize=15, weight="bold")
    
    # Subplot 1: Gender Comparison Violin/KDE
    sns.violinplot(data=df, x="Gender", y=target, hue="Gender", palette=["#3498db", "#e74c3c"], legend=False, ax=axes[0], inner="quartile")
    axes[0].set_title(f"Calorie Expenditure by Gender (p = {p_val_gender:.3f})")
    axes[0].set_ylabel("Calories Burned (kcal)")
    
    # Subplot 2: Sampling Distribution of the Mean across iterations
    sample_means_30 = [df.sample(30)[target].mean() for _ in range(500)]
    sample_means_150 = [df.sample(150)[target].mean() for _ in range(500)]
    
    sns.kdeplot(sample_means_30, label="Sample Size n = 30", color="#e67e22", ax=axes[1], linewidth=2)
    sns.kdeplot(sample_means_150, label="Sample Size n = 150", color="#27ae60", ax=axes[1], linewidth=2)
    axes[1].axvline(pop_mean, color="#c0392b", linestyle="--", label=f"True Pop Mean ({pop_mean:.1f})")
    axes[1].set_title("Sampling Distribution of the Mean (CLT Demonstration)")
    axes[1].set_xlabel("Sample Mean Calories Burned")
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "exp6_hypothesis_testing.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[5] Hypothesis Testing & Sampling plots saved to: {plot_path}")
    return {
        "pop_mean": pop_mean,
        "sampling_results": sampling_results,
        "t_stat_gender": t_stat_gender,
        "p_val_gender": p_val_gender,
        "plot_path": plot_path
    }

# ==============================================================================
# EXP 7 : NUMPY FUNDAMENTALS
# ==============================================================================
def run_exp7_numpy_fundamentals(df: pd.DataFrame) -> dict:
    """
    EXP 7 : NumPy Fundamentals
    1. Use aggregation functions and practice indexing and sorting with NumPy arrays.
    2. Vectorized computation of workout intensity and energy expenditure metrics.
    """
    print_section_header("EXP 7 : NumPy Array Fundamentals, Aggregations & Slicing")
    
    # Extract numerical data as 1D and 2D NumPy arrays
    calories_np = df["Calories_Burned"].to_numpy()
    duration_np = df["Session_Duration (hours)"].to_numpy()
    bpm_np = df["Avg_BPM"].to_numpy()
    weight_np = df["Weight (kg)"].to_numpy()
    
    # 2D feature matrix (N x 4)
    feature_matrix = np.column_stack((duration_np, bpm_np, weight_np, calories_np))
    
    print(f"[1] NumPy Array Architecture:")
    print(f"    - Calories 1D Array Shape : {calories_np.shape}, Dtype: {calories_np.dtype}")
    print(f"    - Feature 2D Matrix Shape : {feature_matrix.shape}, Dtype: {feature_matrix.dtype}\n")
    
    # 1. Aggregations using NumPy
    aggs = [
        ["Count (np.size)", f"{np.size(calories_np)}"],
        ["Sum (np.sum)", f"{np.sum(calories_np):,.2f} kcal"],
        ["Mean (np.mean)", f"{np.mean(calories_np):.2f} kcal"],
        ["Variance (np.var)", f"{np.var(calories_np):.2f}"],
        ["Std Deviation (np.std)", f"{np.std(calories_np):.2f} kcal"],
        ["Minimum (np.min)", f"{np.min(calories_np):.2f} kcal"],
        ["25th Percentile (np.percentile)", f"{np.percentile(calories_np, 25):.2f} kcal"],
        ["50th Percentile / Median (np.median)", f"{np.median(calories_np):.2f} kcal"],
        ["75th Percentile (np.percentile)", f"{np.percentile(calories_np, 75):.2f} kcal"],
        ["Maximum (np.max)", f"{np.max(calories_np):.2f} kcal"]
    ]
    print("[2] NumPy Universal Aggregation Functions:")
    print(tabulate(aggs, headers=["Aggregation Function", "Computed Value"], tablefmt="grid"))
    
    # 2. Advanced Indexing: Boolean Masking & Multi-dimensional Slicing
    # Condition: Sessions with Duration > 1.5 hours AND Avg BPM > 150
    high_intensity_mask = (duration_np > 1.5) & (bpm_np > 150)
    high_intensity_calories = calories_np[high_intensity_mask]
    
    print("\n[3] NumPy Indexing & Slicing:")
    print(f"    - Total High Intensity Sessions (Duration > 1.5h & BPM > 150): {np.sum(high_intensity_mask)}")
    print(f"    - Mean Calories in High Intensity Sub-array: {np.mean(high_intensity_calories):.2f} kcal")
    print(f"    - Mean Calories in Normal Sessions: {np.mean(calories_np[~high_intensity_mask]):.2f} kcal")
    
    # Practice Slicing: First 5 rows and first 3 columns of 2D matrix
    sliced_matrix = feature_matrix[:5, :3]
    print(f"\n    - Matrix 2D Slice [0:5, 0:3] (Duration, BPM, Weight for first 5 records):")
    print(sliced_matrix)
    
    # 3. Sorting with NumPy
    # Sort calories and get index order
    sorted_calories = np.sort(calories_np)
    sort_indices = np.argsort(calories_np)
    
    print("\n[4] NumPy Sorting Demonstrations:")
    print(f"    - Top 5 lowest calorie burns (np.sort): {sorted_calories[:5]}")
    print(f"    - Top 5 highest calorie burns (np.sort): {sorted_calories[-5:]}")
    print(f"    - Indices of top 3 peak calorie burns (np.argsort): {sort_indices[-3:]}")
    
    # Vectorized MET Energy Expenditure Rate (kcal/hour)
    burn_rate_per_hour = calories_np / duration_np
    print(f"\n[5] Vectorized Array Computation:")
    print(f"    - Mean Energy Expenditure Rate: {np.mean(burn_rate_per_hour):.2f} kcal/hour")
    print(f"    - Max Energy Expenditure Rate : {np.max(burn_rate_per_hour):.2f} kcal/hour")
    
    return {
        "mean": np.mean(calories_np),
        "std": np.std(calories_np),
        "high_intensity_count": int(np.sum(high_intensity_mask))
    }

# ==============================================================================
# EXP 8 : ADVANCED DATA MANIPULATION AND VISUALIZATION
# ==============================================================================
def run_exp8_advanced_manipulation_visualization(df: pd.DataFrame, plots_dir: str = PLOTS_DIR) -> dict:
    """
    EXP 8 : Advanced Data Manipulation and Visualization
    1. Manage missing data and demonstrate hierarchical indexing with pandas.
    2. Craft detailed scatter plots and 3D visualizations with Matplotlib.
    """
    print_section_header("EXP 8 : Advanced Data Manipulation & 3D Visualization")
    
    # 1. Missing Data Simulation & Imputation Management
    print("[1] Missing Data Management Simulation:")
    df_impute_demo = df.copy()
    
    # Introduce 10% realistic missing values randomly into Fat_Percentage and Water_Intake
    np.random.seed(101)
    mask_fat = np.random.rand(len(df_impute_demo)) < 0.10
    mask_water = np.random.rand(len(df_impute_demo)) < 0.08
    
    df_impute_demo.loc[mask_fat, "Fat_Percentage"] = np.nan
    df_impute_demo.loc[mask_water, "Water_Intake (liters)"] = np.nan
    
    missing_counts = df_impute_demo.isnull().sum()
    print("    - Simulated Missing Values:")
    print(f"      * Fat_Percentage missing: {missing_counts['Fat_Percentage']} rows ({missing_counts['Fat_Percentage']/len(df)*100:.1f}%)")
    print(f"      * Water_Intake missing  : {missing_counts['Water_Intake (liters)']} rows ({missing_counts['Water_Intake (liters)']/len(df)*100:.1f}%)")
    
    # Method A: Listwise Deletion (dropna)
    df_dropped = df_impute_demo.dropna()
    print(f"    - Listwise Deletion (dropna): Retains {len(df_dropped)} rows (Dropped {len(df_impute_demo) - len(df_dropped)})")
    
    # Method B: Median Imputation
    df_imputed = df_impute_demo.copy()
    fat_median = df_imputed["Fat_Percentage"].median()
    water_median = df_imputed["Water_Intake (liters)"].median()
    df_imputed["Fat_Percentage"] = df_imputed["Fat_Percentage"].fillna(fat_median)
    df_imputed["Water_Intake (liters)"] = df_imputed["Water_Intake (liters)"].fillna(water_median)
    print(f"    - Median Imputation Applied: Fat % median = {fat_median:.1f}%, Water median = {water_median:.1f}L")
    print(f"      * Remaining nulls after imputation: {df_imputed.isnull().sum().sum()}\n")
    
    # 2. Hierarchical Indexing (MultiIndex) with pandas
    print("[2] Hierarchical Indexing (MultiIndex) Demonstration:")
    hierarchical_df = df.groupby(["Workout_Type", "Gender"])[
        ["Calories_Burned", "Session_Duration (hours)", "Avg_BPM"]
    ].agg(["mean", "std", "count"])
    
    print("    - MultiIndex Summary Table (Grouped by Workout_Type & Gender):")
    print(hierarchical_df.round(2))
    
    # MultiIndex Slicing with .xs (Cross-Section)
    hiit_summary = hierarchical_df.xs("HIIT", level="Workout_Type")
    print("\n    - Cross-Section Slicing (.xs('HIIT', level='Workout_Type')):")
    print(hiit_summary.round(2))
    
    # Reshaping: Unstacking
    unstacked = hierarchical_df["Calories_Burned"]["mean"].unstack()
    print("\n    - Reshaped Pivot Table (.unstack() on Calories Mean):")
    print(unstacked.round(2))
    
    # 3. Visualizations: Detailed 2D Interaction Plots & 3D Matplotlib Visualization
    # Plot A: Detailed Multi-Panel Interaction Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("EXP 8: Detailed Interaction & Hierarchical Sub-cohorts", fontsize=15, weight="bold")
    
    sns.scatterplot(
        data=df,
        x="Avg_BPM",
        y="Calories_Burned",
        hue="Workout_Type",
        size="Session_Duration (hours)",
        sizes=(30, 200),
        palette="turbo",
        alpha=0.75,
        ax=axes[0]
    )
    axes[0].set_title("Interaction: Avg BPM, Duration & Workout Type on Calories")
    axes[0].set_xlabel("Average Heart Rate (BPM)")
    axes[0].set_ylabel("Calories Burned (kcal)")
    
    unstacked.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#3498db"], width=0.6)
    axes[1].set_title("Hierarchical MultiIndex: Mean Calories by Workout & Gender")
    axes[1].set_xlabel("Workout Type")
    axes[1].set_ylabel("Mean Calories Burned (kcal)")
    axes[1].legend(title="Gender")
    
    plt.tight_layout()
    plot_2d_path = os.path.join(plots_dir, "exp8_hierarchical_analysis.png")
    plt.savefig(plot_2d_path, dpi=300)
    plt.close()
    
    # Plot B: True 3D Visualization with Matplotlib (Duration, Avg_BPM, Calories + Regression Surface)
    fig_3d = plt.figure(figsize=(12, 9))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    
    x_dur = df["Session_Duration (hours)"].values
    y_bpm = df["Avg_BPM"].values
    z_cal = df["Calories_Burned"].values
    
    # Fit 2-variable linear surface
    X_surf = np.column_stack((x_dur, y_bpm))
    lr_3d = LinearRegression().fit(X_surf, z_cal)
    
    # Create grid for regression plane
    x_grid = np.linspace(x_dur.min(), x_dur.max(), 30)
    y_grid = np.linspace(y_bpm.min(), y_bpm.max(), 30)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    Z_mesh = lr_3d.predict(np.column_stack((X_mesh.ravel(), Y_mesh.ravel()))).reshape(X_mesh.shape)
    
    # Plot regression plane
    surf = ax_3d.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.45, cmap="viridis", edgecolor="none")
    
    # Scatter observed points
    scatter = ax_3d.scatter(
        x_dur,
        y_bpm,
        z_cal,
        c=z_cal,
        cmap="coolwarm",
        s=45,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5
    )
    
    ax_3d.set_title("EXP 8: 3D Visualization of Calorie Burn Surface with OLS Plane", fontsize=14, weight="bold")
    ax_3d.set_xlabel("Session Duration (hours)", labelpad=10)
    ax_3d.set_ylabel("Average Heart Rate (BPM)", labelpad=10)
    ax_3d.set_zlabel("Calories Burned (kcal)", labelpad=10)
    fig_3d.colorbar(scatter, ax=ax_3d, shrink=0.55, aspect=12, label="Calories Burned (kcal)")
    
    # Adjust viewing perspective
    ax_3d.view_init(elev=22, azim=130)
    
    plot_3d_path = os.path.join(plots_dir, "exp8_3d_regression_surface.png")
    plt.savefig(plot_3d_path, dpi=300)
    plt.close()
    
    print(f"\n[3] 2D Hierarchical Plots saved to: {plot_2d_path}")
    print(f"[4] 3D Surface Visualization saved to: {plot_3d_path}")
    
    return {
        "hierarchical_summary": hierarchical_df,
        "plot_2d_path": plot_2d_path,
        "plot_3d_path": plot_3d_path
    }

# ==============================================================================
# MASTER RUNNER FOR ALL EXPERIMENTS
# ==============================================================================
def run_all_experiments(df: pd.DataFrame = None):
    """
    Executes all 8 DAP Lab Experiments sequentially with full console tables and saved artifacts.
    """
    try:
        from src.data_loader import load_gym_dataset
    except ModuleNotFoundError:
        from data_loader import load_gym_dataset
    if df is None:
        df = load_gym_dataset()
    
    print("\n" + "#" * 80)
    print(" EXECUTING DATA ANALYSIS WITH PYTHON (DAP) LAB EXPERIMENTS 1 TO 8 ".center(80, "#"))
    print("#" * 80)
    
    results = {}
    results["exp1"] = run_exp1_data_handling(df)
    results["exp2"] = run_exp2_central_tendency_variability(df)
    results["exp3"] = run_exp3_transformations_weighted_averages(df)
    results["exp4"] = run_exp4_correlation_analysis(df)
    results["exp5"] = run_exp5_regression(df)
    results["exp6"] = run_exp6_sampling_hypothesis_testing(df)
    results["exp7"] = run_exp7_numpy_fundamentals(df)
    results["exp8"] = run_exp8_advanced_manipulation_visualization(df)
    
    print("\n" + "=" * 80)
    print(" ALL 8 DAP LAB EXPERIMENTS SUCCESSFULLY EXECUTED AND ARTIFACTS SAVED! ".center(80, "="))
    print("=" * 80 + "\n")
    return results

if __name__ == "__main__":
    try:
        from src.data_loader import load_gym_dataset
    except ModuleNotFoundError:
        from data_loader import load_gym_dataset
    df = load_gym_dataset()
    run_all_experiments(df)
