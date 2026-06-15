"""
feature_importance.py

Utilities for:

1. Logistic Regression coefficients
2. Linear SVM coefficients
3. Random Forest SHAP
4. XGBoost SHAP
5. KNN Permutation Importance
6. Cross-model comparison plots

Author: Team Project
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import shap

from sklearn.inspection import permutation_importance


# ==========================================================
# Helper
# ==========================================================

def _save_dataframe(df, output_dir, filename):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    df.to_csv(path, index=False)

    return path


# ==========================================================
# Logistic Regression
# ==========================================================

def logistic_regression_importance(
        model,
        feature_names,
        output_dir):

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })

    coef_df["AbsCoefficient"] = np.abs(
        coef_df["Coefficient"]
    )

    coef_df = coef_df.sort_values(
        "AbsCoefficient",
        ascending=False
    )

    _save_dataframe(
        coef_df,
        output_dir,
        "lr_feature_importance.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=coef_df,
        x="Coefficient",
        y="Feature"
    )

    plt.title(
        "Logistic Regression Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / "lr_feature_importance.png",
        dpi=300
    )

    plt.close()

    return coef_df


# ==========================================================
# Linear SVM
# ==========================================================

def svm_importance(
        model,
        feature_names,
        output_dir):

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })

    coef_df["AbsCoefficient"] = np.abs(
        coef_df["Coefficient"]
    )

    coef_df = coef_df.sort_values(
        "AbsCoefficient",
        ascending=False
    )

    _save_dataframe(
        coef_df,
        output_dir,
        "svm_feature_importance.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=coef_df,
        x="Coefficient",
        y="Feature"
    )

    plt.title(
        "Linear SVM Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / "svm_feature_importance.png",
        dpi=300
    )

    plt.close()

    return coef_df


# ==========================================================
# Random Forest SHAP
# ==========================================================

def random_forest_shap(
        model,
        X_sample,
        feature_names,
        output_dir):

    explainer = shap.TreeExplainer(model)

    X_sample_df = pd.DataFrame(X_sample, columns = feature_names)

    X_sampled_subset = X_sample_df.sample(100, random_state=42)

    shap_values = explainer.shap_values(
        X_sampled_subset
    )

    # if isinstance(shap_values, list):
    #     shap_values = shap_values[1]

    # if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    #     # Shape is (samples, features, classes) -> extract class 1
    #     shap_values = shap_values[:, :, 1]
    # elif isinstance(shap_values, list):
    #     shap_values = shap_values[1]

    # 1. Safely handle the 3D numpy array by pulling out Class 1
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape is (samples, features, classes) -> slice to get (samples, features)
        shap_values = shap_values[:, :, 1]
    elif isinstance(shap_values, list):
        # Fallback for older SHAP versions returning a list of arrays
        shap_values = shap_values[1]
    else:
        # Fallback for regression models
        shap_values = shap_values

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance":
            np.abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    _save_dataframe(
        importance,
        output_dir,
        "rf_feature_importance.csv"
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sampled_subset,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / "rf_shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return importance


# ==========================================================
# XGBoost SHAP
# ==========================================================

def xgboost_shap(
        model,
        X_sample,
        feature_names,
        output_dir):

    explainer = shap.TreeExplainer(model)

    X_sample_df = pd.DataFrame(X_sample, columns = feature_names)

    X_sampled_subset = X_sample_df.sample(100, random_state=42)

    shap_values = explainer.shap_values(
        X_sampled_subset
    )

    # if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    #     # Shape is (samples, features, classes) -> extract class 1
    #     shap_values = shap_values[:, :, 1]
    # elif isinstance(shap_values, list):
    #     shap_values = shap_values[1]

    # 1. Safely handle the 3D numpy array by pulling out Class 1
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape is (samples, features, classes) -> slice to get (samples, features)
        shap_values = shap_values[:, :, 1]
    elif isinstance(shap_values, list):
        # Fallback for older SHAP versions returning a list of arrays
        shap_values = shap_values[1]
    else:
        # Fallback for regression models
        shap_values = shap_values

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance":
            np.abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    _save_dataframe(
        importance,
        output_dir,
        "xgb_feature_importance.csv"
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sampled_subset,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / "xgb_shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return importance


# ==========================================================
# KNN Permutation Importance
# ==========================================================

def knn_permutation_importance(
        model,
        X_test,
        y_test,
        feature_names,
        output_dir):

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance":
            perm.importances_mean
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    _save_dataframe(
        importance,
        output_dir,
        "knn_feature_importance.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "KNN Permutation Importance"
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / "knn_feature_importance.png",
        dpi=300
    )

    plt.close()

    return importance


# ==========================================================
# Cross Model Comparison
# ==========================================================

def comparison_plot(
        feature_names,
        rf_df,
        xgb_df,
        lr_df,
        svm_df,
        knn_df,
        output_dir):

    # --- INTERNAL HELPER TO SAFE-EXTRACT MAPPINGS ---
    def get_feature_map(df, value_column):
        if df is None or df.empty:
            return {}
        
        # 1. Detect if 'Feature' is the index or a column
        if "Feature" in df.columns:
            series = df.set_index("Feature")[value_column]
        elif df.index.name == "Feature" or df.index.dtype == "object":
            series = df[value_column]
        else:
            return {}
        
        # 2. Convert to dict and strip whitespace from keys to prevent bugs
        return {str(k).strip(): v for k, v in series.to_dict().items()}

    # Extract clean dictionaries for each model
    rf_map  = get_feature_map(rf_df, "Importance")
    xgb_map = get_feature_map(xgb_df, "Importance")
    lr_map  = get_feature_map(lr_df, "AbsCoefficient")
    svm_map = get_feature_map(svm_df, "AbsCoefficient")
    knn_map = get_feature_map(knn_df, "Importance")

    # --- DEBUG PRINT (Temporary: checks if maps are actually empty) ---
    print("--- Diagnostics ---")
    print(f"Master features looking for: {feature_names}")
    print(f"RF found features: {list(rf_map.keys())}")
    print("-------------------")

    # 3. Build the comparison table safely by explicit lookup
    clean_features = [str(f).strip() for f in feature_names]
    
    comparison = pd.DataFrame({
        "Feature": feature_names,
        "RF":  [rf_map.get(f, np.nan)  for f in clean_features],
        "XGB": [xgb_map.get(f, np.nan) for f in clean_features],
        "LR":  [lr_map.get(f, np.nan)  for f in clean_features],
        "SVM": [svm_map.get(f, np.nan) for f in clean_features],
        "KNN": [knn_map.get(f, np.nan) for f in clean_features]
    })

    # 4. Normalize columns against their maximum value
    for col in comparison.columns[1:]:
        max_val = comparison[col].dropna().abs().max()
        if pd.notna(max_val) and max_val > 0:
            comparison[col] = comparison[col] / max_val
        else:
            comparison[col] = 0.0  # Fill with 0 if column is entirely empty/NaN

    # 5. Save the output
    output_path = Path(output_dir) / "all_models_feature_importance.csv"
    comparison.to_csv(output_path, index=False)

    # 6. Generate Plot
    plt.figure(figsize=(12, 6))
    comparison.set_index("Feature").plot(kind="bar", figsize=(12, 6))
    plt.title("Normalized Feature Importance Across Models")
    plt.ylabel("Normalized Importance")
    plt.tight_layout()
    
    plt.savefig(
        Path(output_dir) / "all_models_feature_importance.png",
        dpi=300
    )
    plt.close()

    return comparison


# ==========================================================
# Master Function
# ==========================================================

def generate_all_feature_importance(
        trained_models,
        X_test,
        y_test,
        feature_names,
        output_dir):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X_sample = pd.DataFrame(
        X_test,
        columns=feature_names
    ).sample(
        min(3000, len(X_test)),
        random_state=42
    )

    lr_df = logistic_regression_importance(
        trained_models["Logistic Regression"],
        feature_names,
        output_dir
    )

    svm_df = svm_importance(
        trained_models["SVM"],
        feature_names,
        output_dir
    )

    rf_df = random_forest_shap(
        trained_models["Random Forest"],
        X_sample,
        feature_names,
        output_dir
    )

    xgb_df = xgboost_shap(
        trained_models["XGBoost"],
        X_sample,
        feature_names,
        output_dir
    )

    knn_df = knn_permutation_importance(
        trained_models["KNN"],
        X_test,
        y_test,
        feature_names,
        output_dir
    )

    comparison_plot(
        feature_names,
        rf_df,
        xgb_df,
        lr_df,
        svm_df,
        knn_df,
        output_dir
    )

def generate_fold_fi_for_lopo(
        model_name,
        trained_model,
        X_test,
        y_test,
        feature_names,
        output_dir):
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X_sample = pd.DataFrame(
        X_test,
        columns=feature_names
    ).sample(
        min(3000, len(X_test)),
        random_state=42
    )

    if model_name == "Logistic Regression":

        imp_df = logistic_regression_importance(
            trained_model,
            feature_names,
            output_dir
        )

    if model_name == "SVM":

        imp_df = svm_importance(
            trained_model,
            feature_names,
            output_dir
        )

    if model_name == "Random Forest":

        imp_df = random_forest_shap(
            trained_model,
            X_sample,
            feature_names,
            output_dir
        )

    if model_name == "XGBoost":

        imp_df = xgboost_shap(
            trained_model,
            X_sample,
            feature_names,
            output_dir
        )

    if model_name == "KNN":

        imp_df = knn_permutation_importance(
            trained_model,
            X_test,
            y_test,
            feature_names,
            output_dir
        )

    if "AbsCoefficient" in imp_df.columns:
        imp_df.rename(columns={"AbsCoefficient": "Importance"}, inplace=True)

    return imp_df