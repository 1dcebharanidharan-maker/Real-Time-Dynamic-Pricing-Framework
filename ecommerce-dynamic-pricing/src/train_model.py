"""
Machine Learning Training Pipeline.
Trains, evaluates, and compares Linear Regression, Random Forest, and Gradient Boosting models.
Generates evaluation charts, serializes the best performing model pipeline, and exports metadata.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.preprocessing import DataPreprocessor
from src.pricing_engine import PricingPipeline

# Set clean styling for matplotlib
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def train_and_evaluate_models():
    print("[INFO] Loading and cleaning dataset for model training...")
    df_raw = pd.read_csv(config.DATASET_PATH)
    preprocessor = DataPreprocessor(random_state=config.RANDOM_STATE)
    df_clean, audit = preprocessor.clean_dataset(df_raw)

    X_train, X_test, y_train, y_test, train_df, test_df = preprocessor.prepare_modeling_data(df_clean)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.07,
            max_depth=5,
            subsample=0.85,
            random_state=config.RANDOM_STATE
        )
    }

    results = {}
    predictions = {}
    best_score = -float("inf")
    best_model_name = ""
    best_model_obj = None

    print("\n--- MODEL TRAINING AND EVALUATION ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        predictions[name] = y_pred

        # Metrics
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        results[name] = {
            "Test_MAE": round(float(test_mae), 3),
            "Test_RMSE": round(float(test_rmse), 3),
            "Test_R2": round(float(test_r2), 4),
            "Train_MAE": round(float(train_mae), 3),
            "Train_RMSE": round(float(train_rmse), 3),
            "Train_R2": round(float(train_r2), 4)
        }

        print(f"  [{name}] Test MAE: ${test_mae:.2f} | RMSE: ${test_rmse:.2f} | R2: {test_r2:.4f}")

        if test_r2 > best_score:
            best_score = test_r2
            best_model_name = name
            best_model_obj = model

    print(f"\n[BEST MODEL SELECTED] {best_model_name} with Test R2 = {best_score:.4f}")

    # =========================================================================
    # GENERATE EVALUATION CHARTS
    # =========================================================================
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model Comparison Chart
    plt.figure(figsize=(10, 5))
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    
    plt.subplot(1, 3, 1)
    sns.barplot(data=metrics_df, x="Model", y="Test_MAE", hue="Model", palette="Blues_d", legend=False)
    plt.title("Mean Absolute Error (Lower is Better)")
    plt.ylabel("MAE ($)")
    plt.xticks(rotation=25)

    plt.subplot(1, 3, 2)
    sns.barplot(data=metrics_df, x="Model", y="Test_RMSE", hue="Model", palette="Oranges_d", legend=False)
    plt.title("Root Mean Squared Error (Lower is Better)")
    plt.ylabel("RMSE ($)")
    plt.xticks(rotation=25)

    plt.subplot(1, 3, 3)
    sns.barplot(data=metrics_df, x="Model", y="Test_R2", hue="Model", palette="Greens_d", legend=False)
    plt.title("R² Score (Higher is Better)")
    plt.ylabel("R²")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "ml_model_comparison.png", dpi=300)
    plt.close()

    # 2. Actual vs Predicted Chart (Best Model)
    plt.figure(figsize=(8, 6))
    best_preds = predictions[best_model_name]
    plt.scatter(y_test, best_preds, alpha=0.4, color="#1f77b4", edgecolors="none", s=25, label="Test Samples")
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Parity (y=x)")
    plt.title(f"Actual vs. Predicted Recommended Price ({best_model_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Policy Target Price ($)")
    plt.ylabel("Model Predicted Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "ml_actual_vs_predicted.png", dpi=300)
    plt.close()

    # 3. Residual Distribution & Residuals vs Predicted
    residuals = y_test - best_preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(residuals, kde=True, ax=axes[0], color="#2ca02c", bins=30)
    axes[0].set_title("Residual Error Distribution")
    axes[0].set_xlabel("Prediction Error (Actual - Predicted) ($)")
    axes[0].axvline(0, color="red", linestyle="--")

    axes[1].scatter(best_preds, residuals, alpha=0.4, color="#9467bd", s=25)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals vs. Predicted Values (Homoscedasticity Check)")
    axes[1].set_xlabel("Predicted Price ($)")
    axes[1].set_ylabel("Residual ($)")

    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "ml_residuals_plot.png", dpi=300)
    plt.close()

    # 4. Feature Importance Plot
    if hasattr(best_model_obj, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        importances = best_model_obj.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_names = [preprocessor.feature_names[i] for i in indices]
        top_vals = importances[indices]

        sns.barplot(x=top_vals, y=top_names, hue=top_names, palette="viridis", legend=False)
        plt.title(f"Top 15 Feature Importances ({best_model_name})", fontsize=12, fontweight="bold")
        plt.xlabel("Relative Importance Score")
        plt.tight_layout()
        plt.savefig(config.OUTPUTS_DIR / "ml_feature_importance.png", dpi=300)
        plt.close()

    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    pipeline = PricingPipeline(preprocessor, best_model_obj)
    joblib.dump(pipeline, config.MODEL_PATH)
    print(f"[SUCCESS] Trained pipeline serialized to: {config.MODEL_PATH}")

    # Export feature metadata and benchmark metrics
    metadata = {
        "best_model": best_model_name,
        "results": results,
        "feature_count": len(preprocessor.feature_names),
        "engineered_numericals": preprocessor.engineered_numerical_cols,
        "categorical_features": preprocessor.categorical_cols,
        "test_sample_count": len(y_test),
        "train_sample_count": len(y_train),
        "methodology_disclaimer": (
            "Empirical verification note: High predictive accuracy against historical/simulated "
            "pricing policies demonstrates model fidelity in capturing encoded business constraints, "
            "but does not replace real-world randomized A/B testing or bandit experiments for long-term profit optimality."
        )
    }

    with open(config.FEATURE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUCCESS] Metadata exported to: {config.FEATURE_METADATA_PATH}")

    return pipeline, metadata, test_df

if __name__ == "__main__":
    train_and_evaluate_models()
