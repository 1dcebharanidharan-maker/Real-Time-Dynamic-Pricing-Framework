"""
Data Preprocessing and Feature Engineering Pipeline.
Handles deduplication, missing value imputation, IQR outlier capping,
feature engineering, target leakage prevention, scaling, and categorical encoding.
"""

from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

class DataPreprocessor:
    """
    Modular preprocessing pipeline ensuring robust feature engineering
    and strict prevention of target leakage.
    """

    def __init__(self, random_state: int = config.RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.iqr_bounds: Dict[str, Tuple[float, float]] = {}
        self.imputation_values: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.engineered_numerical_cols: List[str] = []
        self.categorical_cols: List[str] = ["category", "brand"]

    def clean_dataset(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Step-by-step cleaning: deduplication, missing value imputation, and IQR outlier capping.
        Returns cleaned DataFrame and statistical audit summary.
        """
        audit: Dict[str, Any] = {}
        audit["raw_row_count"] = len(df_raw)
        audit["raw_null_counts"] = df_raw.isnull().sum().to_dict()

        # 1. Deduplication
        duplicates_count = int(df_raw.duplicated().sum())
        audit["duplicates_removed"] = duplicates_count
        df = df_raw.drop_duplicates().copy()

        # 2. Missing Value Imputation
        # Categorical: brand
        df["brand"] = df["brand"].fillna("Generic")
        self.imputation_values["brand"] = "Generic"

        # Numerical: rating (impute by category median)
        rating_medians = df.groupby("category")["product_rating"].transform("median")
        df["product_rating"] = df["product_rating"].fillna(rating_medians).fillna(4.0)

        # Numerical: historical_sales (impute by category median)
        sales_medians = df.groupby("category")["historical_sales"].transform("median")
        df["historical_sales"] = df["historical_sales"].fillna(sales_medians).fillna(50)

        # Numerical: competitor_3_price (impute with average of competitor 1 and 2 or our current price)
        fallback_comp = (df["competitor_1_price"] + df["competitor_2_price"]) / 2.0
        df["competitor_3_price"] = df["competitor_3_price"].fillna(fallback_comp).fillna(df["our_current_price"])

        # 3. IQR Outlier Detection and Capping (Winsorization)
        # Cap extreme spikes in inventory and competitor prices
        outlier_cols = ["inventory_level", "competitor_1_price", "competitor_2_price", "competitor_3_price"]
        for col in outlier_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = max(0.0, q1 - 1.5 * iqr)
            upper_bound = q3 + 1.5 * iqr
            self.iqr_bounds[col] = (float(lower_bound), float(upper_bound))
            
            outliers_detected = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
            audit[f"outliers_capped_{col}"] = outliers_detected
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        audit["clean_row_count"] = len(df)
        audit["clean_null_counts"] = df.isnull().sum().to_dict()
        return df, audit

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates economically meaningful dynamic pricing features without target leakage.
        """
        df = df.copy()

        # Competitor Aggregate Metrics
        df["min_competitor_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].min(axis=1)
        df["avg_competitor_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].mean(axis=1)
        df["max_competitor_price"] = df[["competitor_1_price", "competitor_2_price", "competitor_3_price"]].max(axis=1)
        df["competitor_dispersion"] = df["max_competitor_price"] - df["min_competitor_price"]

        # Financial & Relative Competitiveness Indices
        df["price_to_cost_ratio"] = df["our_current_price"] / np.maximum(df["cost_price"], 0.01)
        df["competitor_price_ratio"] = df["our_current_price"] / np.maximum(df["avg_competitor_price"], 0.01)

        # Scarcity Pressure Metric (Demand relative to current inventory stock)
        df["stock_pressure_index"] = df["demand_score"] / (df["inventory_level"] + 1.0)

        # Quality Elasticity Metric
        df["rating_advantage"] = df["product_rating"] - 3.5

        return df

    def prepare_modeling_data(
        self,
        df: pd.DataFrame,
        test_size: float = config.TEST_SIZE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Prepares training and testing matrices with strict target leakage isolation.
        """
        # Ensure timestamp sorting if temporal split is needed
        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp").reset_index(drop=True)

        # 1. Feature Engineering
        df_featured = self.engineer_features(df)

        # 2. Strict Target Leakage Prevention:
        # Target variable is recommended_price.
        # Exclude product identifiers, timestamps, and target variable from input features.
        y = df_featured[config.TARGET_VARIABLE].values

        self.engineered_numerical_cols = [
            "cost_price",
            "our_current_price",
            "competitor_1_price",
            "competitor_2_price",
            "competitor_3_price",
            "min_competitor_price",
            "avg_competitor_price",
            "max_competitor_price",
            "competitor_dispersion",
            "historical_sales",
            "demand_score",
            "inventory_level",
            "product_rating",
            "discount_percentage",
            "seasonality_index",
            "competitor_price_change",
            "price_to_cost_ratio",
            "competitor_price_ratio",
            "stock_pressure_index",
            "rating_advantage"
        ]

        # 3. Train-Test Split (Chronological 80/20 split to respect time dynamics)
        split_idx = int(len(df_featured) * (1.0 - test_size))
        train_df = df_featured.iloc[:split_idx].copy()
        test_df = df_featured.iloc[split_idx:].copy()

        y_train = y[:split_idx]
        y_test = y[split_idx:]

        # 4. Fit Preprocessors on Training Set ONLY
        # Categorical Encoding
        cat_train_encoded = self.encoder.fit_transform(train_df[self.categorical_cols])
        cat_test_encoded = self.encoder.transform(test_df[self.categorical_cols])
        cat_feature_names = list(self.encoder.get_feature_names_out(self.categorical_cols))

        # Numerical Scaling
        num_train_scaled = self.scaler.fit_transform(train_df[self.engineered_numerical_cols])
        num_test_scaled = self.scaler.transform(test_df[self.engineered_numerical_cols])

        # Concatenate into full feature arrays
        X_train = np.hstack([num_train_scaled, cat_train_encoded])
        X_test = np.hstack([num_test_scaled, cat_test_encoded])

        self.feature_names = self.engineered_numerical_cols + cat_feature_names

        return X_train, X_test, y_train, y_test, train_df, test_df

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single product record into scaled feature vector for online inference.
        """
        df_single = pd.DataFrame([input_dict])
        
        # Fill missing categoricals
        if "brand" not in df_single or pd.isnull(df_single["brand"].iloc[0]):
            df_single["brand"] = "Generic"
            
        # Ensure competitor 3
        if "competitor_3_price" not in df_single or pd.isnull(df_single["competitor_3_price"].iloc[0]):
            df_single["competitor_3_price"] = (df_single["competitor_1_price"] + df_single["competitor_2_price"]) / 2.0

        # Feature engineering
        df_feat = self.engineer_features(df_single)

        # Scale numericals & encode categoricals
        num_scaled = self.scaler.transform(df_feat[self.engineered_numerical_cols])
        cat_encoded = self.encoder.transform(df_feat[self.categorical_cols])

        return np.hstack([num_scaled, cat_encoded])

def main():
    print("[INFO] Running data preprocessing pipeline demonstration...")
    if not config.DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found at: {config.DATASET_PATH}. Run src/data_generation.py first.")
        return

    df_raw = pd.read_csv(config.DATASET_PATH)
    preprocessor = DataPreprocessor()
    df_clean, audit = preprocessor.clean_dataset(df_raw)

    print("\n--- PREPROCESSING AUDIT SUMMARY ---")
    print(f"Original Records:        {audit['raw_row_count']}")
    print(f"Duplicates Removed:      {audit['duplicates_removed']}")
    print(f"Cleaned Records:         {audit['clean_row_count']}")
    print("Outliers Capped:")
    for k, v in audit.items():
        if k.startswith("outliers_capped_"):
            print(f"  - {k.replace('outliers_capped_', '')}: {v} records adjusted to IQR bounds")

    X_train, X_test, y_train, y_test, train_df, test_df = preprocessor.prepare_modeling_data(df_clean)
    print("\n--- MATRIX DIMENSIONS ---")
    print(f"Training Feature Matrix: {X_train.shape}")
    print(f"Testing Feature Matrix:  {X_test.shape}")
    print(f"Total Feature Count:     {len(preprocessor.feature_names)}")
    print(f"Engineered Numericals:   {len(preprocessor.engineered_numerical_cols)}")
    print(f"Categorical Features:    {len(preprocessor.feature_names) - len(preprocessor.engineered_numerical_cols)}")

if __name__ == "__main__":
    main()
