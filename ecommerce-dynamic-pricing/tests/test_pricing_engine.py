"""
Comprehensive Unit and Integration Test Suite.
Validates business rules, minimum profit margins, volatility dampeners,
edge-case handling, database persistence, and ML model inference.
"""

from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src import database
from src.pricing_engine import RuleBasedPricingEngine, PricingPipeline
from src.competitor_monitor import PriceValidator, CompetitorPriceMonitor
from src.preprocessing import DataPreprocessor

@pytest.fixture
def engine():
    """Provides a standard pricing engine instance."""
    return RuleBasedPricingEngine(
        min_profit_margin=0.15,
        max_price_change_pct=0.15,
        low_inventory_threshold=50,
        high_inventory_threshold=350,
        high_demand_threshold=70.0,
        low_demand_threshold=30.0
    )

def test_minimum_margin_floor(engine):
    """Ensure recommended price NEVER breaches cost + 15% margin floor."""
    cost = 100.0
    current = 125.0
    # Competitor tries to undercut drastically below our cost
    aggressive_competitors = [80.0, 85.0, 75.0]
    
    rec = engine.recommend_price(
        cost_price=cost,
        our_current_price=current,
        competitor_prices=aggressive_competitors,
        demand_score=20.0,
        inventory_level=400
    )
    
    expected_floor = cost * 1.15
    assert rec["recommended_price"] >= expected_floor, (
        f"Recommended price {rec['recommended_price']} breached minimum floor {expected_floor}"
    )
    assert rec["expected_margin_pct"] >= 13.0  # Approx 15% markup yields ~13.04% gross margin

def test_maximum_price_change_dampener(engine):
    """Ensure price adjustments are bounded within ±15% of current price."""
    cost = 50.0
    current = 100.0
    # Extreme competitor prices attempting a +50% surge
    surge_competitors = [180.0, 190.0, 200.0]
    
    rec = engine.recommend_price(
        cost_price=cost,
        our_current_price=current,
        competitor_prices=surge_competitors,
        demand_score=95.0,
        inventory_level=10
    )
    
    max_allowed = round(current * 1.15, 2)
    assert rec["recommended_price"] <= max_allowed, (
        f"Price {rec['recommended_price']} exceeded maximum allowed ceiling {max_allowed}"
    )
    assert "PRICE_CAP_DAMPENER" in rec["rule_triggers"]

def test_scarcity_surge_rule(engine):
    """Verify that low inventory and high demand trigger scarcity surge."""
    rec = engine.recommend_price(
        cost_price=60.0,
        our_current_price=90.0,
        competitor_prices=[92.0, 94.0, 91.0],
        demand_score=85.0,     # High demand (>70)
        inventory_level=20     # Low inventory (<50)
    )
    assert "SCARCITY_SURGE" in rec["rule_triggers"]
    assert rec["recommended_price"] > rec["our_current_price"]

def test_clearance_markdown_rule(engine):
    """Verify that high inventory and low demand trigger clearance markdown."""
    rec = engine.recommend_price(
        cost_price=60.0,
        our_current_price=90.0,
        competitor_prices=[88.0, 90.0, 89.0],
        demand_score=25.0,     # Low demand (<30)
        inventory_level=400    # Excess inventory (>350)
    )
    assert "CLEARANCE_MARKDOWN" in rec["rule_triggers"]
    assert rec["recommended_price"] < rec["our_current_price"]

def test_invalid_and_missing_prices():
    """Verify price validator handles strings, None, negative, and extreme values."""
    # Valid numeric
    valid, price, msg = PriceValidator.clean_and_validate(89.99)
    assert valid and price == 89.99
    
    # Valid currency string
    valid, price, msg = PriceValidator.clean_and_validate("$ 1,249.50")
    assert valid and price == 1249.50
    
    # Negative price
    valid, price, msg = PriceValidator.clean_and_validate(-50.0)
    assert not valid and "Non-positive" in msg
    
    # Null price
    valid, price, msg = PriceValidator.clean_and_validate(None)
    assert not valid and "Null" in msg
    
    # Extreme anomaly
    valid, price, msg = PriceValidator.clean_and_validate(500000.0)
    assert not valid and "Anomalously high" in msg

def test_competitor_price_change_reaction(engine):
    """Verify that a competitor price drop triggers undercut response."""
    rec = engine.recommend_price(
        cost_price=40.0,
        our_current_price=80.0,
        competitor_prices=[75.0, 76.0, 74.0],
        competitor_price_change=-5.0  # Significant drop (>3%)
    )
    assert "COMPETITOR_UNDERCUT_MATCH" in rec["rule_triggers"]

def test_database_operations(tmp_path):
    """Verify SQLite database schema creation, snapshots, alerts, and recommendations."""
    test_db = tmp_path / "test_pricing.db"
    database.init_db(test_db)
    
    # Insert snapshot
    snap_id = database.insert_competitor_snapshot(
        product_id="TEST-001",
        competitor_name="Competitor_Alpha",
        price=99.50,
        currency="USD",
        source_type="TEST",
        db_path=test_db
    )
    assert snap_id > 0
    
    # Retrieve latest price
    latest_price = database.get_latest_competitor_price("TEST-001", "Competitor_Alpha", db_path=test_db)
    assert latest_price == 99.50
    
    # Insert alert
    alert_id = database.insert_alert(
        product_id="TEST-001",
        competitor_name="Competitor_Alpha",
        old_price=99.50,
        new_price=89.00,
        delta_pct=-10.55,
        alert_level="CRITICAL",
        db_path=test_db
    )
    assert alert_id > 0
    alerts_df = database.get_recent_alerts(limit=5, db_path=test_db)
    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["alert_level"] == "CRITICAL"

def test_model_pipeline_loading_and_prediction():
    """Verify that the serialized ML model loads and performs valid predictions."""
    assert config.MODEL_PATH.exists(), f"Model file missing at {config.MODEL_PATH}"
    pipeline = joblib.load(config.MODEL_PATH)
    
    sample_input = {
        "category": "Electronics",
        "brand": "Apple",
        "cost_price": 500.0,
        "our_current_price": 650.0,
        "competitor_1_price": 640.0,
        "competitor_2_price": 660.0,
        "competitor_3_price": 645.0,
        "historical_sales": 75,
        "demand_score": 82.0,
        "inventory_level": 30,
        "product_rating": 4.8,
        "discount_percentage": 5.0,
        "seasonality_index": 1.15,
        "competitor_price_change": -0.5
    }
    
    pred_price = pipeline.predict(sample_input)
    assert isinstance(pred_price, (float, int, np.floating))
    assert pred_price > 500.0, f"Predicted price {pred_price} must exceed cost $500.0"
    assert pred_price < 850.0, f"Predicted price {pred_price} is unrealistically high"

def test_preprocessor_iqr_outlier_capping():
    """Verify that extreme outliers are clipped to IQR bounds."""
    preprocessor = DataPreprocessor()
    test_df = pd.DataFrame({
        "inventory_level": [20, 25, 30, 28, 35, 10000],  # 10000 is an extreme outlier
        "competitor_1_price": [50.0, 52.0, 49.0, 51.0, 53.0, 5000.0],
        "competitor_2_price": [50.0, 52.0, 49.0, 51.0, 53.0, 50.0],
        "competitor_3_price": [50.0, 52.0, 49.0, 51.0, 53.0, 50.0],
        "product_id": [f"P{i}" for i in range(6)],
        "category": ["Electronics"] * 6,
        "brand": ["BrandX"] * 6,
        "our_current_price": [60.0] * 6,
        "cost_price": [40.0] * 6,
        "historical_sales": [10] * 6,
        "demand_score": [50.0] * 6,
        "product_rating": [4.0] * 6,
        "discount_percentage": [0.0] * 6,
        "seasonality_index": [1.0] * 6,
        "competitor_price_change": [0.0] * 6,
        "profit_margin": [33.3] * 6,
        "timestamp": ["2026-10-22"] * 6,
        "recommended_price": [60.0] * 6
    })
    
    clean_df, audit = preprocessor.clean_dataset(test_df)
    assert clean_df["inventory_level"].max() < 10000, "Outlier inventory was not capped"
    assert clean_df["competitor_1_price"].max() < 5000.0, "Outlier competitor price was not capped"
