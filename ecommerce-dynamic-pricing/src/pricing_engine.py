"""
Dynamic Pricing Rule-Based Engine.
Implements transparent, explainable economic pricing policies incorporating cost floors,
competitor anchoring, inventory-scarcity elasticities, and volatility dampeners.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

class PricingPipeline:
    """Wrapper encapsulating preprocessor and trained estimator for seamless inference."""
    def __init__(self, preprocessor: Any, model: Any):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, df_or_dict: Any) -> Any:
        if isinstance(df_or_dict, dict):
            X = self.preprocessor.transform_single(df_or_dict)
            return float(self.model.predict(X)[0])
        elif isinstance(df_or_dict, pd.DataFrame):
            df_feat = self.preprocessor.engineer_features(df_or_dict)
            num_scaled = self.preprocessor.scaler.transform(df_feat[self.preprocessor.engineered_numerical_cols])
            cat_encoded = self.preprocessor.encoder.transform(df_feat[self.preprocessor.categorical_cols])
            X = np.hstack([num_scaled, cat_encoded])
            return self.model.predict(X)
        else:
            return self.model.predict(df_or_dict)

class RuleBasedPricingEngine:
    """
    Transparent, explainable pricing engine enforcing hard business bounds
    and economic elasticity adjustments.
    """

    def __init__(
        self,
        min_profit_margin: float = config.MIN_PROFIT_MARGIN,
        max_profit_margin: float = config.MAX_PROFIT_MARGIN,
        max_price_change_pct: float = config.MAX_PRICE_CHANGE_PCT,
        low_inventory_threshold: int = config.LOW_INVENTORY_THRESHOLD,
        high_inventory_threshold: int = config.HIGH_INVENTORY_THRESHOLD,
        high_demand_threshold: float = config.HIGH_DEMAND_THRESHOLD,
        low_demand_threshold: float = config.LOW_DEMAND_THRESHOLD
    ):
        self.min_profit_margin = min_profit_margin
        self.max_profit_margin = max_profit_margin
        self.max_price_change_pct = max_price_change_pct
        self.low_inventory_threshold = low_inventory_threshold
        self.high_inventory_threshold = high_inventory_threshold
        self.high_demand_threshold = high_demand_threshold
        self.low_demand_threshold = low_demand_threshold

    def recommend_price(
        self,
        cost_price: float,
        our_current_price: float,
        competitor_prices: List[float],
        demand_score: float = 50.0,
        inventory_level: int = 100,
        product_rating: float = 4.0,
        seasonality_index: float = 1.0,
        competitor_price_change: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates optimal recommended price with complete audit trace and explanation.
        """
        # 1. Sanitize and validate inputs
        cost_price = max(0.01, float(cost_price))
        our_current_price = max(0.01, float(our_current_price))
        valid_comp_prices = [float(p) for p in competitor_prices if p is not None and not np.isnan(p) and p > 0]
        
        if not valid_comp_prices:
            # Fallback to cost-plus baseline markup if competitor data is completely unavailable
            valid_comp_prices = [our_current_price]

        min_comp = min(valid_comp_prices)
        avg_comp = sum(valid_comp_prices) / len(valid_comp_prices)

        rule_triggers = []
        reasons = []

        # 2. Minimum Margin Floor: Never price below cost + minimum profit margin
        cost_floor = round(cost_price * (1.0 + self.min_profit_margin), 2)
        ceiling_price = round(cost_price * (1.0 + self.max_profit_margin), 2)

        # 3. Competitor Market Benchmark (weighted 60% toward lowest competitor to ensure competitiveness)
        market_anchor = 0.60 * min_comp + 0.40 * avg_comp
        reasons.append(f"Market anchor set to ${market_anchor:.2f} based on competitor range (${min_comp:.2f} - ${max(valid_comp_prices):.2f}).")

        # 4. Inventory Scarcity & Demand Dynamic Adjustment
        demand_multiplier = 1.0
        if demand_score >= self.high_demand_threshold and inventory_level <= self.low_inventory_threshold:
            demand_multiplier = 1.08
            rule_triggers.append("SCARCITY_SURGE")
            reasons.append(f"Scarcity surge applied (+8%): High demand ({demand_score:.1f}) with critical stock ({inventory_level} units).")
        elif demand_score <= self.low_demand_threshold and inventory_level >= self.high_inventory_threshold:
            demand_multiplier = 0.92
            rule_triggers.append("CLEARANCE_MARKDOWN")
            reasons.append(f"Clearance markdown applied (-8%): Low demand ({demand_score:.1f}) with excess inventory ({inventory_level} units).")
        elif demand_score >= self.high_demand_threshold:
            demand_multiplier = 1.04
            rule_triggers.append("HIGH_DEMAND_BOOST")
            reasons.append(f"Demand boost (+4%): High consumer interest score ({demand_score:.1f}).")
        elif demand_score <= self.low_demand_threshold:
            demand_multiplier = 0.96
            rule_triggers.append("LOW_DEMAND_DISCOUNT")
            reasons.append(f"Demand discount (-4%): Low consumer interest score ({demand_score:.1f}).")

        # 5. Product Rating / Reputation Factor
        rating_multiplier = 1.0
        if product_rating >= 4.5:
            rating_multiplier = 1.03
            rule_triggers.append("PREMIUM_QUALITY_SURCHARGE")
            reasons.append(f"Quality premium (+3%): High customer satisfaction rating ({product_rating:.1f}/5.0).")
        elif product_rating < 3.2:
            rating_multiplier = 0.97
            rule_triggers.append("LOW_RATING_DISCOUNT")
            reasons.append(f"Quality discount (-3%): Sub-par customer rating ({product_rating:.1f}/5.0).")

        # 6. Seasonality Index Adjustment
        seasonality_factor = 1.0 + (seasonality_index - 1.0) * 0.40
        if abs(seasonality_index - 1.0) >= 0.15:
            rule_triggers.append("SEASONAL_ADJUSTMENT")
            reasons.append(f"Seasonal adjustment ({seasonality_index:.2f}x multiplier applied).")

        # Compute unconstrained target
        raw_price = market_anchor * demand_multiplier * rating_multiplier * seasonality_factor

        # 7. Competitor Price Drop Reaction
        if competitor_price_change < -3.0:
            undercut_target = min_comp * 0.99
            if raw_price > undercut_target:
                raw_price = undercut_target
                rule_triggers.append("COMPETITOR_UNDERCUT_MATCH")
                reasons.append(f"Competitor undercut match applied: Competitor dropped {competitor_price_change:.1f}%.")

        # 8. Enforce Hard Minimum Margin Floor
        bounded_price = raw_price
        if bounded_price < cost_floor:
            bounded_price = cost_floor
            rule_triggers.append("COST_FLOOR_ENFORCED")
            reasons.append(f"Cost floor enforced: Price guarded at minimum ${cost_floor:.2f} (cost + {self.min_profit_margin*100:.0f}% margin).")

        # 9. Apply Price Volatility Dampener (Bound within ±MAX_PRICE_CHANGE_PCT of current price)
        max_allowed = round(our_current_price * (1.0 + self.max_price_change_pct), 2)
        min_allowed = round(max(cost_floor, our_current_price * (1.0 - self.max_price_change_pct)), 2)

        final_price = float(np.clip(bounded_price, min_allowed, max_allowed))
        if final_price == max_allowed and bounded_price > max_allowed:
            rule_triggers.append("PRICE_CAP_DAMPENER")
            reasons.append(f"Price volatility ceiling dampener applied: Capped at +{self.max_price_change_pct*100:.0f}% (${max_allowed:.2f}).")
        elif final_price == min_allowed and bounded_price < min_allowed:
            rule_triggers.append("PRICE_FLOOR_DAMPENER")
            reasons.append(f"Price volatility floor dampener applied: Bounded at -{self.max_price_change_pct*100:.0f}% (${min_allowed:.2f}).")

        final_price = round(final_price, 2)
        unit_profit = round(final_price - cost_price, 2)
        expected_margin_pct = round((unit_profit / final_price) * 100.0, 2)
        price_delta = round(final_price - our_current_price, 2)
        price_delta_pct = round((price_delta / our_current_price) * 100.0, 2)

        return {
            "recommended_price": final_price,
            "our_current_price": our_current_price,
            "cost_price": cost_price,
            "cost_floor": cost_floor,
            "unit_profit": unit_profit,
            "expected_margin_pct": expected_margin_pct,
            "price_delta": price_delta,
            "price_delta_pct": price_delta_pct,
            "min_competitor_price": min_comp,
            "avg_competitor_price": avg_comp,
            "rule_triggers": rule_triggers,
            "recommendation_reason": " | ".join(reasons)
        }

    def recommend_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dynamic pricing engine across an entire pandas DataFrame."""
        results = []
        for _, row in df.iterrows():
            comp_prices = [
                row.get("competitor_1_price"),
                row.get("competitor_2_price"),
                row.get("competitor_3_price")
            ]
            rec = self.recommend_price(
                cost_price=row["cost_price"],
                our_current_price=row["our_current_price"],
                competitor_prices=comp_prices,
                demand_score=row.get("demand_score", 50.0),
                inventory_level=row.get("inventory_level", 100),
                product_rating=row.get("product_rating", 4.0),
                seasonality_index=row.get("seasonality_index", 1.0),
                competitor_price_change=row.get("competitor_price_change", 0.0)
            )
            results.append(rec)
        return pd.DataFrame(results)

if __name__ == "__main__":
    engine = RuleBasedPricingEngine()
    print("[INFO] Testing RuleBasedPricingEngine on sample scenario:")
    sample_rec = engine.recommend_price(
        cost_price=50.0,
        our_current_price=75.0,
        competitor_prices=[72.0, 78.0, 74.0],
        demand_score=85.0,
        inventory_level=25,
        product_rating=4.7,
        seasonality_index=1.2,
        competitor_price_change=-1.0
    )
    for k, v in sample_rec.items():
        print(f"  {k}: {v}")
