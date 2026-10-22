"""
Synthetic Dataset Generator for E-Commerce Dynamic Pricing Framework.
Generates >= 5,000 realistic e-commerce observations with economic pricing dynamics,
injected missing values, duplicates, and statistical outliers for preprocessing validation.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

def generate_synthetic_data(num_records: int = 5500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic e-commerce pricing data.
    
    Economic Logic:
    - Cost price varies by category and product type.
    - Our current price reflects standard retail markup (20% to 60% above cost).
    - Competitor prices fluctuate around current market clearing levels with stochastic deviations.
    - Demand score correlates negatively with price ratio and positively with rating and seasonality.
    - Recommended price is computed through an explicit multi-factor pricing policy:
      Cost Floor + Competitor Benchmark + Scarcity Surge - Clearance Discount + Quality Premium.
    """
    np.random.seed(random_state)

    categories = {
        "Electronics": {
            "brands": ["Apple", "Samsung", "Sony", "Dell", "Bose", "LG", "Anker"],
            "cost_range": (40.0, 950.0),
            "markup_range": (1.25, 1.65),
            "templates": ["Wireless Noise-Canceling Headphones", "Smart 4K UHD TV 55-inch", "Ultra-Slim Laptop 15.6",
                          "Gaming Console Pro", "Bluetooth Portable Speaker", "Smart Watch Series 8",
                          "High-Speed Wi-Fi Router", "USB-C Fast Charging Hub", "Noise Isolating Earbuds", "Tablet Air 11-inch"]
        },
        "Fashion": {
            "brands": ["Nike", "Adidas", "Zara", "Levi's", "Under Armour", "H&M", "Puma"],
            "cost_range": (12.0, 160.0),
            "markup_range": (1.35, 1.95),
            "templates": ["Slim-Fit Stretch Denim Jeans", "Performance Running Shoes", "Organic Cotton Crewneck T-Shirt",
                          "Water-Resistant Winter Parka", "Breathable Athletic Leggings", "Classic Leather Oxford Shoes",
                          "Fleece Pullover Hoodie", "Tailored Formal Suit Jacket", "Lightweight Windbreaker Jacket"]
        },
        "Home & Kitchen": {
            "brands": ["Philips", "Dyson", "Ninja", "Instant Pot", "Cuisinart", "KitchenAid", "iRobot"],
            "cost_range": (20.0, 420.0),
            "markup_range": (1.30, 1.70),
            "templates": ["Multi-Function Air Fryer 6Qt", "Cordless Stick Vacuum Cleaner", "Automatic Espresso Coffee Maker",
                          "Stainless Steel Cookware Set 10pc", "Programmable Pressure Cooker", "High-Velocity Stand Blender",
                          "Robot Vacuum with Smart Mapping", "HEPA Air Purifier for Large Rooms", "Precision Chef Knife Set"]
        },
        "Sports & Outdoors": {
            "brands": ["Wilson", "Columbia", "Garmin", "Spalding", "The North Face", "Hydro Flask", "Coleman"],
            "cost_range": (15.0, 320.0),
            "markup_range": (1.28, 1.75),
            "templates": ["All-Weather Camping Dome Tent 4-Person", "GPS Multisport Fitness Watch", "Insulated Stainless Water Bottle 32oz",
                          "Official Competition Basketball", "Adjustable Dumbbell Weight Set 50lb", "Thermal Insulated Sleeping Bag",
                          "Carbon Fiber Trekking Poles", "Inflatable Stand Up Paddle Board"]
        },
        "Books & Media": {
            "brands": ["Penguin", "HarperCollins", "Simon & Schuster", "O'Reilly", "Pearson", "Macmillan"],
            "cost_range": (6.0, 65.0),
            "markup_range": (1.30, 1.80),
            "templates": ["Data Science and Machine Learning Handbook", "The Art of Strategic Leadership",
                          "Classic Hardcover Literature Anthology", "Deep Learning Foundations Guide",
                          "Python Architecture in Practice", "Financial Markets and Valuation", "World History Illustrated Atlas"]
        }
    }

    records = []
    cat_keys = list(categories.keys())
    cat_weights = [0.30, 0.25, 0.20, 0.15, 0.10]

    # Generate base products
    for i in range(num_records):
        cat = np.random.choice(cat_keys, p=cat_weights)
        cat_info = categories[cat]
        brand = np.random.choice(cat_info["brands"])
        template = np.random.choice(cat_info["templates"])
        product_id = f"PRD-{cat[:3].upper()}-{1000 + i:05d}"
        product_name = f"{brand} {template}"

        # Financial baselines
        min_cost, max_cost = cat_info["cost_range"]
        cost_price = float(np.round(np.random.uniform(min_cost, max_cost), 2))
        markup = np.random.uniform(*cat_info["markup_range"])
        our_current_price = float(np.round(cost_price * markup, 2))

        # Competitor pricing with realistic dispersion
        # Competitor 1: aggressive discounter (-8% to +4%)
        # Competitor 2: balanced peer (-5% to +6%)
        # Competitor 3: premium or boutique (-2% to +10%)
        comp1_factor = np.random.uniform(0.92, 1.04)
        comp2_factor = np.random.uniform(0.95, 1.06)
        comp3_factor = np.random.uniform(0.97, 1.10)

        competitor_1_price = float(np.round(our_current_price * comp1_factor, 2))
        competitor_2_price = float(np.round(our_current_price * comp2_factor, 2))
        competitor_3_price = float(np.round(our_current_price * comp3_factor, 2))

        # Competitor price change over last 7 days (-12% to +10%)
        competitor_price_change = float(np.round(np.random.normal(-0.5, 4.0), 2))

        # Operational metrics
        inventory_level = int(np.random.choice([
            np.random.randint(5, 50),     # 20% scarce stock
            np.random.randint(50, 300),   # 65% normal stock
            np.random.randint(300, 520)   # 15% overstock / clearance
        ], p=[0.20, 0.65, 0.15]))

        product_rating = float(np.round(np.random.choice([
            np.random.uniform(1.8, 3.4),  # 10% lower quality
            np.random.uniform(3.5, 4.4),  # 60% solid quality
            np.random.uniform(4.5, 5.0)   # 30% top tier
        ], p=[0.10, 0.60, 0.30]), 1))

        seasonality_index = float(np.round(np.random.uniform(0.85, 1.45), 2))

        # Demand score (1 to 100) influenced by price ratio, rating, and seasonality
        avg_comp = (competitor_1_price + competitor_2_price + competitor_3_price) / 3.0
        price_competitiveness = avg_comp / our_current_price  # >1 means our price is cheaper than competitors
        raw_demand = 45.0 + (price_competitiveness - 1.0) * 40.0 + (product_rating - 3.5) * 15.0 + (seasonality_index - 1.0) * 20.0 + np.random.normal(0, 6)
        demand_score = float(np.round(np.clip(raw_demand, 5.0, 99.0), 1))

        # Historical monthly sales correlated with demand and inventory
        sales_potential = (demand_score / 100.0) * 180.0 * seasonality_index + np.random.normal(0, 10)
        historical_sales = int(np.clip(sales_potential, 0, inventory_level + 200))

        discount_percentage = float(np.round(np.random.choice([
            0.0,
            np.random.uniform(5.0, 15.0),
            np.random.uniform(16.0, 30.0)
        ], p=[0.50, 0.35, 0.15]), 1))

        # Current gross margin percentage
        profit_margin = float(np.round(((our_current_price - cost_price) / our_current_price) * 100.0, 2))

        # Timestamp generation across past 90 days
        day_offset = np.random.randint(0, 90)
        timestamp = (pd.Timestamp("2026-10-22") - pd.Timedelta(days=day_offset, hours=np.random.randint(0, 24))).strftime("%Y-%m-%d %H:%M:%S")

        # =========================================================================
        # TARGET VARIABLE GENERATION: Documented Economic Pricing Policy
        # =========================================================================
        # 1. Cost Floor: Must guarantee at least MIN_PROFIT_MARGIN (15%)
        cost_floor = cost_price * (1.0 + config.MIN_PROFIT_MARGIN)

        # 2. Market Competitor Anchor: Weighted toward lowest competitor to ensure competitiveness
        min_comp = min(competitor_1_price, competitor_2_price, competitor_3_price)
        avg_comp = (competitor_1_price + competitor_2_price + competitor_3_price) / 3.0
        market_anchor = 0.60 * min_comp + 0.40 * avg_comp

        # 3. Scarcity & Demand Dynamic Multiplier
        demand_factor = 1.0
        if demand_score >= config.HIGH_DEMAND_THRESHOLD and inventory_level <= config.LOW_INVENTORY_THRESHOLD:
            # Scarcity Surge: High demand, critical low stock -> raise price up to 10%
            demand_factor = 1.08
        elif demand_score <= config.LOW_DEMAND_THRESHOLD and inventory_level >= config.HIGH_INVENTORY_THRESHOLD:
            # Clearance Markdown: Weak demand, surplus inventory -> discount up to 8%
            demand_factor = 0.92
        elif demand_score >= config.HIGH_DEMAND_THRESHOLD:
            demand_factor = 1.04
        elif demand_score <= config.LOW_DEMAND_THRESHOLD:
            demand_factor = 0.96

        # 4. Product Quality / Rating Premium
        rating_factor = 1.0
        if product_rating >= 4.5:
            rating_factor = 1.03  # High reputation allows +3% premium
        elif product_rating < 3.2:
            rating_factor = 0.97  # Poor rating requires -3% discount

        # 5. Seasonality adjustment
        seasonal_factor = 1.0 + (seasonality_index - 1.0) * 0.40

        # Unconstrained policy recommendation
        raw_recommended = market_anchor * demand_factor * rating_factor * seasonal_factor

        # 6. Apply Competitor Undercut Reaction: if competitor price dropped > 3%, match slightly
        if competitor_price_change < -3.0:
            raw_recommended = min(raw_recommended, min_comp * 0.99)

        # 7. Apply Hard Margin Floor Constraint
        bounded_recommended = max(raw_recommended, cost_floor)

        # 8. Apply Price Volatility Dampener (Maximum ±15% deviation from current price)
        max_allowed_price = our_current_price * (1.0 + config.MAX_PRICE_CHANGE_PCT)
        min_allowed_price = max(cost_floor, our_current_price * (1.0 - config.MAX_PRICE_CHANGE_PCT))
        final_recommended_price = float(np.round(np.clip(bounded_recommended, min_allowed_price, max_allowed_price), 2))

        records.append({
            "product_id": product_id,
            "product_name": product_name,
            "category": cat,
            "brand": brand,
            "our_current_price": our_current_price,
            "competitor_1_price": competitor_1_price,
            "competitor_2_price": competitor_2_price,
            "competitor_3_price": competitor_3_price,
            "cost_price": cost_price,
            "historical_sales": historical_sales,
            "demand_score": demand_score,
            "inventory_level": inventory_level,
            "product_rating": product_rating,
            "discount_percentage": discount_percentage,
            "seasonality_index": seasonality_index,
            "competitor_price_change": competitor_price_change,
            "profit_margin": profit_margin,
            "timestamp": timestamp,
            "recommended_price": final_recommended_price
        })

    df = pd.DataFrame(records)

    # =========================================================================
    # INJECT REALISTIC IMPERFECTIONS FOR PREPROCESSING DEMONSTRATION
    # =========================================================================
    # 1. Missing Values (~2.5% in select columns)
    missing_rating_idx = np.random.choice(df.index, size=int(0.025 * len(df)), replace=False)
    df.loc[missing_rating_idx, "product_rating"] = np.nan

    missing_sales_idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_sales_idx, "historical_sales"] = np.nan

    missing_comp3_idx = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_comp3_idx, "competitor_3_price"] = np.nan

    missing_brand_idx = np.random.choice(df.index, size=int(0.015 * len(df)), replace=False)
    df.loc[missing_brand_idx, "brand"] = np.nan

    # 2. Outliers (0.5% extreme values in inventory and price)
    outlier_idx = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df.loc[outlier_idx[:len(outlier_idx)//2], "inventory_level"] = df.loc[outlier_idx[:len(outlier_idx)//2], "inventory_level"] * 5
    df.loc[outlier_idx[len(outlier_idx)//2:], "competitor_1_price"] = df.loc[outlier_idx[len(outlier_idx)//2:], "competitor_1_price"] * 2.8

    # 3. Duplicate Records (~1.5% duplicate rows)
    dup_sample_size = int(0.015 * len(df))
    dup_rows = df.sample(n=dup_sample_size, random_state=random_state)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df

def main():
    print(f"[INFO] Generating synthetic e-commerce pricing dataset (>= 5,000 records)...")
    df = generate_synthetic_data(num_records=5200, random_state=config.RANDOM_STATE)
    
    # Ensure data directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_PATH, index=False)

    print(f"[SUCCESS] Dataset successfully saved to: {config.DATASET_PATH}")
    print(f"[SUMMARY] Total Rows Generated: {len(df):,}")
    print(f"[SUMMARY] Total Columns: {len(df.columns)}")
    print(f"[SUMMARY] Categories: {df['category'].dropna().unique().tolist()}")
    print(f"[SUMMARY] Injected Missing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"[SUMMARY] Injected Duplicate Rows: {df.duplicated().sum()}")
    print(f"[SUMMARY] Price Range: ${df['our_current_price'].min():.2f} - ${df['our_current_price'].max():.2f}")
    print(f"[SUMMARY] Recommended Price Range: ${df['recommended_price'].min():.2f} - ${df['recommended_price'].max():.2f}")

if __name__ == "__main__":
    main()
