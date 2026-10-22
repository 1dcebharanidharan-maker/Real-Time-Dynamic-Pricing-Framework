"""
Database Abstraction Layer for SQLite.
Manages relational storage of product catalogs, competitor price audits,
dynamic pricing recommendations, and competitor price change alerts.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return an active SQLite database connection with row factory enabled."""
    target_path = db_path or config.DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[Path] = None) -> None:
    """Create all required tables and indexes if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Product Catalog Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                brand TEXT,
                cost_price REAL NOT NULL,
                current_price REAL NOT NULL,
                inventory_level INTEGER DEFAULT 0,
                demand_score REAL DEFAULT 50.0,
                product_rating REAL DEFAULT 4.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Competitor Price Snapshots Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS competitor_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                competitor_name TEXT NOT NULL,
                price REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                status TEXT DEFAULT 'VALID',
                source_type TEXT DEFAULT 'SIMULATION',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)

        # 3. Dynamic Pricing Recommendations Log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pricing_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                old_price REAL NOT NULL,
                recommended_price REAL NOT NULL,
                cost_price REAL NOT NULL,
                expected_margin_pct REAL NOT NULL,
                recommendation_reason TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)

        # 4. Competitor Price Change Alerts Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_change_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                competitor_name TEXT NOT NULL,
                old_price REAL NOT NULL,
                new_price REAL NOT NULL,
                delta_pct REAL NOT NULL,
                alert_level TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)

        # Create indexing for rapid dashboard and monitor queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comp_pid ON competitor_snapshots(product_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comp_ts ON competitor_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_pid ON pricing_recommendations(product_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_ts ON price_change_alerts(timestamp)")

        conn.commit()

def seed_products_from_dataframe(df: pd.DataFrame, db_path: Optional[Path] = None) -> int:
    """Seed the products catalog table using a deduplicated slice of the generated dataset."""
    init_db(db_path)
    clean_df = df.dropna(subset=["product_id", "our_current_price", "cost_price"]).drop_duplicates(subset=["product_id"])
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        inserted_count = 0
        for _, row in clean_df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO products (
                    product_id, product_name, category, brand, cost_price, 
                    current_price, inventory_level, demand_score, product_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row["product_id"]),
                str(row["product_name"]),
                str(row["category"]),
                str(row["brand"]) if pd.notnull(row["brand"]) else "Generic",
                float(row["cost_price"]),
                float(row["our_current_price"]),
                int(row["inventory_level"]) if pd.notnull(row["inventory_level"]) else 50,
                float(row["demand_score"]) if pd.notnull(row["demand_score"]) else 50.0,
                float(row["product_rating"]) if pd.notnull(row["product_rating"]) else 4.0
            ))
            inserted_count += 1
        conn.commit()
    return inserted_count

def insert_competitor_snapshot(
    product_id: str,
    competitor_name: str,
    price: float,
    currency: str = "USD",
    status: str = "VALID",
    source_type: str = "SIMULATION",
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None
) -> int:
    """Record a verified competitor price snapshot."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if timestamp:
            cursor.execute("""
                INSERT INTO competitor_snapshots 
                (product_id, competitor_name, price, currency, status, source_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (product_id, competitor_name, price, currency, status, source_type, timestamp))
        else:
            cursor.execute("""
                INSERT INTO competitor_snapshots 
                (product_id, competitor_name, price, currency, status, source_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (product_id, competitor_name, price, currency, status, source_type))
        conn.commit()
        return cursor.lastrowid

def insert_recommendation(
    product_id: str,
    old_price: float,
    recommended_price: float,
    cost_price: float,
    expected_margin_pct: float,
    recommendation_reason: str,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None
) -> int:
    """Log an audit trail entry for an automated or simulated pricing recommendation."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if timestamp:
            cursor.execute("""
                INSERT INTO pricing_recommendations 
                (product_id, old_price, recommended_price, cost_price, expected_margin_pct, recommendation_reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (product_id, old_price, recommended_price, cost_price, expected_margin_pct, recommendation_reason, timestamp))
        else:
            cursor.execute("""
                INSERT INTO pricing_recommendations 
                (product_id, old_price, recommended_price, cost_price, expected_margin_pct, recommendation_reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (product_id, old_price, recommended_price, cost_price, expected_margin_pct, recommendation_reason))
        conn.commit()
        return cursor.lastrowid

def insert_alert(
    product_id: str,
    competitor_name: str,
    old_price: float,
    new_price: float,
    delta_pct: float,
    alert_level: str = "INFO",
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None
) -> int:
    """Log a detected competitor price adjustment alert."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if timestamp:
            cursor.execute("""
                INSERT INTO price_change_alerts 
                (product_id, competitor_name, old_price, new_price, delta_pct, alert_level, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (product_id, competitor_name, old_price, new_price, delta_pct, alert_level, timestamp))
        else:
            cursor.execute("""
                INSERT INTO price_change_alerts 
                (product_id, competitor_name, old_price, new_price, delta_pct, alert_level)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (product_id, competitor_name, old_price, new_price, delta_pct, alert_level))
        conn.commit()
        return cursor.lastrowid

def get_latest_competitor_price(product_id: str, competitor_name: str, db_path: Optional[Path] = None) -> Optional[float]:
    """Retrieve the most recent price snapshot for a product and competitor."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT price FROM competitor_snapshots
            WHERE product_id = ? AND competitor_name = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
        """, (product_id, competitor_name))
        row = cursor.fetchone()
        return row["price"] if row else None

def get_all_products(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return all catalog products as a DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM products ORDER BY category, product_id", conn)

def get_recent_snapshots(limit: int = 100, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return the most recent competitor snapshots."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(f"""
            SELECT cs.id, cs.product_id, p.product_name, cs.competitor_name, cs.price, 
                   cs.currency, cs.status, cs.source_type, cs.timestamp
            FROM competitor_snapshots cs
            LEFT JOIN products p ON cs.product_id = p.product_id
            ORDER BY cs.timestamp DESC, cs.id DESC
            LIMIT {int(limit)}
        """, conn)

def get_recent_alerts(limit: int = 50, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return latest competitor price movement alerts."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(f"""
            SELECT a.id, a.product_id, p.product_name, a.competitor_name, a.old_price, 
                   a.new_price, a.delta_pct, a.alert_level, a.timestamp
            FROM price_change_alerts a
            LEFT JOIN products p ON a.product_id = p.product_id
            ORDER BY a.timestamp DESC, a.id DESC
            LIMIT {int(limit)}
        """, conn)

def get_recent_recommendations(limit: int = 50, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return latest pricing recommendations."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(f"""
            SELECT r.id, r.product_id, p.product_name, r.old_price, r.recommended_price, 
                   r.cost_price, r.expected_margin_pct, r.recommendation_reason, r.timestamp
            FROM pricing_recommendations r
            LEFT JOIN products p ON r.product_id = p.product_id
            ORDER BY r.timestamp DESC, r.id DESC
            LIMIT {int(limit)}
        """, conn)

if __name__ == "__main__":
    init_db()
    print(f"[SUCCESS] Database initialized at: {config.DB_PATH}")
    if config.DATASET_PATH.exists():
        df_seed = pd.read_csv(config.DATASET_PATH)
        count = seed_products_from_dataframe(df_seed)
        print(f"[SUCCESS] Seeded {count} unique products into catalog database.")
