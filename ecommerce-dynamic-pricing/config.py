"""
Global Configuration Module for E-Commerce Dynamic Pricing Framework
Defines system paths, business constraints, scraping headers, and model parameters.
"""

from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"

# Ensure directories exist
for directory in [DATA_DIR, DATABASE_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data File Paths
DATASET_PATH = DATA_DIR / "ecommerce_pricing_dataset.csv"
DB_PATH = DATABASE_DIR / "pricing_history.db"
MODEL_PATH = MODELS_DIR / "pricing_model.pkl"
FEATURE_METADATA_PATH = MODELS_DIR / "feature_metadata.json"
RESULTS_SUMMARY_PATH = REPORTS_DIR / "results_summary.md"
PROJECT_REPORT_PATH = REPORTS_DIR / "project_report.md"

# Pricing Business Rule Constraints
MIN_PROFIT_MARGIN = 0.15          # Absolute floor: at least 15% margin above cost
MAX_PROFIT_MARGIN = 0.65          # Ceiling margin threshold
MAX_PRICE_CHANGE_PCT = 0.15       # Maximum allowable ±15% price adjustment per update
DEFAULT_CURRENCY = "USD"

# Inventory and Demand Scarcity Thresholds
LOW_INVENTORY_THRESHOLD = 50      # Units indicating scarce stock
HIGH_INVENTORY_THRESHOLD = 350    # Units indicating surplus / clearance stock
HIGH_DEMAND_THRESHOLD = 70.0      # Score indicating strong buying interest
LOW_DEMAND_THRESHOLD = 30.0       # Score indicating sluggish interest

# Scraping & Competitor Monitoring Config
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "(DynamicPricingBot/1.0; +https://example.com/bot; compliance-check)"
)
REQUEST_TIMEOUT = 10              # Seconds
PRICE_ALERT_DELTA_PCT = 0.03      # Trigger alert if competitor price moves > 3%
DEFAULT_POLL_INTERVAL_SEC = 60    # Automated monitor polling interval

# Machine Learning Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
NUMERICAL_FEATURES = [
    "cost_price",
    "our_current_price",
    "competitor_1_price",
    "competitor_2_price",
    "competitor_3_price",
    "historical_sales",
    "demand_score",
    "inventory_level",
    "product_rating",
    "discount_percentage",
    "seasonality_index",
    "competitor_price_change"
]
CATEGORICAL_FEATURES = ["category", "brand"]
TARGET_VARIABLE = "recommended_price"
