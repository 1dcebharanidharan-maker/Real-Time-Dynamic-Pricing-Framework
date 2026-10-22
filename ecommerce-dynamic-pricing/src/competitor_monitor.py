"""
Competitor Price Monitoring Engine.
Supports live authorized web scraping, REST API ingestion, robots.txt compliance checking,
data validation, price change detection, and deterministic market simulation fallback.
"""

import re
import sys
import time
import random
import urllib.robotparser
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import requests
from bs4 import BeautifulSoup

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src import database

class PriceValidator:
    """Validates raw extracted prices to prevent bad data ingestion."""
    
    @staticmethod
    def clean_and_validate(raw_price: Any, currency: str = "USD") -> Tuple[bool, Optional[float], str]:
        """
        Clean currency symbols, parse float, and validate range.
        Returns: (is_valid, parsed_price, error_or_status_message)
        """
        if raw_price is None:
            return False, None, "Null price value"
            
        try:
            if isinstance(raw_price, (int, float)):
                val = float(raw_price)
            else:
                # Remove currency symbols and formatting commas
                cleaned = re.sub(r"[^\d.]", "", str(raw_price).strip())
                val = float(cleaned)
                
            if val <= 0:
                return False, None, f"Non-positive price ({val})"
            if val > 100000:
                return False, None, f"Anomalously high price ({val})"
                
            return True, round(val, 2), "VALID"
        except (ValueError, TypeError) as err:
            return False, None, f"Parsing error: {err}"

class RobotsCompliance:
    """Ensures scraping requests comply with target site robots.txt."""
    
    @staticmethod
    def is_allowed(target_url: str, user_agent: str = config.USER_AGENT) -> bool:
        """Inspect robots.txt on domain before requesting."""
        try:
            parsed = requests.utils.urlparse(target_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(user_agent, target_url)
        except Exception:
            # In case of network failure or missing robots.txt, default to polite policy
            return True

class CompetitorPriceMonitor:
    """
    Modular monitor engine capable of live extraction and simulated market updates.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DB_PATH
        self.user_agent = config.USER_AGENT
        self.timeout = config.REQUEST_TIMEOUT
        
    def fetch_live_page_price(
        self,
        url: str,
        price_selector: str,
        name_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract price from an authorized public URL using CSS selectors.
        Respects robots.txt and handles network timeouts.
        """
        if not RobotsCompliance.is_allowed(url, self.user_agent):
            return {
                "status": "BLOCKED_BY_ROBOTS",
                "price": None,
                "product_name": None,
                "error": "Access prohibited by robots.txt"
            }
            
        headers = {"User-Agent": self.user_agent}
        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Extract price element
            price_elem = soup.select_one(price_selector)
            if not price_elem:
                return {"status": "SELECTOR_NOT_FOUND", "price": None, "error": f"Selector '{price_selector}' not found"}
                
            raw_price = price_elem.get_text(strip=True)
            valid, price, msg = PriceValidator.clean_and_validate(raw_price)
            
            product_name = None
            if name_selector:
                name_elem = soup.select_one(name_selector)
                if name_elem:
                    product_name = name_elem.get_text(strip=True)
                    
            return {
                "status": "SUCCESS" if valid else "INVALID_PRICE",
                "price": price,
                "product_name": product_name,
                "currency": config.DEFAULT_CURRENCY,
                "source_type": "LIVE_SCRAPER",
                "error": None if valid else msg
            }
        except requests.RequestException as err:
            return {"status": "NETWORK_ERROR", "price": None, "error": str(err)}

    def fetch_api_price(self, api_url: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch price from an authorized competitor REST API endpoint."""
        headers = {"User-Agent": self.user_agent}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            
        try:
            resp = requests.get(api_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            # Standard generic JSON payload mapping
            raw_price = data.get("price") or data.get("current_price") or data.get("amount")
            valid, price, msg = PriceValidator.clean_and_validate(raw_price)
            
            return {
                "status": "SUCCESS" if valid else "INVALID_PRICE",
                "price": price,
                "product_name": data.get("product_name"),
                "currency": data.get("currency", config.DEFAULT_CURRENCY),
                "source_type": "AUTHORIZED_API",
                "error": None if valid else msg
            }
        except Exception as err:
            return {"status": "API_ERROR", "price": None, "error": str(err)}

    def record_and_evaluate_snapshot(
        self,
        product_id: str,
        competitor_name: str,
        price: float,
        source_type: str = "SIMULATION",
        currency: str = "USD"
    ) -> Dict[str, Any]:
        """
        Record snapshot in SQLite and detect price delta against prior historical price.
        Triggers alert if delta exceeds configured threshold.
        """
        valid, clean_price, msg = PriceValidator.clean_and_validate(price, currency)
        if not valid:
            return {"status": "REJECTED", "reason": msg}

        # Check latest historical price
        old_price = database.get_latest_competitor_price(product_id, competitor_name, self.db_path)
        
        # Save snapshot
        snapshot_id = database.insert_competitor_snapshot(
            product_id=product_id,
            competitor_name=competitor_name,
            price=clean_price,
            currency=currency,
            status="VALID",
            source_type=source_type,
            db_path=self.db_path
        )

        alert_data = None
        if old_price is not None and old_price > 0:
            delta_pct = ((clean_price - old_price) / old_price) * 100.0
            abs_delta = abs(delta_pct)
            
            # If price changed by more than configured threshold
            if abs_delta >= (config.PRICE_ALERT_DELTA_PCT * 100.0):
                alert_level = "CRITICAL" if abs_delta >= 10.0 else "WARNING"
                alert_id = database.insert_alert(
                    product_id=product_id,
                    competitor_name=competitor_name,
                    old_price=old_price,
                    new_price=clean_price,
                    delta_pct=round(delta_pct, 2),
                    alert_level=alert_level,
                    db_path=self.db_path
                )
                alert_data = {
                    "alert_id": alert_id,
                    "old_price": old_price,
                    "new_price": clean_price,
                    "delta_pct": round(delta_pct, 2),
                    "alert_level": alert_level
                }

        return {
            "status": "SAVED",
            "snapshot_id": snapshot_id,
            "product_id": product_id,
            "competitor_name": competitor_name,
            "price": clean_price,
            "source_type": source_type,
            "alert": alert_data
        }

    def simulate_competitor_feed(self, num_samples: int = 15) -> List[Dict[str, Any]]:
        """
        Generate realistic simulated live market tick updates for registered products.
        Marks source_type explicitly as 'SIMULATION'.
        """
        products_df = database.get_all_products(self.db_path)
        if products_df.empty:
            return []

        sampled = products_df.sample(min(num_samples, len(products_df)), random_state=random.randint(1, 10000))
        competitors = ["Amazon_Sim", "Walmart_Sim", "Target_Sim", "BestBuy_Sim"]
        
        feed_results = []
        for _, row in sampled.iterrows():
            comp = random.choice(competitors)
            current_base = float(row["current_price"])
            
            # Simulate market stochastic adjustment (-7% to +5%)
            shock = random.choices([0.0, -0.04, 0.03, -0.08, 0.06], weights=[0.40, 0.25, 0.20, 0.10, 0.05])[0]
            simulated_price = round(current_base * (1.0 + shock) + random.uniform(-0.5, 0.5), 2)
            
            result = self.record_and_evaluate_snapshot(
                product_id=row["product_id"],
                competitor_name=comp,
                price=simulated_price,
                source_type="SIMULATION"
            )
            feed_results.append(result)
            
        return feed_results

if __name__ == "__main__":
    monitor = CompetitorPriceMonitor()
    print("[INFO] Running test competitor price monitoring simulation cycle...")
    results = monitor.simulate_competitor_feed(num_samples=10)
    print(f"[SUCCESS] Recorded {len(results)} competitor snapshots.")
    alerts_triggered = [r for r in results if r.get("alert") is not None]
    print(f"[INFO] Competitor price movement alerts triggered: {len(alerts_triggered)}")
    for alert in alerts_triggered:
        print(f"  -> ALERT: {alert['product_id']} on {alert['competitor_name']} moved {alert['alert']['delta_pct']}% (${alert['alert']['old_price']} -> ${alert['alert']['new_price']})")
