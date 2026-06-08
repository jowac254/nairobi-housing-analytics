"""
scraper.py — Nairobi Housing Data Collector
Scrapes rental listings from BuyRentKenya and Propertybase.
Author: John Wachira | github.com/jowac254
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://www.buyrentkenya.com/houses-for-rent"
DELAY_SECONDS = 1.5  # Polite scraping delay
OUTPUT_DIR = Path("../data/raw")


def scrape_buyrentkenya(pages: int = 100) -> pd.DataFrame:
    """
    Scrape BuyRentKenya rental listings.
    
    Args:
        pages: Number of result pages to scrape
    
    Returns:
        DataFrame of raw listings
    """
    listings = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting BuyRentKenya scrape — {pages} pages")

    for page in range(1, pages + 1):
        url = f"{BASE_URL}?page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Page {page} failed: {e}")
            continue

        soup = BeautifulSoup(res.text, 'html.parser')
        cards = soup.select('.listing-card, [data-testid="listing-card"]')

        if not cards:
            logger.info(f"No listings found on page {page} — stopping")
            break

        for card in cards:
            try:
                listing = {
                    'source':        'buyrentkenya',
                    'title':         _safe_text(card, '.title, h2'),
                    'neighbourhood': _safe_text(card, '.location, .area'),
                    'price':         _safe_text(card, '.price, [data-testid="price"]'),
                    'bedrooms':      _safe_text(card, '.beds, [data-beds]'),
                    'bathrooms':     _safe_text(card, '.baths, [data-baths]'),
                    'size_sqft':     _safe_text(card, '.size, [data-size]'),
                    'listed_date':   _safe_text(card, '.date, time'),
                    'url':           _safe_attr(card, 'a', 'href'),
                    'scraped_at':    datetime.now().isoformat(),
                }
                listings.append(listing)
            except Exception as e:
                logger.debug(f"Card parse error: {e}")
                continue

        logger.info(f"Page {page}/{pages} — collected {len(cards)} listings (total: {len(listings)})")
        time.sleep(DELAY_SECONDS)

    df = pd.DataFrame(listings)
    output_path = OUTPUT_DIR / f"buyrentkenya_listings_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} listings → {output_path}")
    return df


def scrape_propertybase(pages: int = 50) -> pd.DataFrame:
    """
    Scrape Propertybase Kenya listings.
    Mirrors structure of scrape_buyrentkenya.
    """
    listings = []
    base = "https://www.propertybase.co.ke/properties-for-rent/nairobi"

    for page in range(1, pages + 1):
        url = f"{base}?page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')

            for card in soup.select('.property-item, .prop-card'):
                listings.append({
                    'source':        'propertybase',
                    'title':         _safe_text(card, 'h3, .prop-title'),
                    'neighbourhood': _safe_text(card, '.prop-location'),
                    'price':         _safe_text(card, '.prop-price'),
                    'bedrooms':      _safe_text(card, '.prop-beds'),
                    'bathrooms':     _safe_text(card, '.prop-baths'),
                    'size_sqft':     _safe_text(card, '.prop-area'),
                    'listed_date':   datetime.now().strftime('%Y-%m-%d'),
                    'url':           _safe_attr(card, 'a', 'href'),
                    'scraped_at':    datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Propertybase page {page} error: {e}")

        time.sleep(DELAY_SECONDS)

    df = pd.DataFrame(listings)
    output_path = OUTPUT_DIR / f"propertybase_listings_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Propertybase: saved {len(df):,} listings → {output_path}")
    return df


def _safe_text(card, selector: str) -> str | None:
    """Safely extract text from first matching element."""
    for sel in selector.split(','):
        el = card.select_one(sel.strip())
        if el:
            return el.get_text(strip=True)
    return None


def _safe_attr(card, tag: str, attr: str) -> str | None:
    """Safely extract attribute from element."""
    el = card.select_one(tag)
    return el[attr] if el and el.has_attr(attr) else None


def load_all_raw() -> pd.DataFrame:
    """Load and combine all raw CSV files."""
    dfs = []
    for csv_file in OUTPUT_DIR.glob("*.csv"):
        try:
            dfs.append(pd.read_csv(csv_file))
            logger.info(f"Loaded {csv_file.name}")
        except Exception as e:
            logger.warning(f"Could not load {csv_file}: {e}")
    
    if not dfs:
        raise FileNotFoundError("No raw data files found. Run scraper first.")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined):,} rows")
    return combined


if __name__ == "__main__":
    logger.info("=== Nairobi Housing Scraper ===")
    df_brk = scrape_buyrentkenya(pages=100)
    df_pb  = scrape_propertybase(pages=50)
    combined = pd.concat([df_brk, df_pb], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_raw_listings.csv", index=False)
    logger.info(f"Total raw records collected: {len(combined):,}")
