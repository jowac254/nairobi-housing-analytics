"""
cleaner.py — Data Cleaning Pipeline
Cleans, standardises, and enriches raw Nairobi housing listings.
Author: John Wachira | github.com/jowac254
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# NEIGHBOURHOOD STANDARDISATION MAP
# Handles spelling variants found in scraped data
# ============================================================
NEIGHBOURHOOD_MAP = {
    # Westlands cluster
    'westlands': 'Westlands',
    'west lands': 'Westlands',
    'westland': 'Westlands',
    'parklands': 'Parklands',
    'park lands': 'Parklands',

    # Kilimani / Kileleshwa
    'kilimani': 'Kilimani',
    'kileleshwa': 'Kileleshwa',
    'kileleshwa.': 'Kileleshwa',

    # Karen / Lavington
    'karen': 'Karen',
    'lavington': 'Lavington',
    'lower kabete': 'Lower Kabete',

    # South Nairobi
    'south b': 'South B',
    'south b.': 'South B',
    'south-b': 'South B',
    'south c': 'South C',
    'south c.': 'South C',
    'langata': 'Langata',
    "lang'ata": 'Langata',
    'hardy': 'Langata',

    # East Nairobi
    'donholm': 'Donholm',
    'don holm': 'Donholm',
    'savannah': 'Donholm',
    'umoja': 'Umoja',
    'umoja 1': 'Umoja',
    'umoja 2': 'Umoja',
    'kayole': 'Kayole',

    # Satellite towns
    'rongai': 'Rongai',
    'athi river': 'Athi River',
    'athiriver': 'Athi River',
    'athi-river': 'Athi River',
    'ruiru': 'Ruiru',
    'ruaka': 'Ruaka',
    'ruiru/ruaka': 'Ruiru',
    'thika road': 'Thika Road',

    # Northern Nairobi
    'githurai': 'Githurai',
    'githurai 44': 'Githurai',
    'githurai 45': 'Githurai',
    'kahawa': 'Kahawa West',
    'kahawa west': 'Kahawa West',
    'ruai': 'Ruai',

    # CBD adjacent
    'cbd': 'Nairobi CBD',
    'nairobi cbd': 'Nairobi CBD',
    'upper hill': 'Upper Hill',
    'upperhill': 'Upper Hill',
    'milimani': 'Milimani',
    'hurlingham': 'Hurlingham',
}

# Distance to CBD estimates (km) for each canonical neighbourhood
DIST_TO_CBD = {
    'Nairobi CBD': 0,
    'Upper Hill': 3,
    'Kilimani': 5,
    'Kileleshwa': 7,
    'Westlands': 6,
    'Parklands': 7,
    'Hurlingham': 6,
    'Milimani': 4,
    'South B': 8,
    'South C': 9,
    'Langata': 12,
    'Karen': 18,
    'Lavington': 9,
    'Donholm': 14,
    'Umoja': 12,
    'Kayole': 16,
    'Githurai': 18,
    'Kahawa West': 20,
    'Ruai': 22,
    'Rongai': 28,
    'Athi River': 30,
    'Ruiru': 22,
    'Ruaka': 14,
    'Thika Road': 20,
}


def clean_price(price_str: str) -> float | None:
    """
    Parse rent from string formats like:
    'KES 45,000/month', 'Ksh 45000', '45,000 per month', '45K'
    
    Returns float (monthly KES) or None if unparseable.
    """
    if pd.isna(price_str) or str(price_str).strip() == '':
        return None
    
    s = str(price_str).upper().strip()
    
    # Handle 'K' shorthand (e.g., '45K' = 45,000)
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*K\b', s)
    if k_match:
        return float(k_match.group(1)) * 1000

    # Extract all digit sequences
    digits = re.sub(r'[^\d]', '', s)
    if not digits:
        return None
    
    val = float(digits)
    
    # Sanity check: Nairobi rents are between KES 5,000 and KES 500,000/month
    if 5_000 <= val <= 500_000:
        return val
    
    # Some listings show annual rent — divide by 12
    if 60_000 <= val <= 6_000_000:
        monthly = val / 12
        if 5_000 <= monthly <= 500_000:
            return monthly
    
    return None


def clean_bedrooms(bedroom_str: str) -> int | None:
    """Parse bedroom count from '3 Bed', '2BR', 'Studio', 'Bedsitter'."""
    if pd.isna(bedroom_str):
        return None
    
    s = str(bedroom_str).lower().strip()
    
    if any(w in s for w in ['studio', 'bedsit', 'bedsitter', 'single room', 'servant']):
        return 0  # 0 = bedsitter/studio
    
    match = re.search(r'(\d+)', s)
    return int(match.group(1)) if match else None


def standardise_neighbourhood(name: str) -> str:
    """Map variant spellings to canonical neighbourhood names."""
    if pd.isna(name):
        return 'Unknown'
    
    key = str(name).lower().strip()
    
    # Direct match
    if key in NEIGHBOURHOOD_MAP:
        return NEIGHBOURHOOD_MAP[key]
    
    # Partial match
    for variant, canonical in NEIGHBOURHOOD_MAP.items():
        if variant in key or key in variant:
            return canonical
    
    # Fallback: title-case the raw value
    return str(name).strip().title()


def remove_outliers(df: pd.DataFrame, col: str = 'price_kes',
                    lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """Remove listings with extreme rent values using percentile bounds."""
    lower = df[col].quantile(lower_pct)
    upper = df[col].quantile(upper_pct)
    mask = df[col].between(lower, upper)
    removed = (~mask).sum()
    logger.info(f"Outlier removal: dropped {removed:,} rows (bounds: {lower:,.0f}–{upper:,.0f})")
    return df[mask].copy()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    
    Steps:
    1. Parse price → float
    2. Parse bedrooms → int
    3. Standardise neighbourhood names
    4. Add distance-to-CBD feature
    5. Remove outliers
    6. Remove duplicates
    7. Add derived columns
    """
    logger.info(f"Starting cleaning pipeline: {len(df):,} rows")
    
    # 1. Price
    df['price_kes'] = df['price'].apply(clean_price)
    df = df.dropna(subset=['price_kes'])
    logger.info(f"After price parsing: {len(df):,} rows")
    
    # 2. Bedrooms
    df['bedrooms_clean'] = df['bedrooms'].apply(clean_bedrooms)
    
    # 3. Neighbourhood
    df['neighbourhood_clean'] = df['neighbourhood'].apply(standardise_neighbourhood)
    
    # 4. Distance to CBD
    df['dist_to_cbd_km'] = df['neighbourhood_clean'].map(DIST_TO_CBD).fillna(15)
    
    # 5. Remove outliers
    df = remove_outliers(df)
    
    # 6. Remove duplicates
    if 'url' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset='url', keep='last')
        logger.info(f"Removed {before - len(df):,} duplicate URLs")
    
    # 7. Derived columns
    df['price_tier'] = pd.cut(
        df['price_kes'],
        bins=[0, 20_000, 45_000, 80_000, float('inf')],
        labels=['Economy', 'Mid-Range', 'Upper-Mid', 'Premium']
    )
    df['bedroom_label'] = df['bedrooms_clean'].map({
        0: 'Bedsitter', 1: '1 Bedroom', 2: '2 Bedroom',
        3: '3 Bedroom', 4: '4 Bedroom', 5: '5+ Bedroom'
    }).fillna('Unknown')
    
    logger.info(f"Cleaning complete: {len(df):,} clean records")
    logger.info(f"Neighbourhoods: {df['neighbourhood_clean'].nunique()}")
    logger.info(f"Price range: KES {df['price_kes'].min():,.0f} – KES {df['price_kes'].max():,.0f}")
    
    return df


def generate_neighbourhood_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate listings to neighbourhood-level summary statistics."""
    summary = df.groupby('neighbourhood_clean').agg(
        listing_count=('price_kes', 'count'),
        median_rent=('price_kes', 'median'),
        mean_rent=('price_kes', 'mean'),
        min_rent=('price_kes', 'min'),
        max_rent=('price_kes', 'max'),
        std_rent=('price_kes', 'std'),
        median_bedrooms=('bedrooms_clean', 'median'),
        dist_to_cbd=('dist_to_cbd_km', 'first')
    ).reset_index()
    
    # Price-per-bedroom (where available)
    summary['median_rent_per_bed'] = summary['median_rent'] / summary['median_bedrooms'].clip(lower=1)
    
    return summary.sort_values('median_rent')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    raw_path = Path("../data/raw/all_raw_listings.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run scraper.py first.")
    
    df_raw = pd.read_csv(raw_path)
    df_clean = clean_dataset(df_raw)
    
    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_clean.to_csv(out_dir / "cleaned_listings.csv", index=False)
    
    summary = generate_neighbourhood_summary(df_clean)
    summary.to_csv(out_dir / "neighbourhood_summary.csv", index=False)
    
    logger.info("Cleaning pipeline complete.")
    print(summary.to_string())
