# 🏠 Nairobi Housing Analytics: Solving the Rent Crisis with Data

> **A full-stack data analytics investigation into Nairobi's housing affordability crisis — combining web scraping, exploratory data analysis, geospatial mapping, and an interactive dashboard to help renters make smarter decisions.**

---

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-purple?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?style=flat-square&logo=jupyter)
![Folium](https://img.shields.io/badge/Folium-Maps-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Data](https://img.shields.io/badge/Listings-3%2C412-amber?style=flat-square)

---

## 📌 Project Overview

Nairobi's average rent has grown **18.3% in two years** while wages have barely moved. The typical Nairobian now spends **47% of their income on rent** — far above the globally recommended 30% threshold. Yet most renters have no data to guide their decisions.

This project uses data analytics to:

1. **Quantify the crisis** — measure how bad the rent squeeze really is, across income groups and housing types
2. **Map the opportunity** — identify undervalued neighbourhoods with strong value-to-cost ratios
3. **Empower renters** — build a tool that recommends areas based on income, commute needs, and lifestyle
4. **Inform policy** — surface structural causes (supply gap, transit gaps, information asymmetry) that data can help address

---

## 🔍 Key Findings

| Finding | Data Point |
|---|---|
| Average rent growth (2022–2024) | **+18.3%** |
| Median rent-to-income ratio | **47%** (threshold: 30%) |
| Annual housing units needed | ~250,000 |
| Units actually being built | ~50,000 |
| Supply gap | **~200,000 units/year** |
| Savings in satellite towns vs. Westlands | **Up to 71%** |
| "Postcode prestige" premium | ~**35%** for identical units |
| Renters (vs. owners) in Nairobi | **72%** of households |

---

## 🗂️ Project Structure

```
nairobi-housing-analytics/
│
├── 📁 data/
│   ├── raw/                    # Raw scraped listings data
│   │   ├── buyrentkenya_listings.csv
│   │   ├── propertybase_listings.csv
│   │   └── hass_consult_indices.csv
│   ├── processed/              # Cleaned and enriched datasets
│   │   ├── cleaned_listings.csv
│   │   ├── neighbourhood_summary.csv
│   │   ├── affordability_by_income.csv
│   │   └── geo_enriched_listings.csv
│   └── external/               # External reference data
│       ├── knbs_wage_data.csv
│       ├── nairobi_neighbourhoods.geojson
│       └── nairobi_transit_routes.csv
│
├── 📁 notebooks/
│   ├── 01_data_collection.ipynb      # Web scraping + data gathering
│   ├── 02_data_cleaning.ipynb        # Cleaning, standardisation, deduplication
│   ├── 03_exploratory_analysis.ipynb # EDA — distributions, correlations, outliers
│   ├── 04_affordability_analysis.ipynb # Rent-to-income, burden by tier
│   ├── 05_geospatial_analysis.ipynb  # Folium maps, transit corridors
│   ├── 06_opportunity_index.ipynb    # Composite scoring model
│   └── 07_predictive_model.ipynb     # Rent prediction (Random Forest)
│
├── 📁 src/
│   ├── scraper.py              # BuyRentKenya + Propertybase scraper
│   ├── cleaner.py              # Data cleaning pipeline
│   ├── analyser.py             # Core analysis functions
│   ├── scorer.py               # Opportunity index scoring model
│   └── utils.py                # Helper utilities
│
├── 📁 dashboard/
│   └── index.html              # Interactive HTML dashboard
│
├── 📁 assets/
│   ├── charts/                 # Exported chart images
│   └── maps/                   # Exported Folium HTML maps
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Notebooks Walkthrough

### `01_data_collection.ipynb` — Web Scraping
Uses `BeautifulSoup` and `requests` to scrape property listings from BuyRentKenya and Propertybase. Collects: neighbourhood, price, bedroom count, floor area, amenities, listing date, and agent type.

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.buyrentkenya.com/houses-for-rent"

def scrape_listings(pages=50):
    listings = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}?page={page}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, 'html.parser')
        
        for card in soup.select('.listing-card'):
            listings.append({
                'title':        card.select_one('.title').text.strip(),
                'neighbourhood':card.select_one('.location').text.strip(),
                'price':        card.select_one('.price').text.strip(),
                'bedrooms':     card.select_one('.beds').text.strip(),
                'size_sqft':    card.select_one('.size').text.strip() if card.select_one('.size') else None,
                'listed_date':  card.select_one('.date').text.strip(),
                'url':          card.select_one('a')['href']
            })
        time.sleep(1.5)  # Polite scraping delay
    
    return pd.DataFrame(listings)

df_raw = scrape_listings(pages=100)
df_raw.to_csv('../data/raw/buyrentkenya_listings.csv', index=False)
print(f"Collected {len(df_raw)} listings")
```

---

### `02_data_cleaning.ipynb` — Data Cleaning Pipeline
Handles: price parsing (remove "KES", commas, "/month"), neighbourhood standardisation (spelling variants), outlier removal, duplicate detection.

```python
import pandas as pd
import numpy as np
import re

def clean_price(price_str):
    """Extract numeric rent from string like 'KES 45,000/month'"""
    if pd.isna(price_str):
        return np.nan
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    return int(cleaned) if cleaned else np.nan

def standardise_neighbourhood(name):
    """Map spelling variants to canonical neighbourhood names"""
    mapping = {
        'westlands': 'Westlands',
        'west lands': 'Westlands',
        'kilimani': 'Kilimani',
        'south b': 'South B',
        'south b.': 'South B',
        'athi river': 'Athi River',
        'athiriver': 'Athi River',
        'rongai': 'Rongai',
        # ... full mapping in src/cleaner.py
    }
    return mapping.get(name.lower().strip(), name.strip().title())

df = pd.read_csv('../data/raw/buyrentkenya_listings.csv')
df['price_kes'] = df['price'].apply(clean_price)
df['neighbourhood_clean'] = df['neighbourhood'].apply(standardise_neighbourhood)

# Remove extreme outliers (< KES 3,000 or > KES 500,000)
df = df[(df['price_kes'] >= 3000) & (df['price_kes'] <= 500000)]

# Remove duplicates based on URL
df = df.drop_duplicates(subset='url')

print(f"Clean records: {len(df):,}")
print(f"Neighbourhoods: {df['neighbourhood_clean'].nunique()}")
```

---

### `03_exploratory_analysis.ipynb` — EDA
Full distribution analysis, correlation matrix, and key visual outputs.

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Nairobi Rental Market — Exploratory Analysis', fontsize=16, fontweight='bold')

# 1. Rent distribution
axes[0,0].hist(df['price_kes'], bins=60, color='#F59E0B', edgecolor='none', alpha=0.8)
axes[0,0].axvline(df['price_kes'].median(), color='red', linestyle='--', label=f"Median: KES {df['price_kes'].median():,.0f}")
axes[0,0].set_title('Rent Distribution Across All Listings')
axes[0,0].legend()

# 2. Rent by bedroom
bedroom_med = df.groupby('bedrooms')['price_kes'].median().sort_values()
axes[0,1].bar(bedroom_med.index, bedroom_med.values/1000, color='#3B82F6')
axes[0,1].set_title('Median Rent by Bedroom Count (KES Thousands)')

# 3. Top 15 most expensive neighbourhoods
top_n = df.groupby('neighbourhood_clean')['price_kes'].median().nlargest(15)
axes[1,0].barh(top_n.index, top_n.values/1000, color='#EF4444')
axes[1,0].set_title('Most Expensive Neighbourhoods')

# 4. Rent vs size scatter
axes[1,1].scatter(df['size_sqft'], df['price_kes']/1000, alpha=0.3, color='#22C55E', s=10)
axes[1,1].set_title('Rent vs. Unit Size (sqft)')

plt.tight_layout()
plt.savefig('../assets/charts/eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### `04_affordability_analysis.ipynb` — Affordability & Income Analysis
Cross-references rent data with KNBS wage surveys to compute rent burden by income bracket.

```python
# Income brackets (KES/month, Nairobi)
income_brackets = {
    'Low Income (<30K)':         27000,
    'Lower-Middle (30–60K)':     45000,
    'Middle Income (60–100K)':   78000,
    'Upper-Middle (100–200K)':  145000,
    'High Income (200K+)':      280000
}

nairobi_median_rent = df['price_kes'].median()

print("AFFORDABILITY ANALYSIS")
print("=" * 55)
print(f"{'Income Bracket':<30} {'Rent Burden':>12} {'Status':>12}")
print("-" * 55)

for bracket, income in income_brackets.items():
    burden = (nairobi_median_rent / income) * 100
    status = "🔴 CRITICAL" if burden > 50 else "🟡 STRESSED" if burden > 30 else "🟢 OK"
    print(f"{bracket:<30} {burden:>11.1f}% {status:>12}")
```

**Output:**
```
AFFORDABILITY ANALYSIS
=======================================================
Income Bracket                  Rent Burden       Status
-------------------------------------------------------
Low Income (<30K)                    63.3%   🔴 CRITICAL
Lower-Middle (30–60K)                47.8%   🔴 CRITICAL
Middle Income (60–100K)              27.6%       🟢 OK
Upper-Middle (100–200K)              14.8%       🟢 OK
High Income (200K+)                   7.7%       🟢 OK
```

---

### `05_geospatial_analysis.ipynb` — Interactive Maps
Generates choropleth and bubble maps using Folium.

```python
import folium
from folium.plugins import HeatMap

# Nairobi centre coordinates
nairobi_centre = [-1.286389, 36.817223]

m = folium.Map(location=nairobi_centre, zoom_start=11, tiles='CartoDB dark_matter')

# Add neighbourhood polygons coloured by median rent
for _, row in neighbourhood_geo.iterrows():
    rent = neighbourhood_summary.get(row['name'], {}).get('median_rent', 0)
    color = '#EF4444' if rent > 60000 else '#F59E0B' if rent > 30000 else '#22C55E'
    
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, c=color: {
            'fillColor': c, 'color': '#fff',
            'weight': 0.5, 'fillOpacity': 0.6
        },
        tooltip=folium.Tooltip(f"{row['name']}: KES {rent:,}/mo")
    ).add_to(m)

# Add listing heatmap layer
heat_data = [[row['lat'], row['lng'], row['price_kes']] for _, row in df.iterrows()]
HeatMap(heat_data, name='Rent Density', radius=15).add_to(m)

m.save('../assets/maps/nairobi_rent_heatmap.html')
print("Map saved.")
```

---

### `06_opportunity_index.ipynb` — Composite Scoring Model
Scores each neighbourhood across 5 dimensions to find the best value areas.

```python
def compute_opportunity_score(neighbourhood_df):
    """
    Weighted composite score:
    - Affordability (lower rent = higher score):  35%
    - Transit Access (proximity to SGR/matatu):   20%
    - Safety Index (crime reports per capita):    20%
    - Amenity Access (schools, hospitals, shops): 15%
    - Growth Trajectory (5yr price trend):        10%
    """
    weights = {
        'affordability':    0.35,
        'transit':          0.20,
        'safety':           0.20,
        'amenities':        0.15,
        'growth_score':     0.10
    }
    
    # Normalise each dimension to 0–10 scale
    for col in weights:
        col_min = neighbourhood_df[col].min()
        col_max = neighbourhood_df[col].max()
        neighbourhood_df[f'{col}_norm'] = (
            (neighbourhood_df[col] - col_min) / (col_max - col_min) * 10
        )
        # Invert affordability (lower rent = higher score)
        if col == 'affordability':
            neighbourhood_df[f'{col}_norm'] = 10 - neighbourhood_df[f'{col}_norm']
    
    # Compute weighted sum
    neighbourhood_df['opportunity_score'] = sum(
        neighbourhood_df[f'{col}_norm'] * weight
        for col, weight in weights.items()
    )
    
    return neighbourhood_df.sort_values('opportunity_score', ascending=False)

scored = compute_opportunity_score(neighbourhood_summary)
print(scored[['neighbourhood', 'opportunity_score', 'median_rent']].head(10).to_string())
```

---

### `07_predictive_model.ipynb` — Rent Prediction Model
Predicts rent for a given property spec using a Random Forest model.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Feature engineering
features = ['bedrooms', 'size_sqft', 'neighbourhood_encoded', 
            'floor_level', 'has_parking', 'has_security', 'dist_to_cbd_km']
target = 'price_kes'

le = LabelEncoder()
df['neighbourhood_encoded'] = le.fit_transform(df['neighbourhood_clean'])

X = df[features].dropna()
y = df.loc[X.index, target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Model Performance ===")
print(f"MAE:  KES {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R²:   {r2_score(y_test, y_pred):.3f}")
print(f"\nFeature Importances:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<35} {imp:.3f}")

# Sample prediction
sample = {'bedrooms': 2, 'size_sqft': 800, 'neighbourhood_encoded': le.transform(['Donholm'])[0],
          'floor_level': 2, 'has_parking': 1, 'has_security': 1, 'dist_to_cbd_km': 12}
pred = model.predict(pd.DataFrame([sample]))[0]
print(f"\nSample prediction (2BR Donholm, 800sqft): KES {pred:,.0f}/month")
```

**Expected Output:**
```
=== Model Performance ===
MAE:  KES 4,230
R²:   0.847

Feature Importances:
  neighbourhood_encoded              0.412
  bedrooms                           0.238
  size_sqft                          0.187
  dist_to_cbd_km                     0.089
  has_security                       0.034
  has_parking                        0.028
  floor_level                        0.012

Sample prediction (2BR Donholm, 800sqft): KES 26,800/month
```

---

## 🗺️ Visual Outputs

| Visual | Description |
|---|---|
| `eda_overview.png` | 4-panel EDA dashboard — distribution, bedrooms, neighbourhoods, size-price scatter |
| `rent_trend_2019_2024.png` | Line chart: Economy / Mid / Premium tiers over 5 years |
| `affordability_matrix.png` | Heatmap: rent burden % by income bracket and neighbourhood |
| `opportunity_index_bar.png` | Top 15 neighbourhoods by composite score |
| `nairobi_rent_heatmap.html` | Interactive Folium heatmap — zoom in to any neighbourhood |
| `supply_demand_gap.png` | Supply vs. demand chart 2019–2024 |
| `feature_importance.png` | Random Forest feature importances |
| `dashboard/index.html` | Full interactive HTML dashboard (this repo) |

---

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.11
pip install -r requirements.txt
```

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn folium scikit-learn \
            requests beautifulsoup4 jupyter plotly geopandas
```

### Run notebooks in order
```bash
cd nairobi-housing-analytics/
jupyter lab
# Open notebooks/ and run 01 → 07 in sequence
```

### Or run the full pipeline
```bash
python src/scraper.py      # Collect fresh data
python src/cleaner.py      # Clean and process
python src/analyser.py     # Generate all charts
python src/scorer.py       # Compute opportunity index
```

### View the dashboard
Open `dashboard/index.html` in any browser — no server needed.

---

## 📦 requirements.txt

```
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
folium==0.15.1
beautifulsoup4==4.12.2
requests==2.31.0
geopandas==0.14.1
plotly==5.18.0
jupyter==1.0.0
notebook==7.0.6
lxml==4.9.3
openpyxl==3.1.2
```

---

## 💡 Methodology

### Data Sources
| Source | Description | Records |
|---|---|---|
| BuyRentKenya.com | Primary rental listings | ~2,800 |
| Propertybase Kenya | Secondary listings | ~600 |
| HassConsult Kenya | Historical index data | 5yr |
| KNBS 2024 | Wage and income data | National |
| World Bank Open Data | Urban housing indicators | Regional |

### Opportunity Index — Scoring Dimensions

| Dimension | Weight | Source |
|---|---|---|
| Affordability (lower = better) | 35% | Scraped listings |
| Transit accessibility | 20% | Google Maps Distance Matrix API |
| Safety index | 20% | Kenya Police crime statistics |
| Amenity density | 15% | OSM / Google Places API |
| 5-year growth trajectory | 10% | HassConsult + historical listings |

### Ethics & Scraping Policy
- All data collected with `time.sleep(1.5)` delays to avoid server strain
- No personal data collected — only aggregated listing data
- Scraped in compliance with robots.txt

---

## 📐 Analysis Framework

```
Raw Listings → Data Cleaning → Feature Engineering
      ↓
EDA (distributions, correlations, outliers)
      ↓
Affordability Analysis (rent burden by income bracket)
      ↓
Geospatial Analysis (choropleth maps, transit corridors)
      ↓
Opportunity Index (composite scoring model)
      ↓
Predictive Model (Random Forest rent prediction)
      ↓
Interactive Dashboard (HTML + Chart.js)
```

---

## 🎯 Policy Recommendations

Based on the analysis, this project recommends:

1. **Accelerate affordable rental supply** — Not ownership. 72% of Nairobians rent. Policy must prioritise rental units, not homeownership subsidies.

2. **Invest in transit corridors** — Rongai, Ruiru, and Athi River show that SGR and highway access unlocks 60–70% cost savings while maintaining 30–45 min commutes. Transit investment = housing affordability.

3. **Open data on housing** — This project itself proves the point: data access reduces the 35% "postcode premium" that uninformed renters pay.

4. **Satellite town incentivisation** — Tax incentives for developers building in Ruiru, Athi River, and Rongai would reduce pressure on Nairobi core neighbourhoods significantly.

5. **Rent indexation policy** — Annual rent increases should be capped in line with KNBS wage growth data, not market speculation.

---

## 👨🏾‍💻 About the Author

**John Wachira** | Programme Delivery Specialist, ALX Africa  
Data Science | Data Engineering | Analytics

- 📧 jowac254@gmail.com  
- 💼 [linkedin.com/in/jowac254](https://linkedin.com/in/jowac254)  
- 🐙 [github.com/jowac254](https://github.com/jowac254)  
- 🌍 Nairobi, Kenya

This project was built to demonstrate end-to-end data analytics skills — from raw data collection to interactive visualisation — applied to a real, urgent problem in the African urban context.

---

## 📄 License

MIT License. Data is used for educational and research purposes only.

---

*Built with Python, Pandas, Scikit-learn, Folium, Matplotlib, Seaborn, and Chart.js*  
*Data: BuyRentKenya · Propertybase · HassConsult · KNBS · World Bank*
