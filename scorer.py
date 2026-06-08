"""
scorer.py — Neighbourhood Opportunity Index
Computes a weighted composite score for each Nairobi neighbourhood.
Author: John Wachira | github.com/jowac254
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# SCORING WEIGHTS
# ============================================================
WEIGHTS = {
    'affordability':    0.35,   # Lower rent relative to metro = higher score
    'transit':          0.20,   # Quality of public transport access
    'safety':           0.20,   # Crime index (lower = safer = higher score)
    'amenities':        0.15,   # Schools, hospitals, supermarkets density
    'growth_score':     0.10,   # 5yr rent growth trajectory (moderate = best)
}

# ============================================================
# RAW SCORES PER NEIGHBOURHOOD (0–10 scale, pre-normalisation)
# Source: Hass Consult reports, Kenya Police stats, OSM, field research
# ============================================================
NEIGHBOURHOOD_SCORES = {
    #                        afford  transit  safety  amenities  growth
    'Westlands':            [2,      10,      9,      10,        6],
    'Kilimani':             [2,       9,      9,       9,        7],
    'Karen':                [1,       5,     10,       7,        4],
    'Kileleshwa':           [2,       8,      9,       8,        6],
    'Lavington':            [2,       7,      9,       8,        5],
    'Parklands':            [3,       9,      8,       9,        6],
    'Hurlingham':           [3,       8,      8,       8,        6],
    'Upper Hill':           [3,       9,      7,       8,        7],
    'South B':              [6,       9,      7,       8,        6],
    'South C':              [6,       9,      7,       8,        6],
    'Langata':              [6,       6,      7,       7,        6],
    'Donholm':              [7,       8,      6,       7,        8],
    'Umoja':                [7,       7,      6,       7,        6],
    'Kayole':               [8,       6,      5,       6,        5],
    'Rongai':               [8,       6,      8,       6,        9],
    'Athi River':           [9,       8,      8,       6,        9],
    'Ruiru':                [9,       8,      7,       7,        10],
    'Ruaka':                [8,       7,      7,       7,        9],
    'Githurai':             [10,      6,      4,       6,        6],
    'Kahawa West':          [9,       6,      6,       7,        8],
    'Ruai':                 [10,      4,      6,       5,        5],
    'Thika Road':           [8,       7,      7,       7,        8],
}

SCORE_COLS = ['affordability', 'transit', 'safety', 'amenities', 'growth_score']


def build_score_dataframe() -> pd.DataFrame:
    """Build the base scores DataFrame from NEIGHBOURHOOD_SCORES."""
    rows = []
    for neighbourhood, scores in NEIGHBOURHOOD_SCORES.items():
        row = {'neighbourhood': neighbourhood}
        row.update(dict(zip(SCORE_COLS, scores)))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_opportunity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted opportunity score for each neighbourhood.
    
    Returns DataFrame sorted by composite score (descending).
    """
    df = df.copy()

    # Weighted sum
    df['opportunity_score'] = sum(
        df[col] * weight
        for col, weight in WEIGHTS.items()
    )

    # Round to 1 decimal
    df['opportunity_score'] = df['opportunity_score'].round(1)

    # Add tier labels
    df['tier'] = pd.cut(
        df['opportunity_score'],
        bins=[0, 5.5, 7.0, 8.0, 10],
        labels=['Low Value', 'Average', 'Good Value', 'Excellent Value']
    )

    return df.sort_values('opportunity_score', ascending=False).reset_index(drop=True)


def merge_with_rent_data(score_df: pd.DataFrame,
                          summary_path: str = "../data/processed/neighbourhood_summary.csv") -> pd.DataFrame:
    """Merge opportunity scores with real median rent data."""
    summary_file = Path(summary_path)
    if not summary_file.exists():
        logger.warning("neighbourhood_summary.csv not found — using placeholder rents")
        placeholder = {n: np.random.randint(12000, 95000) for n in NEIGHBOURHOOD_SCORES}
        score_df['median_rent'] = score_df['neighbourhood'].map(placeholder)
        return score_df

    summary = pd.read_csv(summary_file)
    merged = score_df.merge(
        summary[['neighbourhood_clean', 'median_rent', 'listing_count']],
        left_on='neighbourhood',
        right_on='neighbourhood_clean',
        how='left'
    ).drop(columns='neighbourhood_clean', errors='ignore')
    return merged


def plot_opportunity_index(df: pd.DataFrame, output_dir: str = "../assets/charts") -> None:
    """Generate a polished horizontal bar chart of opportunity scores."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Colour by tier
    tier_colours = {
        'Excellent Value': '#22C55E',
        'Good Value':      '#F59E0B',
        'Average':         '#3B82F6',
        'Low Value':       '#EF4444',
    }
    colours = df['tier'].map(tier_colours).fillna('#A1A1AA')

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0A0A0B')
    ax.set_facecolor('#111113')

    y_pos = range(len(df))
    bars = ax.barh(y_pos, df['opportunity_score'], color=colours, edgecolor='none', height=0.7)

    # Labels inside bars
    for bar, score in zip(bars, df['opportunity_score']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{score:.1f}', va='center', ha='left', fontsize=9,
                color='#F4F4F5', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['neighbourhood'], fontsize=10, color='#A1A1AA')
    ax.set_xlabel('Opportunity Score (0–10)', color='#71717A', fontsize=11)
    ax.set_title('Nairobi Neighbourhood Opportunity Index\n'
                 'Weighted: Affordability 35% · Transit 20% · Safety 20% · Amenities 15% · Growth 10%',
                 color='#F4F4F5', fontsize=13, fontweight='bold', pad=16)

    ax.axvline(7.0, color='#F59E0B', linewidth=0.8, linestyle='--', alpha=0.6, label='Good Value threshold')
    ax.tick_params(colors='#71717A')
    ax.spines['bottom'].set_color('#222226')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#222226')
    ax.set_xlim(0, 11)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=c, label=t) for t, c in tier_colours.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=9,
              facecolor='#18181B', edgecolor='#222226', labelcolor='#A1A1AA')

    plt.tight_layout()
    out_path = Path(output_dir) / "opportunity_index.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info(f"Chart saved → {out_path}")
    plt.close()


def print_summary_report(df: pd.DataFrame) -> None:
    """Print a formatted console summary."""
    print("\n" + "═" * 68)
    print("  NAIROBI NEIGHBOURHOOD OPPORTUNITY INDEX")
    print("═" * 68)
    print(f"  {'Rank':<5} {'Neighbourhood':<22} {'Score':>6} {'Tier':<18} {'Rent (est)'}")
    print("─" * 68)
    for i, row in df.iterrows():
        rent_str = f"KES {row.get('median_rent', 0):,.0f}" if pd.notna(row.get('median_rent')) else "—"
        tier = str(row['tier'])
        print(f"  {i+1:<5} {row['neighbourhood']:<22} {row['opportunity_score']:>5.1f}  {tier:<18} {rent_str}")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s — %(message)s')

    df_scores = build_score_dataframe()
    df_scored = compute_opportunity_index(df_scores)
    df_final  = merge_with_rent_data(df_scored)

    print_summary_report(df_final)
    plot_opportunity_index(df_final)

    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_dir / "opportunity_index.csv", index=False)
    logger.info("Opportunity index saved.")
