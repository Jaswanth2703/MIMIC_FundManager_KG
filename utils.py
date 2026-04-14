"""
Shared utilities for the Fund Manager KG pipeline.
"""

import pandas as pd
import numpy as np
import os
import time


def safe_name(s, max_len=15):
    """Create a safe variable name from a string."""
    return str(s).replace(' ', '_').replace('&', 'n').replace('/', '_')[:max_len]


def clean_record(record):
    """Convert numpy types to Python natives, replace NaN with None.
    Used for Neo4j ingestion.
    """
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, (np.int64, np.int32)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)):
            cleaned[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
        elif isinstance(v, pd.Period):
            cleaned[k] = str(v)
        elif pd.isna(v) if not isinstance(v, str) else False:
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


def standardize_industry(raw):
    """Map ~150 raw industry names to ~20 SEBI-aligned sectors.

    Uses keyword matching with priority ordering to handle
    inconsistent casing, spelling, and truncated entries.
    """
    if pd.isna(raw):
        return 'OTHERS'

    s = str(raw).strip().strip('"').upper()

    if s in ('', 'N.A.', 'INDUSTRY', 'NOT CLASSIFIED', 'MISCELLANEOUS'):
        return 'OTHERS'

    # 1. Debt & Money Market instruments (most specific)
    if any(kw in s for kw in ['SOVEREIGN', 'A1+', 'TREASURY', 'CBLO', 'TREPS']):
        return 'DEBT & MONEY MARKET'
    # 2. IT & Software (before generic 'SERVICE' match)
    if any(kw in s for kw in ['SOFTWARE', 'IT ENABLED', 'SOFTWARE PRODUCT']):
        return 'INFORMATION TECHNOLOGY'
    if s.startswith('IT') and any(kw in s for kw in ['SERVICE', 'HARDWARE']):
        return 'INFORMATION TECHNOLOGY'
    # 3. Telecom
    if 'TELECOM' in s:
        return 'TELECOM'
    # 4. Healthcare & Pharma
    if any(kw in s for kw in ['PHARMA', 'HEALTH', 'BIOTECH', 'HOSPITAL']):
        return 'HEALTHCARE'
    # 5. Financial Services
    if any(kw in s for kw in ['BANK', 'FINANC', 'INSURANCE', 'CAPITAL MARKET', 'FINTECH']):
        return 'FINANCIAL SERVICES'
    # 6. Automobile
    if any(kw in s for kw in ['AUTO', 'VEHICLE', 'MOTOR']):
        return 'AUTOMOBILE'
    # 7. FMCG
    if any(kw in s for kw in ['CONSUMER NON', 'FMCG', 'FOOD', 'BEVERAGE',
                               'TOBACCO', 'CIGARETTE', 'PERSONAL PRODUCT',
                               'HOUSEHOLD PRODUCT']):
        return 'FMCG'
    # 8. Consumer Discretionary
    if any(kw in s for kw in ['CONSUMER DURABLE', 'CONSUMER SERVICE',
                               'ENTERTAINMENT', 'HOTEL', 'LEISURE',
                               'RETAILING', 'RETAIL']):
        return 'CONSUMER DISCRETIONARY'
    # 9. Energy
    if any(kw in s for kw in ['PETROLEUM', 'POWER', 'CONSUMABLE FUEL', 'UTILIT']):
        return 'ENERGY'
    if s in ('OIL', 'GAS') or s.startswith('OIL') or s.startswith('GAS'):
        return 'ENERGY'
    # 10. Metals & Mining
    if any(kw in s for kw in ['METAL', 'MINING', 'MINERAL', 'FERROUS']):
        return 'METALS & MINING'
    # 11. Chemicals
    if any(kw in s for kw in ['CHEMICAL', 'PETROCHEMICAL']):
        return 'CHEMICALS'
    # 12. Cement & Construction
    if any(kw in s for kw in ['CEMENT', 'CONSTRUCT']):
        return 'CEMENT & CONSTRUCTION'
    # 13. Industrials
    if any(kw in s for kw in ['INDUSTRIAL', 'CAPITAL GOOD', 'ELECTRICAL EQUIP',
                               'ENGINEERING']):
        return 'INDUSTRIALS'
    # 14. Textiles
    if any(kw in s for kw in ['TEXTILE', 'APPAREL']):
        return 'TEXTILES'
    # 15. Realty
    if any(kw in s for kw in ['REALTY', 'REAL ESTATE']):
        return 'REALTY'
    # 16. Agriculture
    if any(kw in s for kw in ['AGRI', 'FERTILI', 'PESTICIDE', 'PAPER',
                               'FOREST', 'JUTE']):
        return 'AGRICULTURE'
    # 17. Transport
    if any(kw in s for kw in ['TRANSPORT', 'LOGISTICS', 'SHIPPING']):
        return 'TRANSPORT'
    # 18. Aerospace & Defense
    if any(kw in s for kw in ['AEROSPACE', 'DEFEN']):
        return 'AEROSPACE & DEFENSE'
    # 19. Media
    if 'MEDIA' in s:
        return 'MEDIA'
    # 20. Services
    if any(kw in s for kw in ['SERVICE', 'TRADING', 'COMMERCIAL']):
        return 'SERVICES'
    # 21. Others
    if 'DIVERSIFIED' in s:
        return 'OTHERS'

    return 'OTHERS'


def rate_limit_sleep(calls_per_sec=3):
    """Sleep to respect API rate limits."""
    time.sleep(1.0 / calls_per_sec)


def coverage_report(df, exclude_cols=None):
    """Print feature coverage (% non-null) for a DataFrame."""
    exclude = set(exclude_cols or [])
    exclude.update(['Date', 'Fund_Name', 'Fund_Type', 'ISIN', 'stock_name_raw',
                    'stock_name', 'Industry', 'year_month_str', 'sector',
                    'date', 'symbol', 'holding_period_id'])
    print("\nFeature Coverage:")
    for col in sorted(df.columns):
        if col not in exclude:
            pct = df[col].notna().mean() * 100
            if pct > 0:
                print(f"  {col:35s}: {pct:5.1f}%")


def quarter_from_month(year_month_str):
    """Convert '2024-03' to quarter string 'Q1-2024'."""
    year, month = int(year_month_str[:4]), int(year_month_str[5:7])
    q = (month - 1) // 3 + 1
    return f"Q{q}-{year}"


def fiscal_year_from_month(year_month_str):
    """Convert '2024-03' to Indian fiscal year 'FY2024'."""
    year, month = int(year_month_str[:4]), int(year_month_str[5:7])
    fy = year if month >= 4 else year - 1
    return f"FY{fy}"
