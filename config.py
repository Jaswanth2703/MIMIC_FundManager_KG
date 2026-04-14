"""
Central configuration for the Fund Manager KG pipeline.
All paths, API keys, and constants in one place.
"""

import os

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use system env vars

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR  # everything is now self-contained inside here
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

# Input data (from existing project)
MASTER_CSV = os.path.join(PROJECT_ROOT, 'MASTER_CONSOLIDATED_CLEAN_FINAL.csv')
ISIN_MAPPING_CSV = os.path.join(DATASET_DIR, 'isin_symbol_mapping.csv')
TICKER_LIST_CSV = os.path.join(DATASET_DIR, 'Fundamentals(1)', 'Fundamentals', 'Ticker_List_NSE_India.csv')
NEWS_DIR = os.path.join(PROJECT_ROOT, 'final_news')

# CMIE Fundamentals
FUNDAMENTALS_DIR = os.path.join(DATASET_DIR, 'Fundamentals(1)', 'Fundamentals')
FUNDAMENTALS_CSV = os.path.join(FUNDAMENTALS_DIR, 'all_fundamentals_monthly_with_tickers.csv')

# Macro data files (local)
GOLD_USD_CSV = os.path.join(DATASET_DIR, 'test', 'gold_monthly_202209_202509.csv')
GOLD_INR_CSV = os.path.join(DATASET_DIR, 'test', 'gold_monthly_inr_per_10g_202209_202509.csv')
INTEREST_RATE_XLS = os.path.join(DATASET_DIR, 'test', 'Interest rate.xls')
GDP_XLS = os.path.join(DATASET_DIR, 'test', 'GDP.xls')
INFLATION_XLS = os.path.join(DATASET_DIR, 'test', 'Inflation.xls')

# Existing processed data (from old pipeline — used as fallback / reference)
EXISTING_TEMPORAL_CSV = os.path.join(DATASET_DIR, 'TEMPORAL_KG_READY.csv')
EXISTING_CAUSAL_CSV = os.path.join(DATASET_DIR, 'CAUSAL_FEATURES_READY.csv')
EXISTING_EXITS_CSV = os.path.join(DATASET_DIR, 'EXIT_EVENTS.csv')
EXISTING_MACRO_CSV = os.path.join(DATASET_DIR, 'macro_indicators.csv')
EXISTING_SENTIMENT_CSV = os.path.join(DATASET_DIR, 'news_sentiment.csv')

# Output directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
MAPPINGS_DIR = os.path.join(DATA_DIR, 'mappings')
PORTFOLIO_DIR = os.path.join(DATA_DIR, 'portfolio')
MARKET_DIR = os.path.join(DATA_DIR, 'market')
SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment')
MACRO_DIR = os.path.join(DATA_DIR, 'macro')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
CAUSAL_DIR = os.path.join(DATA_DIR, 'causal_output')
EVAL_DIR = os.path.join(DATA_DIR, 'evaluation')
FINAL_DIR = os.path.join(DATA_DIR, 'final')

# Ensure all output dirs exist
for d in [MAPPINGS_DIR, PORTFOLIO_DIR, MARKET_DIR, SENTIMENT_DIR,
          MACRO_DIR, FEATURES_DIR, CAUSAL_DIR, EVAL_DIR, FINAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# API KEYS  (set as environment variables — DO NOT hardcode for GitHub)
# ============================================================
# Kite Connect (Zerodha) — access token expires daily
KITE_API_KEY = os.environ.get('KITE_API_KEY', '')
KITE_ACCESS_TOKEN = os.environ.get('KITE_ACCESS_TOKEN', '')

# Gemini (Google AI)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# Neo4j
NEO4J_URI = os.environ.get('NEO4J_URI', 'neo4j://127.0.0.1:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '')

# ============================================================
# PIPELINE CONSTANTS
# ============================================================
DATE_START = '2022-01-01'
DATE_END = '2025-10-01'
DATE_RANGE_STR = ('2022-01', '2025-10')

GAP_THRESHOLD = 3         # months — position gap tolerance
BATCH_SIZE = 500           # Neo4j batch import size

# Causal discovery
TAU_MAX = 6                # month lookback for LPCMCI
ALPHA_STRICT = 0.01        # conservative significance
ALPHA_EXPLORE = 0.05       # exploratory significance
MIN_MONTHS = 10            # minimum data points per variable
BOOTSTRAP_ENABLED = True   # enabled for GPU machine (9800X3D+3070)
BOOTSTRAP_N = 100          # number of bootstrap samples
BOOTSTRAP_THRESHOLD = 0.7  # minimum edge frequency to keep

# Kite API rate limit
KITE_RATE_LIMIT = 3        # requests per second

# FinBERT
FINBERT_MODEL = 'ProsusAI/finbert'
FINBERT_BATCH_SIZE = 32

# Macro indices — Kite instrument search tokens (NSE index names)
KITE_INDEX_MAP = {
    'NIFTY 50': 'nifty50',
    'NIFTY BANK': 'nifty_bank',
    'NIFTY IT': 'nifty_it',
    'NIFTY PHARMA': 'nifty_pharma',
    'NIFTY AUTO': 'nifty_auto',
    'NIFTY FMCG': 'nifty_fmcg',
    'NIFTY METAL': 'nifty_metal',
    'NIFTY REALTY': 'nifty_realty',
    'NIFTY ENERGY': 'nifty_energy',
    'INDIA VIX': 'india_vix',
}

# Global macro from yfinance (acceptable for non-Indian data)
YFINANCE_GLOBAL_MAP = {
    'crude_oil': 'CL=F',
    'brent_crude': 'BZ=F',
    'gold_usd': 'GC=F',
    'usd_inr': 'INR=X',
    'us_10y_yield': '^TNX',
    'sp500': '^GSPC',
}

# SEBI-aligned sectors (21 categories)
SEBI_SECTORS = [
    'DEBT & MONEY MARKET', 'INFORMATION TECHNOLOGY', 'TELECOM',
    'HEALTHCARE', 'FINANCIAL SERVICES', 'AUTOMOBILE', 'FMCG',
    'CONSUMER DISCRETIONARY', 'ENERGY', 'METALS & MINING', 'CHEMICALS',
    'CEMENT & CONSTRUCTION', 'INDUSTRIALS', 'TEXTILES', 'REALTY',
    'AGRICULTURE', 'TRANSPORT', 'AEROSPACE & DEFENSE', 'MEDIA',
    'SERVICES', 'OTHERS',
]

# Hardcoded RBI macro (changes infrequently)
REPO_RATE_MONTHLY = {
    '2022-01': 4.00, '2022-02': 4.00, '2022-03': 4.00, '2022-04': 4.00,
    '2022-05': 4.40, '2022-06': 4.90, '2022-07': 4.90, '2022-08': 5.40,
    '2022-09': 5.40, '2022-10': 5.90, '2022-11': 5.90, '2022-12': 6.25,
    '2023-01': 6.25, '2023-02': 6.50, '2023-03': 6.50, '2023-04': 6.50,
    '2023-05': 6.50, '2023-06': 6.50, '2023-07': 6.50, '2023-08': 6.50,
    '2023-09': 6.50, '2023-10': 6.50, '2023-11': 6.50, '2023-12': 6.50,
    '2024-01': 6.50, '2024-02': 6.50, '2024-03': 6.50, '2024-04': 6.50,
    '2024-05': 6.50, '2024-06': 6.50, '2024-07': 6.50, '2024-08': 6.50,
    '2024-09': 6.50, '2024-10': 6.50, '2024-11': 6.50, '2024-12': 6.50,
    '2025-01': 6.50, '2025-02': 6.25, '2025-03': 6.25, '2025-04': 6.00,
    '2025-05': 6.00, '2025-06': 6.00, '2025-07': 6.00, '2025-08': 6.00,
    '2025-09': 6.00, '2025-10': 6.00,
}

CPI_INFLATION_MONTHLY = {
    '2022-01': 6.01, '2022-02': 6.07, '2022-03': 6.95, '2022-04': 7.79,
    '2022-05': 7.04, '2022-06': 7.01, '2022-07': 6.71, '2022-08': 7.00,
    '2022-09': 7.41, '2022-10': 6.77, '2022-11': 5.88, '2022-12': 5.72,
    '2023-01': 6.52, '2023-02': 6.44, '2023-03': 5.66, '2023-04': 4.70,
    '2023-05': 4.25, '2023-06': 4.81, '2023-07': 7.44, '2023-08': 6.83,
    '2023-09': 5.02, '2023-10': 4.87, '2023-11': 5.55, '2023-12': 5.69,
    '2024-01': 5.10, '2024-02': 5.09, '2024-03': 4.85, '2024-04': 4.83,
    '2024-05': 4.75, '2024-06': 5.08, '2024-07': 3.54, '2024-08': 3.65,
    '2024-09': 5.49, '2024-10': 6.21, '2024-11': 5.48, '2024-12': 5.22,
    '2025-01': 4.31, '2025-02': 3.61, '2025-03': 3.34, '2025-04': 3.16,
    '2025-05': 4.00, '2025-06': 4.00, '2025-07': 4.00, '2025-08': 4.00,
    '2025-09': 4.00, '2025-10': 4.00,
}

GDP_GROWTH_QUARTERLY = {
    'Q1-2022': 13.1, 'Q2-2022': 6.2, 'Q3-2022': 4.5, 'Q4-2022': 6.1,
    'Q1-2023': 7.8, 'Q2-2023': 7.6, 'Q3-2023': 8.4, 'Q4-2023': 7.8,
    'Q1-2024': 6.7, 'Q2-2024': 5.4, 'Q3-2024': 6.2, 'Q4-2024': 6.2,
    'Q1-2025': 6.3, 'Q2-2025': 6.5, 'Q3-2025': 6.5, 'Q4-2025': 6.5,
}
