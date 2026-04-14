"""
Step 10 -- Build Temporal Knowledge Graph in Neo4j
====================================================
Creates the temporal KG with Fund, Stock, Sector, TimePeriod,
FundSnapshot, StockSnapshot, and MarketRegime nodes plus all
relationships.

Schema
------
NODES:
  (:Fund {name, type, aum_avg})
  (:Stock {isin, name, symbol, sector})
  (:Sector {name})
  (:TimePeriod {id, year, month, quarter, fiscal_year})
  (:FundSnapshot {fund_name, month, total_stocks, total_nav})
  (:StockSnapshot {isin, month, pe, pb, eps, beta, market_cap, rsi, macd,
                   sentiment_mean, monthly_return, volatility})
  (:MarketRegime {id, regime_type, vix_level, nifty_trend})

RELATIONSHIPS:
  (Fund)-[:HOLDS {month, pct_nav, quantity, market_value, position_action,
           allocation_change, size, holding_tenure, rank, consensus}]->(Stock)
  (Fund)-[:EXITED {month, last_pct_nav, holding_tenure}]->(Stock)
  (Stock)-[:BELONGS_TO]->(Sector)
  (TimePeriod)-[:NEXT]->(TimePeriod)
  (Fund)-[:ACTIVE_IN]->(TimePeriod)
  (FundSnapshot)-[:OF_FUND]->(Fund)
  (FundSnapshot)-[:AT_TIME]->(TimePeriod)
  (StockSnapshot)-[:OF_STOCK]->(Stock)
  (StockSnapshot)-[:AT_TIME]->(TimePeriod)
  (TimePeriod)-[:IN_REGIME]->(MarketRegime)

Input : data/portfolio/portfolio_clean.csv  (step 01)
        data/portfolio/exit_events.csv      (step 01)
        data/features/LPCMCI_READY.csv      (step 08, optional enrichment)
        data/macro/macro_monthly.csv        (step 06, optional for regimes)
Output: Neo4j temporal KG
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FEATURES_DIR, CAUSAL_DIR, PORTFOLIO_DIR, EVAL_DIR, FINAL_DIR, MAPPINGS_DIR,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BATCH_SIZE,
    TAU_MAX, ALPHA_STRICT, ALPHA_EXPLORE, MIN_MONTHS,
    BOOTSTRAP_ENABLED, BOOTSTRAP_N, BOOTSTRAP_THRESHOLD,
    SEBI_SECTORS, DATE_RANGE_STR,
    EXISTING_TEMPORAL_CSV, EXISTING_EXITS_CSV, MACRO_DIR,
)
from utils import clean_record, standardize_industry, quarter_from_month, fiscal_year_from_month

import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from neo4j import GraphDatabase


# ============================================================
# Input file resolution
# ============================================================
PORTFOLIO_CLEAN_CSV = os.path.join(PORTFOLIO_DIR, 'portfolio_clean.csv')
EXIT_EVENTS_CSV = os.path.join(PORTFOLIO_DIR, 'exit_events.csv')
FEATURES_CSV = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
FEATURES_FALLBACK = os.path.join(FEATURES_DIR, 'CAUSAL_FEATURES_COMPLETE.csv')
MACRO_CSV = os.path.join(MACRO_DIR, 'macro_indicators_monthly.csv')


# ============================================================
# Market Regime Classification
# ============================================================
def classify_market_regime(vix, nifty_return):
    """
    Classify a month into a market regime based on India VIX and
    Nifty 50 monthly return.

    Supports both raw values and z-score standardised values
    (detected automatically):

    Raw VIX level (> 10):
        > 25  => HIGH_VOLATILITY   (fear/crisis)
        > 15  => MODERATE_VOLATILITY
        else  => LOW_VOLATILITY    (complacency)

    Z-scored VIX (values typically in [-3, 5]):
        > +0.75σ  => HIGH_VOLATILITY
        < -0.50σ  => LOW_VOLATILITY
        else      => MODERATE_VOLATILITY

    Raw Nifty return (%):
        > +3%  => BULL
        < -3%  => BEAR
        else   => SIDEWAYS

    Z-scored Nifty return (values typically in [-3, 3]):
        > +0.75σ  => BULL    (~top 25% monthly returns)
        < -0.75σ  => BEAR    (~bottom 25%)
        else      => SIDEWAYS
    """
    # --- VIX classification ---
    if vix is None or (isinstance(vix, float) and np.isnan(vix)):
        vix_level = 'UNKNOWN'
    else:
        # Auto-detect z-scored vs raw: raw VIX is always > 10
        if abs(vix) <= 8:
            # Z-scored
            if vix > 0.75:
                vix_level = 'HIGH_VOLATILITY'
            elif vix < -0.50:
                vix_level = 'LOW_VOLATILITY'
            else:
                vix_level = 'MODERATE_VOLATILITY'
        else:
            # Raw VIX level
            if vix > 25:
                vix_level = 'HIGH_VOLATILITY'
            elif vix > 15:
                vix_level = 'MODERATE_VOLATILITY'
            else:
                vix_level = 'LOW_VOLATILITY'

    # --- Nifty trend classification ---
    if nifty_return is None or (isinstance(nifty_return, float) and np.isnan(nifty_return)):
        nifty_trend = 'UNKNOWN'
    else:
        # Auto-detect z-scored vs raw: raw monthly % return rarely exceeds ±20
        if abs(nifty_return) <= 10:
            # Could be z-scored (range ~[-3,3]) or small raw % return
            # Use ±0.75 for z-scored, ±3 for raw-like values
            threshold = 0.75 if abs(nifty_return) <= 4 else 3.0
        else:
            threshold = 3.0  # definitely raw %

        if nifty_return > threshold:
            nifty_trend = 'BULL'
        elif nifty_return < -threshold:
            nifty_trend = 'BEAR'
        else:
            nifty_trend = 'SIDEWAYS'

    regime_type = f"{vix_level}_{nifty_trend}"
    return regime_type, vix_level, nifty_trend


class TemporalKGBuilder:
    """Builds the temporal knowledge graph in Neo4j."""

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        with self.driver.session() as session:
            session.run("RETURN 1 AS test").single()
        print("  Neo4j connected.")
        self.stats = defaultdict(int)

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, cypher, params=None):
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return result.consume()

    def _run_data(self, cypher, params=None):
        """Execute and return data rows."""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return result.data()

    def _batch_run(self, cypher, records, batch_size=BATCH_SIZE):
        """Execute Cypher in batches."""
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            with self.driver.session() as session:
                session.run(cypher, {'batch': batch})
            total += len(batch)
        return total

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------
    def create_constraints(self):
        """Create uniqueness constraints and indexes."""
        print("\n  Creating constraints and indexes...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Fund) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Stock) REQUIRE s.isin IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sec:Sector) REQUIRE sec.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TimePeriod) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (mr:MarketRegime) REQUIRE mr.id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (s:Stock) ON (s.symbol)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Stock) ON (s.sector)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Fund) ON (f.type)",
            "CREATE INDEX IF NOT EXISTS FOR (t:TimePeriod) ON (t.year)",
            "CREATE INDEX IF NOT EXISTS FOR (fs:FundSnapshot) ON (fs.fund_name, fs.month)",
            "CREATE INDEX IF NOT EXISTS FOR (ss:StockSnapshot) ON (ss.isin, ss.month)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[h:HOLDS]-() ON (h.month)",
        ]
        for stmt in constraints + indexes:
            try:
                self._run(stmt)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    print(f"    Warning: {e}")
        print("  Constraints and indexes created.")

    def clear_graph(self):
        """Remove all nodes and relationships (use with caution)."""
        print("  Clearing existing graph...")
        self._run("MATCH (n) DETACH DELETE n")
        print("  Graph cleared.")

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------
    def create_sectors(self, df):
        """Create Sector nodes."""
        if 'sector' not in df.columns:
            print("  No sector column; skipping sector nodes.")
            return

        sectors = df['sector'].dropna().unique().tolist()
        records = [{'name': s} for s in sectors]
        cypher = """
        UNWIND $batch AS row
        MERGE (sec:Sector {name: row.name})
        """
        n = self._batch_run(cypher, records)
        self.stats['Sector'] = n
        print(f"  Created {n} Sector nodes")

    def create_funds(self, df):
        """Create Fund nodes."""
        fund_info = df.groupby('Fund_Name').agg({
            'Fund_Type': 'first',
        }).reset_index()

        # Compute AUM average (sum of pct_nav as proxy)
        aum_avg = df.groupby('Fund_Name')['pct_nav'].mean().reset_index()
        aum_avg.columns = ['Fund_Name', 'aum_avg']
        fund_info = fund_info.merge(aum_avg, on='Fund_Name', how='left')

        records = []
        for _, row in fund_info.iterrows():
            records.append(clean_record({
                'name': row['Fund_Name'],
                'type': row.get('Fund_Type', 'Unknown'),
                'aum_avg': row.get('aum_avg', 0),
            }))

        cypher = """
        UNWIND $batch AS row
        MERGE (f:Fund {name: row.name})
        SET f.type = row.type, f.aum_avg = row.aum_avg
        """
        n = self._batch_run(cypher, records)
        self.stats['Fund'] = n
        print(f"  Created {n} Fund nodes")

    def create_stocks(self, df):
        """Create Stock nodes."""
        stock_info = df.groupby('ISIN').agg({
            'stock_name': 'first',
        }).reset_index()

        # Add symbol and sector
        if 'symbol' in df.columns:
            symbol_map = df.groupby('ISIN')['symbol'].first()
            stock_info = stock_info.merge(symbol_map, on='ISIN', how='left')
        if 'sector' in df.columns:
            sector_map = df.groupby('ISIN')['sector'].first()
            stock_info = stock_info.merge(sector_map, on='ISIN', how='left')

        records = []
        for _, row in stock_info.iterrows():
            records.append(clean_record({
                'isin': row['ISIN'],
                'name': row.get('stock_name', ''),
                'symbol': row.get('symbol', ''),
                'sector': row.get('sector', 'OTHERS'),
            }))

        cypher = """
        UNWIND $batch AS row
        MERGE (s:Stock {isin: row.isin})
        SET s.name = row.name, s.symbol = row.symbol, s.sector = row.sector
        """
        n = self._batch_run(cypher, records)
        self.stats['Stock'] = n
        print(f"  Created {n} Stock nodes")

    def create_timeperiods(self, df):
        """Create TimePeriod nodes and NEXT chain."""
        if 'year_month_str' not in df.columns:
            print("  No year_month_str; skipping TimePeriod nodes.")
            return

        months = sorted(df['year_month_str'].dropna().unique().tolist())
        records = []
        for m in months:
            year = int(m[:4])
            month = int(m[5:7])
            quarter = quarter_from_month(m)
            fy = fiscal_year_from_month(m)
            records.append({
                'id': m, 'year': year, 'month': month,
                'quarter': quarter, 'fiscal_year': fy,
            })

        cypher = """
        UNWIND $batch AS row
        MERGE (t:TimePeriod {id: row.id})
        SET t.year = row.year, t.month = row.month,
            t.quarter = row.quarter, t.fiscal_year = row.fiscal_year
        """
        n = self._batch_run(cypher, records)
        self.stats['TimePeriod'] = n
        print(f"  Created {n} TimePeriod nodes")

        # NEXT chain
        print("  Creating NEXT chain...")
        for i in range(len(months) - 1):
            self._run("""
                MATCH (t1:TimePeriod {id: $m1}), (t2:TimePeriod {id: $m2})
                MERGE (t1)-[:NEXT]->(t2)
            """, {'m1': months[i], 'm2': months[i + 1]})
        self.stats['NEXT'] = len(months) - 1
        print(f"  Created {len(months) - 1} NEXT relationships")

    def create_market_regimes(self, df, macro_df=None):
        """
        Create MarketRegime nodes based on VIX level and Nifty 50 return.
        Each unique (regime_type) gets a node; TimePeriods link via IN_REGIME.

        Looks for india_vix / nifty50_return in the macro DataFrame first,
        then falls back to the main portfolio DataFrame.
        """
        print("\n  Creating MarketRegime nodes...")

        # Collect VIX and Nifty return data
        vix_data = {}
        nifty_return_data = {}

        # 1. Try macro DataFrame
        if macro_df is not None and not macro_df.empty:
            m_col = None
            for candidate in ['year_month_str', 'month', 'date']:
                if candidate in macro_df.columns:
                    m_col = candidate
                    break

            if m_col:
                # Try multiple VIX column name variants
                _vix_col = next(
                    (c for c in ['india_vix', 'india_vix_close',
                                 'vix', 'india_vix_level', 'vix_close']
                     if c in macro_df.columns), None)
                if _vix_col:
                    for _, row in macro_df.iterrows():
                        m = str(row[m_col])[:7]  # ensure YYYY-MM format
                        val = row[_vix_col]
                        if pd.notna(val):
                            vix_data[m] = float(val)

                if 'nifty50_return' in macro_df.columns:
                    for _, row in macro_df.iterrows():
                        m = str(row[m_col])[:7]
                        val = row['nifty50_return']
                        if pd.notna(val):
                            nifty_return_data[m] = float(val)
                elif 'nifty50' in macro_df.columns:
                    # Compute returns from price levels
                    sorted_macro = macro_df.sort_values(m_col)
                    prev_val = None
                    for _, row in sorted_macro.iterrows():
                        m = str(row[m_col])[:7]
                        val = row['nifty50']
                        if pd.notna(val) and prev_val is not None and prev_val != 0:
                            nifty_return_data[m] = ((float(val) - prev_val) / prev_val) * 100
                        if pd.notna(val):
                            prev_val = float(val)

        # 2. Fallback: check main DataFrame (try multiple VIX column variants)
        if not vix_data:
            _vix_col = next(
                (c for c in ['india_vix', 'india_vix_close',
                             'vix', 'india_vix_level', 'vix_close']
                 if c in df.columns), None)
            if _vix_col:
                monthly_vix = df.groupby('year_month_str')[_vix_col].mean()
                vix_data = {k: float(v) for k, v in monthly_vix.items() if pd.notna(v)}
                print(f"  VIX data from main df ({_vix_col}): {len(vix_data)} months")

        if not nifty_return_data and 'nifty50_return' in df.columns:
            monthly_ret = df.groupby('year_month_str')['nifty50_return'].mean()
            nifty_return_data = {k: float(v) for k, v in monthly_ret.items() if pd.notna(v)}

        if not vix_data and not nifty_return_data:
            print("  WARNING: No VIX or Nifty return data available. "
                  "Creating default UNKNOWN regime.")

        # Classify each month
        months = sorted(df['year_month_str'].dropna().unique())
        regime_records = []
        period_regime_links = []
        seen_regimes = set()

        for m in months:
            vix = vix_data.get(m, np.nan)
            nifty_ret = nifty_return_data.get(m, np.nan)

            regime_type, vix_level, nifty_trend = classify_market_regime(vix, nifty_ret)
            regime_id = regime_type

            if regime_id not in seen_regimes:
                seen_regimes.add(regime_id)
                regime_records.append({
                    'id': regime_id,
                    'regime_type': regime_type,
                    'vix_level': vix_level,
                    'nifty_trend': nifty_trend,
                })

            period_regime_links.append({
                'period_id': m,
                'regime_id': regime_id,
                'vix_value': None if (isinstance(vix, float) and np.isnan(vix)) else float(vix),
                'nifty_return': None if (isinstance(nifty_ret, float) and np.isnan(nifty_ret)) else float(nifty_ret),
            })

        # Create MarketRegime nodes
        if regime_records:
            cypher = """
            UNWIND $batch AS row
            MERGE (mr:MarketRegime {id: row.id})
            SET mr.regime_type = row.regime_type,
                mr.vix_level = row.vix_level,
                mr.nifty_trend = row.nifty_trend
            """
            n = self._batch_run(cypher, regime_records)
            self.stats['MarketRegime'] = n
            print(f"  Created {n} MarketRegime nodes")

        # Link TimePeriod -> MarketRegime via IN_REGIME
        if period_regime_links:
            cypher = """
            UNWIND $batch AS row
            MATCH (t:TimePeriod {id: row.period_id})
            MATCH (mr:MarketRegime {id: row.regime_id})
            MERGE (t)-[r:IN_REGIME]->(mr)
            SET r.vix_value = row.vix_value, r.nifty_return = row.nifty_return
            """
            n = self._batch_run(cypher, period_regime_links)
            self.stats['IN_REGIME'] = n
            print(f"  Created {n} IN_REGIME relationships")

    # ------------------------------------------------------------------
    # Relationship creation
    # ------------------------------------------------------------------
    def create_belongs_to(self, df):
        """Create Stock-[:BELONGS_TO]->Sector relationships."""
        if 'sector' not in df.columns:
            return

        stock_sectors = df.groupby('ISIN')['sector'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        ).reset_index()
        records = [{'isin': row['ISIN'], 'sector': row['sector']}
                   for _, row in stock_sectors.iterrows()
                   if pd.notna(row['sector'])]

        cypher = """
        UNWIND $batch AS row
        MATCH (s:Stock {isin: row.isin}), (sec:Sector {name: row.sector})
        MERGE (s)-[:BELONGS_TO]->(sec)
        """
        n = self._batch_run(cypher, records)
        self.stats['BELONGS_TO'] = n
        print(f"  Created {n} BELONGS_TO relationships")

    def create_holds(self, df):
        """Create Fund-[:HOLDS]->Stock relationships with properties."""
        print("  Creating HOLDS relationships...")

        # Build records from portfolio
        optional_cols = [
            ('pct_nav', 'pct_nav'),
            ('Quantity', 'quantity'),
            ('Market Value', 'market_value'),
            ('position_action', 'position_action'),
            ('allocation_change', 'allocation_change'),
            ('size', 'size'),
            ('holding_tenure', 'holding_tenure'),
            ('rank_in_fund', 'rank'),
            ('rank', 'rank'),
            ('consensus_count', 'consensus'),
            ('consensus', 'consensus'),
            ('monthly_return', 'monthly_return'),
            ('rsi', 'rsi'),
            ('beta', 'beta'),
            ('sentiment_score', 'sentiment_score'),
        ]

        records = []
        for _, row in df.iterrows():
            rec = {
                'fund_name': row['Fund_Name'],
                'isin': row['ISIN'],
                'month': row['year_month_str'],
                'pct_nav': float(row['pct_nav']) if pd.notna(row.get('pct_nav')) else None,
            }

            for src_col, tgt_key in optional_cols:
                if src_col in df.columns and tgt_key not in rec:
                    val = row.get(src_col)
                    if pd.notna(val) if not isinstance(val, str) else True:
                        if isinstance(val, (np.int64, np.int32)):
                            rec[tgt_key] = int(val)
                        elif isinstance(val, (np.float64, np.float32)):
                            rec[tgt_key] = None if (np.isnan(val) or np.isinf(val)) else float(val)
                        else:
                            rec[tgt_key] = val
            records.append(rec)

        cypher = """
        UNWIND $batch AS row
        MATCH (f:Fund {name: row.fund_name}), (s:Stock {isin: row.isin})
        MERGE (f)-[h:HOLDS {month: row.month, isin: row.isin}]->(s)
        SET h.pct_nav = row.pct_nav,
            h.quantity = row.quantity,
            h.market_value = row.market_value,
            h.position_action = row.position_action,
            h.allocation_change = row.allocation_change,
            h.size = row.size,
            h.holding_tenure = row.holding_tenure,
            h.rank = row.rank,
            h.consensus = row.consensus,
            h.monthly_return = row.monthly_return,
            h.rsi = row.rsi,
            h.beta = row.beta,
            h.sentiment_score = row.sentiment_score
        """
        n = self._batch_run(cypher, records, batch_size=BATCH_SIZE)
        self.stats['HOLDS'] = n
        print(f"  Created {n} HOLDS relationships")

    def create_exits(self, exit_df):
        """Create Fund-[:EXITED]->Stock relationships."""
        if exit_df is None or exit_df.empty:
            print("  No exit events to process.")
            return

        records = []
        for _, row in exit_df.iterrows():
            rec = clean_record({
                'fund_name': row.get('Fund_Name', ''),
                'isin': row.get('ISIN', ''),
                'month': row.get('exit_month', row.get('year_month_str', '')),
                'last_pct_nav': row.get('last_pct_nav', row.get('pct_nav', 0)),
                'holding_tenure': row.get('holding_tenure', row.get('holding_duration', 0)),
            })
            if rec['fund_name'] and rec['isin']:
                records.append(rec)

        cypher = """
        UNWIND $batch AS row
        MATCH (f:Fund {name: row.fund_name}), (s:Stock {isin: row.isin})
        MERGE (f)-[e:EXITED {month: row.month, isin: row.isin}]->(s)
        SET e.last_pct_nav = row.last_pct_nav,
            e.holding_tenure = row.holding_tenure
        """
        n = self._batch_run(cypher, records)
        self.stats['EXITED'] = n
        print(f"  Created {n} EXITED relationships")

    def create_active_in(self, df):
        """Create Fund-[:ACTIVE_IN]->TimePeriod relationships."""
        if 'year_month_str' not in df.columns:
            return

        fund_months = df.groupby(['Fund_Name', 'year_month_str']).size().reset_index()
        fund_months.columns = ['fund_name', 'month', 'holdings_count']

        records = [
            {'fund_name': str(row['fund_name']), 'month': str(row['month']),
             'holdings_count': int(row['holdings_count'])}
            for _, row in fund_months.iterrows()
        ]

        cypher = """
        UNWIND $batch AS row
        MATCH (f:Fund {name: row.fund_name}), (t:TimePeriod {id: row.month})
        MERGE (f)-[r:ACTIVE_IN]->(t)
        SET r.holdings_count = row.holdings_count
        """
        n = self._batch_run(cypher, records)
        self.stats['ACTIVE_IN'] = n
        print(f"  Created {n} ACTIVE_IN relationships")

    def create_fund_snapshots(self, df):
        """Create FundSnapshot nodes with OF_FUND and AT_TIME links."""
        if 'year_month_str' not in df.columns:
            return

        grouped = df.groupby(['Fund_Name', 'year_month_str']).agg({
            'ISIN': 'nunique',
            'pct_nav': 'sum',
        }).reset_index()

        records = []
        for _, row in grouped.iterrows():
            records.append(clean_record({
                'fund_name': row['Fund_Name'],
                'month': row['year_month_str'],
                'total_stocks': row['ISIN'],
                'total_nav': row['pct_nav'],
            }))

        cypher = """
        UNWIND $batch AS row
        CREATE (fs:FundSnapshot {
            fund_name: row.fund_name,
            month: row.month,
            total_stocks: row.total_stocks,
            total_nav: row.total_nav
        })
        WITH fs, row
        MATCH (f:Fund {name: row.fund_name})
        MERGE (fs)-[:OF_FUND]->(f)
        WITH fs, row
        MATCH (t:TimePeriod {id: row.month})
        MERGE (fs)-[:AT_TIME]->(t)
        """
        n = self._batch_run(cypher, records)
        self.stats['FundSnapshot'] = n
        print(f"  Created {n} FundSnapshot nodes")

    def create_stock_snapshots(self, df):
        """Create StockSnapshot nodes with OF_STOCK and AT_TIME links."""
        if 'year_month_str' not in df.columns:
            return

        # Aggregate numeric features per stock per month
        num_cols = ['pct_nav', 'monthly_return', 'rsi', 'beta', 'volatility',
                    'sentiment_score', 'macd']
        available = [c for c in num_cols if c in df.columns]

        # Also try fundamental columns
        fund_cols = ['pe', 'pb', 'eps', 'market_cap']
        available.extend([c for c in fund_cols if c in df.columns])

        if not available:
            print("  No numeric columns for StockSnapshot; skipping.")
            return

        agg_dict = {c: 'mean' for c in available}
        grouped = df.groupby(['ISIN', 'year_month_str']).agg(agg_dict).reset_index()

        records = []
        for _, row in grouped.iterrows():
            rec = {
                'isin': row['ISIN'],
                'month': row['year_month_str'],
            }
            for col in available:
                val = row.get(col)
                if pd.notna(val) and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    rec[col] = float(val)
                else:
                    rec[col] = None
            records.append(rec)

        cypher = """
        UNWIND $batch AS row
        CREATE (ss:StockSnapshot {
            isin: row.isin,
            month: row.month,
            pe: row.pe,
            pb: row.pb,
            eps: row.eps,
            beta: row.beta,
            market_cap: row.market_cap,
            rsi: row.rsi,
            macd: row.macd,
            sentiment_mean: row.sentiment_score,
            monthly_return: row.monthly_return,
            volatility: row.volatility
        })
        WITH ss, row
        MATCH (s:Stock {isin: row.isin})
        MERGE (ss)-[:OF_STOCK]->(s)
        WITH ss, row
        MATCH (t:TimePeriod {id: row.month})
        MERGE (ss)-[:AT_TIME]->(t)
        """
        n = self._batch_run(cypher, records, batch_size=200)
        self.stats['StockSnapshot'] = n
        print(f"  Created {n} StockSnapshot nodes")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify(self):
        """Run verification queries."""
        print("\n  Verification Queries:")
        queries = [
            ("Funds", "MATCH (f:Fund) RETURN COUNT(f) AS cnt"),
            ("Stocks", "MATCH (s:Stock) RETURN COUNT(s) AS cnt"),
            ("Sectors", "MATCH (sec:Sector) RETURN COUNT(sec) AS cnt"),
            ("TimePeriods", "MATCH (t:TimePeriod) RETURN COUNT(t) AS cnt"),
            ("MarketRegimes", "MATCH (mr:MarketRegime) RETURN COUNT(mr) AS cnt"),
            ("HOLDS", "MATCH ()-[h:HOLDS]->() RETURN COUNT(h) AS cnt"),
            ("EXITED", "MATCH ()-[e:EXITED]->() RETURN COUNT(e) AS cnt"),
            ("BELONGS_TO", "MATCH ()-[b:BELONGS_TO]->() RETURN COUNT(b) AS cnt"),
            ("ACTIVE_IN", "MATCH ()-[a:ACTIVE_IN]->() RETURN COUNT(a) AS cnt"),
            ("IN_REGIME", "MATCH ()-[r:IN_REGIME]->() RETURN COUNT(r) AS cnt"),
            ("FundSnapshots", "MATCH (fs:FundSnapshot) RETURN COUNT(fs) AS cnt"),
            ("StockSnapshots", "MATCH (ss:StockSnapshot) RETURN COUNT(ss) AS cnt"),
        ]
        for label, cypher in queries:
            result = self._run_data(cypher)
            count = result[0]['cnt'] if result else 0
            print(f"    {label:20s}: {count:>8,}")

    def print_stats(self):
        """Print build statistics."""
        print("\n  Build Statistics:")
        for key, val in sorted(self.stats.items()):
            print(f"    {key:20s}: {val:>8,}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("STEP 10 -- BUILD TEMPORAL KNOWLEDGE GRAPH")
    print("=" * 70)

    # Load portfolio data
    if os.path.exists(PORTFOLIO_CLEAN_CSV):
        df = pd.read_csv(PORTFOLIO_CLEAN_CSV, low_memory=False)
        print(f"  Loaded portfolio: {df.shape}")
    elif os.path.exists(EXISTING_TEMPORAL_CSV):
        df = pd.read_csv(EXISTING_TEMPORAL_CSV, low_memory=False)
        print(f"  Fallback to existing temporal CSV: {df.shape}")
    else:
        print(f"  ERROR: No portfolio data found.")
        print(f"    Checked: {PORTFOLIO_CLEAN_CSV}")
        print(f"    Checked: {EXISTING_TEMPORAL_CSV}")
        return

    # Ensure required columns
    if 'year_month_str' not in df.columns:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['year_month_str'] = df['Date'].dt.to_period('M').astype(str)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year_month_str'] = df['date'].dt.to_period('M').astype(str)

    if 'stock_name' not in df.columns:
        if 'Name of the Instrument' in df.columns:
            df['stock_name'] = df['Name of the Instrument']
        elif 'stock_name_raw' in df.columns:
            df['stock_name'] = df['stock_name_raw']
        else:
            df['stock_name'] = df.get('ISIN', 'Unknown')

    if 'sector' not in df.columns and 'Industry' in df.columns:
        df['sector'] = df['Industry'].apply(standardize_industry)

    if 'pct_nav' not in df.columns and '% to Net Assets' in df.columns:
        df['pct_nav'] = pd.to_numeric(df['% to Net Assets'], errors='coerce')

    if 'Fund_Type' not in df.columns:
        df['Fund_Type'] = 'Unknown'

    print(f"  Columns: {list(df.columns[:15])}...")
    print(f"  Months: {df['year_month_str'].nunique()}")
    print(f"  Funds: {df['Fund_Name'].nunique()}")
    print(f"  Stocks: {df['ISIN'].nunique()}")

    # Load exit events
    exit_df = None
    if os.path.exists(EXIT_EVENTS_CSV):
        exit_df = pd.read_csv(EXIT_EVENTS_CSV)
        print(f"  Exit events: {len(exit_df)}")
    elif os.path.exists(EXISTING_EXITS_CSV):
        exit_df = pd.read_csv(EXISTING_EXITS_CSV)
        print(f"  Fallback exit events: {len(exit_df)}")

    # Load macro data for market regime classification
    macro_df = None
    if os.path.exists(MACRO_CSV):
        macro_df = pd.read_csv(MACRO_CSV, low_memory=False)
        print(f"  Macro data for regimes: {macro_df.shape}")
    else:
        print("  No macro CSV found; regime classification will use "
              "portfolio data or default to UNKNOWN.")

    # Optionally enrich with features data
    features_path = FEATURES_CSV if os.path.exists(FEATURES_CSV) else (
        FEATURES_FALLBACK if os.path.exists(FEATURES_FALLBACK) else None
    )
    if features_path:
        feat_df = pd.read_csv(features_path, low_memory=False)
        print(f"  Features for enrichment: {feat_df.shape}")
        # Merge enrichment columns
        enrich_cols = ['ISIN', 'year_month_str', 'rsi', 'macd', 'beta',
                       'volatility', 'sentiment_score', 'monthly_return',
                       'pe', 'pb', 'eps', 'market_cap']
        enrich_cols = [c for c in enrich_cols if c in feat_df.columns]
        if len(enrich_cols) > 2:
            feat_agg = feat_df[enrich_cols].groupby(
                ['ISIN', 'year_month_str']).mean().reset_index()
            merge_on = ['ISIN', 'year_month_str']
            new_cols = [c for c in enrich_cols if c not in merge_on and c not in df.columns]
            if new_cols:
                df = df.merge(feat_agg[merge_on + new_cols],
                              on=merge_on, how='left')
                print(f"  Enriched with {len(new_cols)} feature columns")

    # Build KG
    builder = TemporalKGBuilder()
    try:
        builder.create_constraints()
        builder.clear_graph()

        print("\n--- Creating Nodes ---")
        builder.create_sectors(df)
        builder.create_funds(df)
        builder.create_stocks(df)
        builder.create_timeperiods(df)
        builder.create_market_regimes(df, macro_df)

        print("\n--- Creating Relationships ---")
        builder.create_belongs_to(df)
        builder.create_holds(df)
        builder.create_exits(exit_df)
        builder.create_active_in(df)

        print("\n--- Creating Snapshots ---")
        builder.create_fund_snapshots(df)
        builder.create_stock_snapshots(df)

        print("\n--- Verification ---")
        builder.verify()
        builder.print_stats()

        print(f"\n{'='*70}")
        print("  STEP 10 COMPLETE -- Temporal KG built in Neo4j")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        builder.close()


if __name__ == '__main__':
    main()
