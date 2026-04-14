"""
Step 05 -- FinBERT Sentiment Analysis on News Headlines (v3)
============================================================

Loads all CSV files from NEWS_DIR (~1,093 files), runs ProsusAI/finbert
inference in batches, and aggregates headline-level sentiment to monthly
per symbol/ISIN.

Input:  NEWS_DIR (~1,093 CSV files, each with Date, Heading, Source, Symbol)
Output: SENTIMENT_DIR / finbert_monthly_sentiment.csv

FinBERT label order: positive=0, negative=1, neutral=2

v3 improvements:
- compound = (pos - neg) * (1 - neutral)  [was: pos - neg]
  Neutral-weighted: high neutral dampens signal, not ignored
- Confidence-weighted monthly aggregation (weight = max probability)
- Recency-weighted aggregation within month (recent headlines matter more)
- Minimum news_count filter (>=3, else unreliable)
- Richer feature set: sentiment_confidence, sentiment_extremity,
  sentiment_dispersion, sentiment_reversal, pct_strongly_positive,
  pct_strongly_negative, max_sentiment, min_sentiment, sentiment_skew
- Date parsing: dayfirst=True for Indian DD-MM-YYYY format

FIXES (v2):
- FIX 1: ISIN mapping — detect if Symbol already looks like ISIN.
- FIX 2: Non-English headlines — filter non-ASCII before inference.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NEWS_DIR, SENTIMENT_DIR, MAPPINGS_DIR,
    FINBERT_MODEL, FINBERT_BATCH_SIZE, DATE_RANGE_STR,
)
from utils import coverage_report

import pandas as pd
import numpy as np
import glob
import warnings

warnings.filterwarnings('ignore')

OUTPUT_CSV = os.path.join(SENTIMENT_DIR, 'finbert_monthly_sentiment.csv')
CHECKPOINT_CSV = os.path.join(SENTIMENT_DIR, '_finbert_raw_checkpoint.csv')
UNIFIED_MAP_CSV = os.path.join(MAPPINGS_DIR, 'unified_isin_symbol_map.csv')

ISIN_PATTERN = r'^INE[0-9A-Z]{9}$'


# ============================================================
# FinBERT model loading
# ============================================================

def load_finbert():
    """Load ProsusAI/finbert tokenizer and model for direct logit access."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        print("[step05] ERROR: transformers/torch not installed.")
        print("         Install with: pip install transformers torch")
        return None, None, None

    print(f"[step05] Loading FinBERT model: {FINBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    device_name = "GPU (CUDA)" if device.type == 'cuda' else "CPU"
    print(f"[step05] Device: {device_name}")

    return tokenizer, model, device


def score_headlines_batch(headlines, tokenizer, model, device, batch_size=32):
    """Score headlines using FinBERT with direct softmax on logits.

    FinBERT label order: positive=0, negative=1, neutral=2

    Returns list of dicts with keys:
        positive_prob, negative_prob, neutral_prob, compound, label

    FIX: preserves original ordering when batch has mixed empty/non-empty
    headlines. Previously clean_indices was tracked but never used to
    reorder results, causing date mismatches.
    """
    import torch

    n = len(headlines)
    # Pre-fill all results as neutral
    results = [{
        'positive_prob': 0.0, 'negative_prob': 0.0,
        'neutral_prob': 1.0, 'compound': 0.0,
        'confidence': 1.0, 'label': 'neutral',
    } for _ in range(n)]

    for start in range(0, n, batch_size):
        batch = headlines[start:start + batch_size]

        clean_batch = []
        clean_positions = []  # original positions within full results list

        for i, h in enumerate(batch):
            global_i = start + i
            if pd.isna(h) or str(h).strip() == '':
                pass  # already neutral in results
            else:
                clean_text = str(h).strip()[:512]
                clean_batch.append(clean_text)
                clean_positions.append(global_i)

        if not clean_batch:
            continue

        try:
            inputs = tokenizer(
                clean_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

            # FIX: write each result back to its original position
            for j, prob_row in enumerate(probs):
                pos_prob = float(prob_row[0])
                neg_prob = float(prob_row[1])
                neu_prob = float(prob_row[2])
                # v3: neutral-dampened compound — high neutral suppresses signal
                compound = (pos_prob - neg_prob) * (1.0 - neu_prob)
                max_idx = int(np.argmax(prob_row))
                label = ['positive', 'negative', 'neutral'][max_idx]
                confidence = float(prob_row[max_idx])  # v3: model confidence

                results[clean_positions[j]] = {
                    'positive_prob': pos_prob,
                    'negative_prob': neg_prob,
                    'neutral_prob': neu_prob,
                    'compound': compound,
                    'confidence': confidence,
                    'label': label,
                }

        except Exception as e:
            print(f"[step05] WARNING: Batch inference error: {e}")
            # leave as neutral (already set)

    return results


# ============================================================
# News file loading
# ============================================================

def is_english(text):
    """Return True if text is ASCII-only (English)."""
    try:
        return str(text).encode('ascii', errors='strict') and True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def load_all_news(news_dir, date_range=None):
    """Load all CSV files from NEWS_DIR, standardize columns, filter dates.

    FIX: filters out non-ASCII (Tamil/Hindi) headlines before returning.
    """
    news_files = sorted(glob.glob(os.path.join(news_dir, '*.csv')))
    print(f"[step05] Found {len(news_files)} news CSV files in: {news_dir}")

    if not news_files:
        return pd.DataFrame()

    all_frames = []

    for fpath in news_files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        # Standardize column names
        df.columns = df.columns.str.strip()
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ('date', 'published', 'datetime'):
                col_map[c] = 'Date'
            elif cl in ('heading', 'headline', 'title', 'text'):
                col_map[c] = 'Heading'
            elif cl in ('source', 'publisher'):
                col_map[c] = 'Source'
            elif cl in ('symbol', 'ticker', 'stock'):
                col_map[c] = 'Symbol'
        df = df.rename(columns=col_map)

        if 'Heading' not in df.columns:
            continue

        base_name = os.path.splitext(os.path.basename(fpath))[0]
        if 'Symbol' not in df.columns:
            df['Symbol'] = base_name
        df['file_name'] = base_name

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Clean dates — dayfirst=True for Indian DD-MM-YYYY format
    if 'Date' in combined.columns:
        combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce',
                                          dayfirst=True)
        combined = combined.dropna(subset=['Date'])

        if date_range is not None:
            start_str, end_str = date_range
            start_date = pd.to_datetime(start_str + '-01')
            end_date = pd.to_datetime(end_str + '-28') + pd.offsets.MonthEnd(0)
            combined = combined[(combined['Date'] >= start_date) &
                                (combined['Date'] <= end_date)]

    print(f"[step05] Total headlines loaded: {len(combined):,}")

    # FIX 2: Filter non-English headlines
    before = len(combined)
    combined = combined[combined['Heading'].apply(is_english)].copy()
    removed = before - len(combined)
    print(f"[step05] Removed {removed:,} non-English headlines ({removed/before*100:.1f}%)")
    print(f"[step05] English headlines remaining: {len(combined):,}")
    print(f"         Unique symbols: {combined['Symbol'].nunique() if 'Symbol' in combined.columns else 'N/A'}")

    return combined


# ============================================================
# Symbol to ISIN mapping
# ============================================================

def build_symbol_to_isin_map():
    """Build symbol -> ISIN mapping from unified mapping file."""
    symbol_to_isin = {}
    if os.path.isfile(UNIFIED_MAP_CSV):
        try:
            umap = pd.read_csv(UNIFIED_MAP_CSV, low_memory=False)
            umap.columns = umap.columns.str.strip()

            sym_col = None
            isin_col = None
            for c in umap.columns:
                cu = c.upper()
                if cu in ('SYMBOL', 'TICKER', 'NSE_SYMBOL'):
                    sym_col = c
                if cu == 'ISIN' or 'ISIN' in cu:
                    isin_col = c

            if sym_col and isin_col:
                for _, row in umap.dropna(subset=[sym_col, isin_col]).iterrows():
                    sym = str(row[sym_col]).strip().upper()
                    isin = str(row[isin_col]).strip().upper()
                    if sym and isin and sym != 'NAN' and isin != 'NAN':
                        symbol_to_isin[sym] = isin
                print(f"[step05] Symbol->ISIN mapping: {len(symbol_to_isin)} entries")
        except Exception as e:
            print(f"[step05] WARNING: Could not load symbol->ISIN map: {e}")
    else:
        print(f"[step05] WARNING: Unified mapping not found: {UNIFIED_MAP_CSV}")

    return symbol_to_isin


# ============================================================
# Checkpoint helpers
# ============================================================

def load_checkpoint():
    """Load previously processed raw sentiment results."""
    if os.path.isfile(CHECKPOINT_CSV):
        try:
            df = pd.read_csv(CHECKPOINT_CSV, low_memory=False)
            processed = set(df['Symbol'].dropna().unique()) if 'Symbol' in df.columns else set()
            print(f"[step05] Checkpoint loaded: {len(processed)} symbols already processed, "
                  f"{len(df):,} rows")
            return df, processed
        except Exception:
            pass
    return pd.DataFrame(), set()


def save_checkpoint(df_raw):
    """Save raw sentiment results for checkpoint/resume."""
    df_raw.to_csv(CHECKPOINT_CSV, index=False)


# ============================================================
# Monthly aggregation
# ============================================================

def aggregate_sentiment_monthly(df_raw, symbol_to_isin):
    """Aggregate raw headline-level sentiment to monthly per symbol.

    v3 enhancements:
    - Confidence-weighted mean (headlines the model is sure about count more)
    - Recency-weighted mean (recent headlines within a month count more)
    - Min news_count filter (>=3 required, else NaN — unreliable)
    - Richer features: confidence, extremity, dispersion, reversal,
      strongly positive/negative ratios, min/max, skew

    FIX 1: Symbol column may contain ISINs — detect and assign directly.
    """
    MIN_NEWS_COUNT = 3  # months with <3 headlines marked unreliable

    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date'])
    df['year_month_str'] = df['Date'].dt.to_period('M').astype(str)

    # Ensure confidence column exists (backward compat with old checkpoints)
    if 'confidence' not in df.columns:
        df['confidence'] = df[['positive_prob', 'negative_prob', 'neutral_prob']].max(axis=1)

    group_col = 'Symbol'

    # -- Recency weight: within each (symbol, month), assign exponential decay --
    df = df.sort_values([group_col, 'Date'])
    df['_day_in_month'] = df['Date'].dt.day
    # Decay: lambda=0.03 per day from month end → last day weight ~1, day 1 ~0.41
    df['_recency_wt'] = df.groupby([group_col, 'year_month_str'])['_day_in_month'].transform(
        lambda x: np.exp(-0.03 * (x.max() - x))
    )
    # Combined weight: confidence × recency
    df['_weight'] = df['confidence'] * df['_recency_wt']

    grouped = df.groupby([group_col, 'year_month_str'])

    def _weighted_mean(grp, col='compound'):
        w = grp['_weight'].values
        v = grp[col].values
        ws = w.sum()
        return np.dot(w, v) / ws if ws > 0 else np.nanmean(v)

    agg = grouped.agg(
        # Core sentiment (unweighted for backward compat + weighted for v3)
        sentiment_mean_raw=('compound', 'mean'),
        sentiment_std=('compound', 'std'),
        sentiment_median=('compound', 'median'),
        news_count=('compound', 'count'),
        # Probability averages
        avg_positive_prob=('positive_prob', 'mean'),
        avg_negative_prob=('negative_prob', 'mean'),
        avg_neutral_prob=('neutral_prob', 'mean'),
        # v3: richer features
        sentiment_confidence=('confidence', 'mean'),
        max_sentiment=('compound', 'max'),
        min_sentiment=('compound', 'min'),
        sentiment_skew=('compound', lambda x: x.skew() if len(x) >= 3 else 0.0),
        # Strong signal ratios (>0.6 confidence threshold)
        pct_strongly_positive=('positive_prob', lambda x: (x > 0.6).mean()),
        pct_strongly_negative=('negative_prob', lambda x: (x > 0.6).mean()),
        # Standard ratios (>0.5 threshold, backward compat)
        positive_ratio=('positive_prob', lambda x: (x > 0.5).mean()),
        negative_ratio=('negative_prob', lambda x: (x > 0.5).mean()),
    ).reset_index()

    # Confidence-weighted mean (main sentiment_mean)
    try:
        wm = grouped.apply(_weighted_mean, include_groups=False)
    except TypeError:
        wm = grouped.apply(_weighted_mean)
    wm = wm.reset_index()
    wm.columns = list(wm.columns[:-1]) + ['sentiment_mean']
    agg = agg.merge(wm, on=[group_col, 'year_month_str'], how='left')

    agg['sentiment_std'] = agg['sentiment_std'].fillna(0)

    # v3: derived features
    # Extremity: how far from neutral (absolute compound)
    agg['sentiment_extremity'] = (agg['max_sentiment'].abs() +
                                   agg['min_sentiment'].abs()) / 2.0
    # Dispersion: range of sentiment within month
    agg['sentiment_dispersion'] = agg['max_sentiment'] - agg['min_sentiment']

    # Sentiment momentum: 3-month rolling change
    agg = agg.sort_values([group_col, 'year_month_str'])
    agg['sentiment_momentum'] = agg.groupby(group_col)['sentiment_mean'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().diff()
    )
    agg['sentiment_momentum'] = agg['sentiment_momentum'].fillna(0)

    # Sentiment reversal: sign change from previous month
    agg['sentiment_reversal'] = agg.groupby(group_col)['sentiment_mean'].transform(
        lambda x: (np.sign(x) != np.sign(x.shift(1))).astype(float)
    )
    agg['sentiment_reversal'] = agg['sentiment_reversal'].fillna(0)

    # v3: Mark low-news months as unreliable (NaN out their sentiment)
    low_news = agg['news_count'] < MIN_NEWS_COUNT
    n_low = low_news.sum()
    if n_low > 0:
        print(f"[step05] WARNING: {n_low} symbol-months have <{MIN_NEWS_COUNT} headlines "
              f"({n_low/len(agg)*100:.1f}%) — sentiment set to NaN (unreliable)")
        sentiment_cols = ['sentiment_mean', 'sentiment_mean_raw', 'sentiment_std',
                          'sentiment_median', 'sentiment_confidence',
                          'sentiment_extremity', 'sentiment_dispersion',
                          'sentiment_momentum', 'sentiment_reversal',
                          'sentiment_skew']
        for col in sentiment_cols:
            if col in agg.columns:
                agg.loc[low_news, col] = np.nan

    # FIX 1: symbol column contains ISINs — assign directly
    agg['symbol'] = agg[group_col].astype(str).str.strip().str.upper()

    is_isin = agg['symbol'].str.match(ISIN_PATTERN, na=False)
    n_isin = is_isin.sum()
    n_ticker = (~is_isin).sum()
    print(f"[step05] Symbol format: {n_isin} ISINs, {n_ticker} tickers")

    # Assign ISIN: if symbol IS an ISIN → use directly
    #              if symbol is a ticker → look up in mapping
    agg['ISIN'] = np.where(
        is_isin,
        agg['symbol'],                          # already an ISIN
        agg['symbol'].map(symbol_to_isin)       # ticker → ISIN lookup
    )

    isin_coverage = agg['ISIN'].notna().mean() * 100
    print(f"[step05] ISIN coverage after fix: {isin_coverage:.1f}%")

    if group_col != 'symbol':
        agg = agg.drop(columns=[group_col], errors='ignore')

    return agg


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("STEP 05: FinBERT Sentiment Analysis (v3 — enhanced features)")
    print("=" * 70)

    # --- Cache check ---
    # Delete old output to force reaggregation with the fix
    if os.path.isfile(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"[step05] Removed old output to force re-aggregation with ISIN fix.")

    # --- Check news directory ---
    if not os.path.isdir(NEWS_DIR):
        print(f"[step05] ERROR: News directory not found: {NEWS_DIR}")
        return

    # --- Load symbol->ISIN mapping ---
    symbol_to_isin = build_symbol_to_isin_map()

    # --- Load checkpoint ---
    df_checkpoint, processed_symbols = load_checkpoint()

    # --- If checkpoint has all data, skip inference and just re-aggregate ---
    if len(processed_symbols) > 0:
        print(f"\n[step05] Checkpoint has {len(df_checkpoint):,} rows from "
              f"{len(processed_symbols)} symbols.")
        print(f"[step05] Skipping FinBERT inference — re-aggregating with ISIN fix ...")
        df_all_raw = df_checkpoint.copy()
    else:
        # --- Load news data ---
        print(f"\n[step05] Loading news headlines ...")
        news_df = load_all_news(NEWS_DIR, date_range=DATE_RANGE_STR)

        if news_df.empty:
            print("[step05] ERROR: No news data loaded.")
            return

        # Filter to unprocessed symbols
        if 'Symbol' in news_df.columns:
            remaining_df = news_df[~news_df['Symbol'].isin(processed_symbols)]
        else:
            remaining_df = news_df

        print(f"[step05] Already processed: {len(processed_symbols)} symbols")
        print(f"[step05] Remaining headlines: {len(remaining_df):,}")

        if len(remaining_df) > 0:
            print("\n[step05] Loading FinBERT model ...")
            tokenizer, model, device = load_finbert()
            if tokenizer is None:
                print("[step05] ERROR: Could not load FinBERT. Aborting.")
                return

            print(f"\n[step05] Running FinBERT inference on {len(remaining_df):,} headlines ...")
            print(f"         Batch size: {FINBERT_BATCH_SIZE}")

            all_raw_rows = []
            symbols_in_remaining = (remaining_df['Symbol'].dropna().unique()
                                    if 'Symbol' in remaining_df.columns else ['ALL'])
            checkpoint_interval = 50

            for sym_idx, symbol in enumerate(sorted(symbols_in_remaining)):
                if (sym_idx + 1) % 25 == 0 or (sym_idx + 1) == len(symbols_in_remaining):
                    print(f"  [{sym_idx + 1}/{len(symbols_in_remaining)}] Processing: {symbol}")

                sym_df = (remaining_df[remaining_df['Symbol'] == symbol]
                          if 'Symbol' in remaining_df.columns else remaining_df)
                headlines = sym_df['Heading'].tolist()
                dates = sym_df['Date'].tolist() if 'Date' in sym_df.columns else [None] * len(headlines)

                if not headlines:
                    continue

                scores = score_headlines_batch(
                    headlines, tokenizer, model, device,
                    batch_size=FINBERT_BATCH_SIZE,
                )

                for i, score_dict in enumerate(scores):
                    all_raw_rows.append({
                        'Symbol': symbol,
                        'Date': dates[i] if i < len(dates) else None,
                        'Heading': headlines[i] if i < len(headlines) else None,
                        'positive_prob': score_dict['positive_prob'],
                        'negative_prob': score_dict['negative_prob'],
                        'neutral_prob': score_dict['neutral_prob'],
                        'compound': score_dict['compound'],
                        'confidence': score_dict['confidence'],
                        'label': score_dict['label'],
                    })

                if (sym_idx + 1) % checkpoint_interval == 0 and all_raw_rows:
                    df_new = pd.DataFrame(all_raw_rows)
                    df_combined = pd.concat([df_checkpoint, df_new], ignore_index=True)
                    save_checkpoint(df_combined)
                    print(f"    Checkpoint saved ({len(df_combined):,} total raw rows)")

            if all_raw_rows:
                df_new = pd.DataFrame(all_raw_rows)
                df_all_raw = pd.concat([df_checkpoint, df_new], ignore_index=True)
            else:
                df_all_raw = df_checkpoint.copy()

            save_checkpoint(df_all_raw)
            print(f"\n[step05] Total raw sentiment rows: {len(df_all_raw):,}")
        else:
            df_all_raw = df_checkpoint.copy()

    if df_all_raw.empty:
        print("[step05] ERROR: No sentiment data available.")
        return

    # --- Aggregate to monthly ---
    print("\n[step05] Aggregating sentiment to monthly ...")
    df_monthly = aggregate_sentiment_monthly(df_all_raw, symbol_to_isin)
    print(f"  Monthly sentiment rows: {len(df_monthly):,}")
    print(f"  Unique symbols: {df_monthly['symbol'].nunique()}")
    print(f"  ISINs mapped: {df_monthly['ISIN'].notna().sum()}/{len(df_monthly)} "
          f"({df_monthly['ISIN'].notna().mean()*100:.1f}%)")
    print(f"  Unique months: {df_monthly['year_month_str'].nunique()}")

    coverage_report(df_monthly)

    df_monthly.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[step05] Saved: {OUTPUT_CSV}")
    print(f"         Shape: {df_monthly.shape}")

    print("\n[step05] Sample output (first 5 rows):")
    cols_to_show = ['symbol', 'ISIN', 'year_month_str', 'sentiment_mean',
                    'sentiment_confidence', 'sentiment_extremity',
                    'news_count', 'pct_strongly_positive', 'pct_strongly_negative']
    cols_to_show = [c for c in cols_to_show if c in df_monthly.columns]
    print(df_monthly[cols_to_show].head().to_string(index=False))

    print("\n[step05] Sentiment distribution:")
    valid = df_monthly['sentiment_mean'].dropna()
    print(f"  Mean compound:         {valid.mean():.4f}")
    print(f"  Positive months:       {(valid > 0.05).sum()}")
    print(f"  Negative months:       {(valid < -0.05).sum()}")
    print(f"  Neutral months:        {((valid >= -0.05) & (valid <= 0.05)).sum()}")
    print(f"  Avg news/month:        {df_monthly['news_count'].mean():.1f}")
    print(f"  Avg confidence:        {df_monthly['sentiment_confidence'].dropna().mean():.3f}")
    print(f"  Unreliable (NaN):      {df_monthly['sentiment_mean'].isna().sum()}")
    print(f"  Total features:        {len([c for c in df_monthly.columns if 'sentiment' in c or 'positive' in c or 'negative' in c or 'neutral' in c or 'news' in c])}")

    print("\n[step05] Done.")


if __name__ == '__main__':
    main()