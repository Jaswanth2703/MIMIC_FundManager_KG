# MIMIC_FundManager_KG

**Imitating Fund Manager Decision-Making via Causal Knowledge Graphs**

An end-to-end pipeline that learns *how* Indian mutual fund managers make stock selection and allocation decisions, then imitates their decision-making **process** — not just outcomes — using a causally-informed Knowledge Graph.

---

## Architecture

```
Phase 1: Data → Features → Causal Discovery → Knowledge Graph → Evaluation
Phase 2: CBR Mimicry → Path Transformer → GNN → Comparison → XAI → Backtest → Ablation
```

### Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 00 | `step00_build_mapping.py` | Build ISIN ↔ ticker mapping |
| 01 | `step01_load_portfolio.py` | Load fund manager portfolio holdings |
| 02 | `step02_merge_fundamentals.py` | Merge fundamental data (PE, PB, etc.) |
| 03 | `step03_fetch_kite_ohlcv.py` | Fetch OHLCV price data via Kite API |
| 04 | `step04_technical_indicators.py` | Compute RSI, MACD, Bollinger, etc. |
| 05 | `step05_finbert_sentiment.py` | FinBERT sentiment on news headlines |
| 06 | `step06_macro_indicators.py` | Macro indicators (VIX, Nifty, FII flows) |
| 07 | `step07_build_causal_dataset.py` | Assemble causal analysis dataset |
| 08 | `step08_feature_engineering.py` | Feature engineering (lags, interactions, standardization) |
| 09 | `step09_targeted_pannel.py` | Panel Granger causality (PanelOLS + FDR) |
| 09a | `step09a_icp_discovery.py` | Invariant Causal Prediction (ICP) parents |
| 09b | `step09b_dml_effects.py` | Double Machine Learning causal effects |
| 10 | `step10_build_temporal_kg.py` | Build temporal Knowledge Graph in Neo4j |
| 11 | `step11_build_causal_kg.py` | Add causal edges (Granger, association) |
| 11b | `step11b_add_icp_dml_to_kg.py` | Add ICP and DML edges to KG |
| 12 | `step12_intrinsic_evaluation.py` | KG intrinsic quality evaluation |
| 12b | `step12b_novel_metrics.py` | Novel KG quality metrics (CSCS, CI, etc.) |
| 13 | `step13_cbr_inference_engine.py` | Case-Based Reasoning via WL graph kernel |
| 13a | `step13a_causal_path_engine.py` | Causal path Transformer mimicry model |
| 13b | `step13b_rgcn.py` | HGT and CI-HGT (Causally-Informed HGT) |
| 14b | `step14b_full_comparison.py` | 5-way model comparison (walk-forward + holdout) |
| 15 | `step15_explainable_ai_v2.py` | XAI: KG multi-hop paths + counterfactuals |
| 16 | `step16_fixed_real_returns.py` | Walk-forward backtest with real returns |
| 16b | `step16b_ablation_study.py` | Ablation study framework |
| 16c | `step16c_style_clustering.py` | Fund manager style clustering |

---

## Novel Contributions

1. **Causally-Informed HGT (CI-HGT)**: A `CausalGate` module that modulates GNN message-passing using causal edge strength — edges discovered by ICP/DML/Granger receive learned attention weights.

2. **Multi-Method Causal Discovery**: Combines Panel Granger Causality, Invariant Causal Prediction (ICP), and Double Machine Learning (DML) — each with different assumptions, providing triangulated causal evidence.

3. **WL Graph Kernel CBR**: Case-Based Reasoning using Weisfeiler-Leman subtree hashing over Neo4j k-hop subgraphs for structural similarity matching.

4. **Causal Path Transformer**: Encodes fund→stock→sector→regime→driver paths as sequences and trains a Transformer to learn the decision *process*, not just predict outcomes.

5. **Novel KG Quality Metrics**: Causal-Structural Coherence Score (CSCS), Temporal Consistency Index, and other domain-specific KG evaluation metrics.

6. **Counterfactual Explanations**: Uses DML θ and Granger β to generate "If X had been different..." explanations grounded in causal relationships.

7. **Fund Manager Style Clustering**: HDBSCAN on path embeddings to discover latent management styles (Value, Momentum, Contrarian, Balanced).

---

## System Requirements

### Hardware
- **CPU**: Any modern multi-core (tested on AMD Ryzen 9 9800X3D)
- **GPU**: NVIDIA RTX 3070 or better (8GB+ VRAM) — for FinBERT, HGT, CI-HGT, PathTransformer
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: ~5GB for data and outputs

### Software
- Python 3.10+
- CUDA Toolkit 12.1+
- Neo4j 5.x (Community or Enterprise)
- Windows 10/11 or Linux

---

## Setup

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/<your-username>/MIMIC_FundManager_KG.git
cd MIMIC_FundManager_KG

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyG (torch-geometric)
pip install torch-geometric torch-sparse torch-scatter

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy template and fill in your credentials
cp .env.example .env
# Edit .env with your API keys (Kite, Gemini, Neo4j)
```

### 3. Set Up Neo4j

1. Install [Neo4j Desktop](https://neo4j.com/download/) or use Docker
2. Create a database named `fundmanager`
3. Set credentials in `.env`:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

### 4. Prepare Data

Place your data files in the appropriate directories:
- `MASTER_CONSOLIDATED_CLEAN_FINAL.csv` — raw portfolio holdings
- `dataset/` — fundamental data files
- `final_news/` — news headline files per ISIN

---

## Running the Pipeline

### Full Pipeline (Recommended)

```bash
# Set encoding (Windows)
set PYTHONIOENCODING=utf-8

# Phase 1: Data processing → Causal discovery → Knowledge Graph
python run_phase1.py

# Phase 2: Mimicry models → Comparison → XAI → Backtest
python run_phase2.py
```

### Individual Steps

```bash
# Run from a specific step
python run_phase1.py --from 5    # Re-run from sentiment analysis
python run_phase2.py --from 13   # Re-run from CBR

# Run a single step
python step05_finbert_sentiment.py
```

### Phase 2 Step Ordering

Steps in Phase 2 must be run in order:
1. `step13` → CBR inference (requires Neo4j)
2. `step13a` → Causal path Transformer
3. `step13b` → HGT / CI-HGT (GPU recommended)
4. `step14b` → 5-way model comparison
5. `step15` → XAI explanations (requires Neo4j)
6. `step16` → Real returns backtest
7. `step16b` → Ablation study
8. `step16c` → Style clustering

---

## Evaluation

### Mimicry Metrics
- **Decision Agreement Rate**: % of stock-months where model matches fund manager
- **Cohen's Kappa**: Agreement beyond chance (>0.4 = moderate, >0.6 = substantial)
- **Buy/Sell Recall**: Directional accuracy for position changes
- **F1 Weighted**: Balanced metric across BUY/HOLD/SELL classes

### Backtest Metrics (step16)
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: Return / max drawdown
- **Hit Rate**: % of profitable months

### Two Evaluation Protocols
1. **Walk-Forward** (expanding window): 24-month initial training, test 1 month at a time
2. **Temporal Holdout**: Train first 65%, test last 35% — single split for comparison

---

## Output Structure

```
data/
├── features/
│   └── LPCMCI_READY.csv          # Main feature dataset
├── causal/
│   ├── all_causal_links.csv       # Granger causality results
│   ├── icp_causal_parents.csv     # ICP parents
│   └── dml_causal_effects.csv     # DML treatment effects
├── kg/
│   ├── intrinsic_evaluation.json  # KG quality metrics
│   └── novel_metrics.json         # Novel KG metrics
└── final/
    ├── cbr_metrics.json           # CBR results
    ├── path_model_metrics.json    # Path Transformer results
    ├── full_comparison.json       # 5-way comparison
    ├── full_comparison.csv        # Comparison table
    ├── xai_explanations.json      # XAI results
    ├── backtest_results.json      # Real returns backtest
    ├── ablation_results.json      # Ablation study
    └── style_clusters.json        # Fund manager styles
```

---

## Ablation Study

The ablation framework (`step16b`) tests:

| Ablation | What It Tests |
|----------|--------------|
| Feature Group | Remove one feature group at a time (fundamental, technical, sentiment, macro, causal, interaction) |
| Causal Method | Compare Granger-only, ICP-only, DML-only, and combinations |
| Model Architecture | LR vs RF vs GBM vs XGBoost vs LightGBM |

Each ablation uses walk-forward evaluation with paired t-tests and Cohen's d for statistical significance.

---

## Statistical Rigor

All comparisons use the `StatisticalTester` class (`utils_stats.py`):
- **Bootstrap CI**: BCa-corrected confidence intervals (N=200 resamples)
- **McNemar's Test**: Pairwise model comparison on classification disagreements
- **Friedman + Nemenyi**: Multi-model comparison with critical difference diagram
- **Cohen's d**: Effect size for practical significance

---

## Configuration

All paths, parameters, and constants are centralized in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BOOTSTRAP_ENABLED` | `True` | Enable bootstrap confidence intervals |
| `NEO4J_URI` | from env | Neo4j connection URI |
| `HOLDOUT_TRAIN_FRAC` | 0.65 | Train/test split for temporal holdout |

---

## Reproducing Results

1. Ensure all data files are in place
2. Start Neo4j and verify connection
3. Run `python run_phase1.py` (takes ~2-4 hours with GPU)
4. Run `python run_phase2.py` (takes ~1-2 hours with GPU)
5. Results are saved as JSON/CSV in `data/final/`

**Important**: If re-running sentiment analysis, delete the checkpoint first:
```bash
del data\sentiment\_finbert_raw_checkpoint.csv
```

---

## License

This project is part of an M.Tech thesis. Please cite appropriately if using any part of this work.

---

## Acknowledgments

- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for financial sentiment analysis
- [Neo4j](https://neo4j.com/) for graph database
- [PyTorch Geometric](https://pyg.org/) for GNN implementations
- Indian mutual fund data from SEBI/AMFI disclosures
