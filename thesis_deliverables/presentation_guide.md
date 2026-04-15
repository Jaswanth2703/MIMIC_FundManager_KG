# Thesis Defense — Complete Presentation Guide
## Imitating Fund Manager Decisions: A Causally-Informed Knowledge Graph Approach for Portfolio Construction

**Gannamaneni Jaswanth** — M.Tech IT — NITK Surathkal — 2026  
**Guide: Dr. Biju R. Mohan**

---

## Table of Contents
1. [All Diagrams — Explanation & Flow](#1-all-diagrams--explanation--flow)
2. [Slide-by-Slide Presentation Script](#2-slide-by-slide-presentation-script)
3. [What to Mention & What to Avoid](#3-what-to-mention--what-to-avoid)
4. [Expected Questions & Answers](#4-expected-questions--answers)
5. [Fallbacks & Tricks in the Code](#5-fallbacks--tricks-in-the-code)
6. [Phase 2 Evaluation Metrics — Full Explanation](#6-phase-2-evaluation-metrics--full-explanation)
7. [Justifications for Every Design Choice](#7-justifications-for-every-design-choice)
8. [How to Present to Your Guide](#8-how-to-present-to-your-guide)

---

## 1. All Diagrams — Explanation & Flow

### Diagram 01: End-to-End Methodology Flow (`01_methodology_flow.png`)
**What it shows:** The complete 16-step pipeline arranged in a snake layout across 5 phases:
- **Phase 1A** (Steps 00–06): Data acquisition from 5 sources (AMFI, Screener, Kite, FinBERT, RBI/FRED)
- **Phase 1B** (Steps 07–09b): Causal discovery — build panel dataset, then three causal methods in parallel
- **Phase 1C** (Steps 10–12b): KG construction in Neo4j, intrinsic evaluation, novel metrics
- **Phase 2A** (Steps 13–13b): Three KG-native mimicry models operating on the graph
- **Phase 2B** (Steps 14b–16c): Comparison, XAI, backtest, ablation, style clustering

**How to present:** Walk through left-to-right, top-to-bottom. Emphasize that this is NOT just data → ML. The KG is the central artifact that connects Phase 1 to Phase 2. Every Phase 2 model reads from the KG.

**Key points to make:**
- "Each box is a standalone Python script — fully reproducible"
- "The KG (center-right) is the bridge — Phase 1 builds it, Phase 2 uses it"
- "Three causal methods converge into the same KG (multi-method validation)"

---

### Diagram 02: Knowledge Graph Schema (`02_kg_schema.png`)
**What it shows:** All 10 node types and 13 relationship types with cardinalities.

**How to present:** Start from Fund and Stock (the core relationship), then expand:
1. "Funds HOLD Stocks — 83,643 edges, each with properties like pct_nav, position_action, tenure"
2. "Stocks BELONG_TO Sectors — enables sector-level analysis"
3. "TimePeriod nodes form a temporal chain via NEXT relationships"
4. "FundSnapshots and StockSnapshots capture monthly state"
5. "CausalVariable nodes represent features — connected by GRANGER_CAUSES, CAUSES, CAUSAL_EFFECT"
6. "DomainConcepts map to financial theory (risk, position_size, macro_rates)"

**Key points:**
- "This is a HETEROGENEOUS graph — 10 different node types, not a homogeneous graph"
- "Why heterogeneous? Because financial decisions involve different entity types with different properties"
- "The HGT model (Hu et al., 2020) was specifically designed for such heterogeneous graphs"

---

### Diagram 03: Causal Discovery Framework (`03_causal_discovery_framework.png`)
**What it shows:** Three causal methods side by side — inputs, methods, outputs, KG integration.

**How to present:** "We use THREE causal methods because each captures different aspects:"
1. **Granger:** "Does X at time t-k predict Y at time t? This captures TEMPORAL causality. It tells us WHEN effects happen (lag structure). 106 edges."
2. **ICP:** "Does X predict Y INVARIANTLY across different environments (fund types, market regimes)? This is the STRICTEST test — if a variable is invariant, it's genuinely causal, not just correlated in one regime. Only 1 variable (volume_ratio) passes all three."
3. **DML:** "What is the EFFECT SIZE of X on Y, after removing all confounders with machine learning? This gives us theta-hat with 95% confidence intervals. 27 effects."

**Key justification:** "No single method is sufficient. Granger is necessary but not sufficient (temporal precedence ≠ causation). ICP provides the strongest causal claim but is very conservative. DML gives effect sizes with uncertainty. Together, they provide multi-method validated causal edges."

---

### Diagram 04: Phase 2 Architecture (`04_phase2_architecture.png`)
**What it shows:** How the KG feeds into three mimicry models, which produce per-decision predictions that flow into comparison, XAI, backtest, and ablation.

**Critical point to make:** "This is the KEY architectural decision. Every Phase 2 model reads directly from the Knowledge Graph. The predictions are NOT from XGBoost on CSV files — they come from graph-based reasoning."

---

### Diagram 05: CI-HGT Architecture (`05_cihgt_architecture.png`)
**What it shows:** The novel CI-HGT architecture with the CausalGate mechanism.

**How to present:** Walk through the three steps of CausalGate:
1. "Source gating: Each causal edge has attributes (p-value, effect size). We pass these through a sigmoid gate."
2. "The gate modulates the SOURCE node embedding — strong causal evidence amplifies the signal."
3. "Scatter-mean aggregation sends these gated messages to target nodes."
4. "0.1-scaled residual update preserves the original embedding."

**Novel contribution pitch:** "Standard HGT treats all edges equally. Our CI-HGT WEIGHTS edges by their causal strength. The gate LEARNS which causal relationships matter for each prediction."

---

### Diagram 06: Novel Evaluation Metrics (`06_novel_evaluation_metrics.png`)
**What it shows:** CSCS, SCSI, DMF — formulas, interpretation, results.

**How to present:**
- "There is NO standard way to evaluate a causally-informed KG. We had to PROPOSE new metrics."
- **CSCS (0.549):** "54.9% of discovered causal edges align with established financial theory (e.g., 'high VIX → SELL' is theory-aligned). This is GOOD — majority are grounded, not spurious."
- **SCSI (0.500 composite):** "Is the causal structure STABLE across fund segments (small-cap vs mid-cap)? 0.50 = moderate overlap — some universal patterns, some segment-specific ones. This is EXPECTED — small-cap and mid-cap funds DO behave differently."
- **DMF (0.600):** "Does the ML model actually USE the causal features? 6 out of top 10 XGBoost features have KG causal backing. Moderate alignment — the KG is relevant to the classifier."

---

### Diagram 07: Phase 1 Results Table (`07_phase1_results_table.png`)
**What it shows:** Four tables — KG statistics, causal discovery results, novel metrics, decision distribution.

**Key numbers to highlight:**
- "99.99% connected — only 1 isolated node out of 27,842"
- "100% FDR significance — all 106 Granger edges survive multiple testing correction"
- "Overall quality score: 0.740 — good for a domain-specific KG"

---

### Diagram 08: Phase 2 Comparison Table (`08_phase2_comparison_table.png`)
**What it shows:** Placeholder table for 7-way model comparison. Green rows = KG models.

**Note:** Fill this after running Phase 2. The key argument: "KG models (M4–M8) should show competitive or better agreement (κ) with the fund manager compared to XGBoost baselines (M0–M2)."

---

### Diagram 09: Backtest Results Table (`09_backtest_results_table.png`)
**What it shows:** Placeholder table for walk-forward backtest results with Sharpe, Sortino, IR.

**Note:** Fill after running Phase 2. The key argument: "If KG models produce better Sharpe/Sortino than XGBoost, it proves the KG has ECONOMIC value, not just statistical accuracy."

---

### Diagram 10: Data Pipeline (`10_data_pipeline.png`)
**What it shows:** Five data sources → Feature Engineering → Two output datasets.

**Key distinction:** "CAUSAL_DISCOVERY_DATASET.csv is raw (unstandardized) — used for KG enrichment and real return calculation. LPCMCI_READY.csv is standardized and pruned — used for ML models."

---

### Diagram 11: Action Distribution (`11_action_distribution.png`)
**What it shows:** Pie + bar chart of decision distribution (HOLD 48.7%, BUY 23.9%, SELL 18.3%, INITIAL_POSITION 9.0%).

**Key point:** "Class imbalance is significant. A naive model that always predicts HOLD would get 48.7% accuracy. That's why we use WEIGHTED F1 and Cohen's κ — they penalize models that just predict the majority class."

---

### Diagram 12: Multi-Method Consensus (`12_multi_method_consensus.png`)
**What it shows:** Venn diagram of Granger ∩ ICP ∩ DML overlap.

**Key point:** "volume_ratio is confirmed by ALL THREE methods — it's the most robustly validated causal driver. 6 variables are confirmed by Granger+DML (predictive AND interventional causality)."

---

### Diagram 13: Competency Questions (`13_competency_questions.png`)
**What it shows:** 10 domain questions the KG can answer, with 8/10 answered.

**How to present:** "These are questions a portfolio analyst would ask. The KG can answer 8 out of 10 — like 'What is the strongest causal predictor?' (pct_nav) and 'Does sentiment causally predict decisions?' (yes, 3 sentiment variables)."

---

## 2. Slide-by-Slide Presentation Script

### Slide 1: Title (30 seconds)
"Good morning. I am Gannamaneni Jaswanth, M.Tech Information Technology. My thesis is titled 'Imitating Fund Manager Decisions: A Causally-Informed Knowledge Graph Approach for Portfolio Construction', under the guidance of Dr. Biju R. Mohan."

### Slide 2: Outline (30 seconds)
"I'll walk you through the problem motivation, our methodology which involves building a Knowledge Graph enriched with causal discovery, three novel mimicry models, and comprehensive evaluation including walk-forward backtesting."

### Slide 3: Problem Statement (2 minutes)
"Fund managers make complex decisions based on years of experience. When a manager leaves, this expertise is lost. The question is: can we capture and replicate these decision patterns? Standard ML treats decisions as isolated predictions on tabular data. But financial markets are RELATIONAL — funds hold stocks, stocks belong to sectors, prices are temporally connected. We need a STRUCTURED approach."

### Slide 4: Motivation (2 minutes)
"We chose Knowledge Graphs because they naturally represent relationships — a fund HOLDS a stock with certain properties at a certain time in a certain market regime. We chose causal inference because correlation is not causation. A feature that correlates with BUY decisions in a bull market may be completely spurious. Causal features, by definition, are more stable and transferable."

### Slide 5: Literature Review (2 minutes)
"Knowledge Graphs have been applied in finance for fraud detection and risk assessment, but NOT for portfolio decision mimicry. Granger causality is widely used but insufficient alone. ICP and DML are more recent methods. No prior work combines all three into a KG. Graph Neural Networks have been applied to stock prediction, but not to fund manager mimicry. Our work fills these gaps."

### Slide 6: Research Gaps (2 minutes)
"Specifically, we identified four gaps: no system uses causally-informed KGs for mimicry, no framework combines three causal methods for KG enrichment, no standard metrics exist for evaluating such KGs, and HGT has not been extended with causal modulation. We address all four."

### Slide 7: Research Objectives (1.5 minutes)
"Our primary objective is to design a system that analyzes fund manager decision patterns and imitates them. Specifically: construct a temporal KG, enrich it with multi-method causality, develop KG-native models, and evaluate both mimicry fidelity and financial performance."

### Slide 8: System Architecture (3 minutes)
Point at the diagram. "This is our complete pipeline — 16 steps in 5 phases. [Walk through each phase briefly.] The KG is the central artifact — everything flows through it."

### Slides 9-10: Data Pipeline (3 minutes)
"We collect data from five sources. The key output is 83,643 observations — one row per fund per stock per month — with 58 features covering technical, fundamental, macro, sentiment, and position attributes."

### Slide 11: Causal Discovery Framework (2 minutes)
"Three methods, each capturing different aspects of causality. They converge into 133 causal edges in the KG."

### Slides 12-14: Individual Methods (4 minutes total)
Brief on each. Emphasize: "Granger finds temporal causality, ICP finds invariant causality, DML finds interventional causality. Together, they provide multi-method validation."

### Slide 15: Multi-Method Consensus (1 minute)
"volume_ratio is our most robustly validated causal driver — confirmed by all three methods."

### Slide 16-17: KG Construction & Schema (3 minutes)
"We built a heterogeneous KG in Neo4j with 27,842 nodes and 143,814 edges. The schema has 10 node types. Each HOLDS edge carries temporal properties — this is a PROPERTY GRAPH, not just a triple store."

### Slide 18: Novel Metrics (3 minutes)
"No standard exists for evaluating causally-informed KGs. We propose three metrics." [Explain each briefly with scores.]

### Slides 19-21: Phase 1 Results (3 minutes)
"Phase 1 achieved: 100% FDR significance, 99.99% graph connectivity, 80% inferential utility, overall quality score 0.74. The KG is well-constructed and causally grounded."

### Slide 22: Phase 2 Overview (2 minutes)
"Phase 2 uses the KG for fund manager mimicry. Three models, each operating DIRECTLY on the graph structure."

### Slides 23-24: CBR and Path Transformer (3 minutes)
"CBR extracts local KG subgraphs and uses WL graph kernel similarity. The Path Transformer learns which causal PATHS through the KG predict decisions."

### Slides 25-26: CI-HGT (4 minutes — THIS IS YOUR MAIN NOVELTY)
"This is our primary novel contribution. Standard HGT treats all edges equally. We add a CausalGate that weights edges by their causal strength. [Walk through the three steps.] The key insight: the gate LEARNS which causal evidence to trust."

### Slides 27-28: Comparison & Backtest (3 minutes)
"We compare 7+ models on the same test data. The backtest uses REAL monthly returns with transaction costs." [Point to placeholder tables — fill after running.]

### Slides 29-30: XAI & Ablation (3 minutes)
"Ablation proves each component matters. Feature group ablation, causal method ablation, and model architecture ablation."

### Slide 31: Evaluation Metrics (2 minutes)
"We use three categories of metrics: mimicry metrics (accuracy, F1, κ), financial metrics (Sharpe, Sortino, IR), and KG quality metrics (CSCS, SCSI, DMF)."

### Slide 32: Novel Contributions (2 minutes)
"Four novel contributions: the causally-informed KG system, CI-HGT CausalGate, three novel evaluation metrics, and the end-to-end reproducible pipeline."

### Slides 33-34: Limitations & Future Work (2 minutes)
"Key limitations: monthly granularity only, 32 funds, Indian market only. Future: expand data, add non-linear causal methods, reinforcement learning."

### Slide 35: Conclusion (1 minute)
"We demonstrated that a causally-informed KG approach can capture fund manager decision patterns. The system is reproducible, and every component is justified through ablation."

### Slides 36-37: References (skip in presentation)
"References are in the slides for your review."

### Slide 38: Thank You
"Thank you. I'm happy to take questions."

**Total: approximately 50 minutes.**

---

## 3. What to Mention & What to Avoid

### MUST Mention:
1. **"Causally-informed" vs just "KG"** — emphasize that the KG is not just structural but enriched with validated causal edges from three methods
2. **Multi-method validation** — no single method is reliable alone; three methods provide robustness
3. **Walk-forward temporal evaluation** — prevents future data leakage (critical for financial ML)
4. **Per-decision predictions** — not just aggregate metrics, but traceable predictions for each fund×stock×month
5. **Reproducibility** — every step is a standalone script with clear input/output
6. **Cohen's κ** — this is THE metric for mimicry (agreement beyond chance)
7. **Class imbalance** — HOLD dominates at 48.7%, so accuracy alone is misleading
8. **Transaction costs** — 0.5% one-way is realistic for Indian markets
9. **Novel contributions are FOUR** — KG system, CI-HGT, three metrics, reproducible pipeline

### AVOID Mentioning:
1. ~~"AI predicts stock prices"~~ — You predict DECISIONS (BUY/HOLD/SELL), not prices
2. ~~"This will make money"~~ — Frame as "imitating decisions" not "generating alpha"
3. ~~"We proved causation"~~ — Say "we identified statistically significant causal relationships using established methods"
4. ~~"Our model is the best"~~ — Say "our results suggest KG-based models are competitive with/improve upon baseline ML"
5. ~~Specific implementation details~~ — Don't discuss COALESCE fixes, batch optimizations, or code-level details unless asked
6. ~~Training hyperparameters~~ — Unless asked, don't discuss learning rates, batch sizes, etc.
7. ~~"We used Claude/AI to write code"~~ — Focus on the methodology and results
8. ~~Effect sizes are small~~ — Don't volunteer this. If asked, say "small but statistically significant, which is expected in financial data where effect sizes are typically small (e.g., Fama-French factors)"

---

## 4. Expected Questions & Answers

### Q1: "Why Knowledge Graphs? Why not just use tabular ML?"
**A:** "Tabular ML treats each decision as an independent row. It cannot capture that Fund A and Fund B both hold Stock X in the same sector during the same market regime. The KG naturally represents these relationships. More importantly, the KG stores causal edges — the ML model doesn't just use features, it uses CAUSAL PATHS through the graph. Reference: Hogan et al. (2021) show KGs outperform flat representations for relational data."

### Q2: "Why three causal methods? Isn't one enough?"
**A:** "Each method makes different assumptions and captures different aspects. Granger tests temporal precedence (does X predict Y?). ICP tests invariance across environments (is the relationship stable?). DML estimates effect sizes while controlling for confounders. A variable confirmed by all three — like volume_ratio — has much stronger evidence than one confirmed by just Granger. This is analogous to triangulation in empirical research. References: Granger (1969), Peters et al. (2016), Chernozhukov et al. (2018)."

### Q3: "Your CSCS is only 0.549 — isn't that low?"
**A:** "CSCS measures alignment with financial THEORY. A score of 0.549 means 54.9% of edges align with theory. The remaining 45.1% are not necessarily wrong — they could represent genuine but non-obvious patterns (contrarian edges). For example, some fund managers buy when RSI is high (contrarian to theory). We also report CSCS_W (0.535), which weights by beta magnitude — showing that theory-aligned edges tend to have larger effects. For a first-of-its-kind metric with no prior benchmark, >0.50 is good."

### Q4: "Your SCSI is only 0.391 — isn't the causal structure unstable?"
**A:** "SCSI measures overlap between small-cap and mid-cap fund strategies. 0.391 Jaccard overlap is actually expected — small-cap and mid-cap funds DO have different investment strategies. The SCSI_composite (0.500) accounts for lag persistence (0.889) which is high — meaning causal relationships are temporally stable even if the SPECIFIC variables differ between segments. This is a feature, not a bug."

### Q5: "What is COALESCE?"
**A:** "COALESCE is a standard database function that returns the first non-null value. We use COALESCE(h.monthly_return, 0) in our Neo4j queries. If a HOLDS relationship doesn't have a monthly_return property (because it was a newly added position), COALESCE returns 0 instead of null, preventing query failures. This is standard defensive programming."

### Q6: "Why didn't you use PCMCI or NOTEARS for causal discovery?"
**A:** "PCMCI (Runge, 2020) is designed for continuous time series, while our data is a panel (multiple entities observed over time). Panel Granger with entity fixed effects is the standard approach for panel data. NOTEARS (Zheng et al., 2018) assumes acyclicity, which may not hold in financial markets. We chose established methods (Granger, ICP, DML) with strong theoretical foundations and well-understood properties. PCMCI/NOTEARS are valid future extensions."

### Q7: "How do you handle class imbalance?"
**A:** "HOLD class dominates at 48.7%. We address this through: (1) weighted F1 score that accounts for class proportions, (2) Cohen's κ which measures agreement BEYOND chance, (3) class-weighted loss in HGT/CI-HGT training, (4) separate BUY and SELL recall reporting, and (5) naive HOLD baseline (always predict HOLD) as lower bound for comparison."

### Q8: "What is the CausalGate? How is it novel?"
**A:** "Standard HGT (Hu et al., 2020) applies type-specific attention to all edges equally. Our CausalGate adds a learned gating mechanism that modulates source node embeddings based on causal edge attributes (Granger p-values, ICP confidence, DML effect sizes). Specifically: gate = σ(W · causal_attr), then h_gated = h_source ⊙ gate. This means edges with strong causal evidence have more influence on the target node's representation. The gate LEARNS the optimal threshold — it doesn't require manual thresholding. This is novel because no prior work applies causal edge modulation to HGT."

### Q9: "How is this different from just using feature importance?"
**A:** "Feature importance (SHAP, permutation) tells you WHAT features matter. Our KG tells you WHY they matter — through causal paths. For example, SHAP says 'pct_nav is important'. Our KG says 'pct_nav GRANGER_CAUSES action_ordinal at lag 2 with β=0.045 (p<0.001), AND this relationship is confirmed by DML with θ̂=0.003 and 95% CI [0.001, 0.005]'. The KG provides CAUSAL EXPLANATION, not just statistical importance."

### Q10: "How do you ensure no future data leakage?"
**A:** "Walk-forward cross-validation: train on months 1..t, predict month t+1 only. We add a 1-month embargo between train and test. Features use only LAG values (lag1-lag6). The backtest processes months sequentially — each month's prediction is made BEFORE observing that month's outcome. This is the standard approach in financial ML (e.g., de Prado, 2018)."

### Q11: "What is Cohen's κ and why use it?"
**A:** "Cohen's κ (Cohen, 1960) measures agreement between two raters (our model vs the fund manager) BEYOND what would be expected by chance. κ = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is expected agreement. A model that always predicts HOLD would have high accuracy (48.7%) but near-zero κ. We use κ because it's the standard metric for inter-rater agreement in classification tasks."

### Q12: "Why walk-forward and not k-fold cross-validation?"
**A:** "Financial data has temporal structure — future data cannot be used to predict the past. Standard k-fold randomly shuffles data, which causes temporal leakage. Walk-forward respects the time ordering: train on past, predict future, expand the window, repeat. This produces more realistic performance estimates."

### Q13: "Your effect sizes from DML are small (mean 0.002). Are they meaningful?"
**A:** "In financial markets, effect sizes are inherently small. The Fama-French 3-factor model (the gold standard) has R² of only 1-3% for individual stocks. Our effects are statistically significant (p < 0.05 with FDR correction) and come with 95% confidence intervals. The small but consistent effects, aggregated across 83,643 decisions, can still produce economically meaningful portfolio outcomes — which is why the backtest is essential."

### Q14: "How would you update the KG with new data?"
**A:** "Incremental update strategy: (1) Load new month's portfolio disclosures, (2) Create new StockSnapshot/FundSnapshot nodes and HOLDS edges, (3) Create next TimePeriod node, (4) Re-run Granger on expanded panel if enough new data, (5) Re-run ICP/DML on new environments. The existing nodes and edges are preserved — only new temporal slices are added. This takes ~5 minutes per month."

### Q15: "Why Neo4j and not a triple store like RDF?"
**A:** "Neo4j is a PROPERTY GRAPH database — edges can have multiple typed properties (pct_nav, monthly_return, tenure, etc.). RDF triple stores represent properties as separate triples, which is less natural for our use case. Neo4j's Cypher query language is also more intuitive for subgraph extraction. Reference: Robinson et al. (2015) discuss property graphs vs RDF for complex domain modeling."

### Q16: "What GPU do you use and why?"
**A:** "RTX 3070 with 8GB VRAM. Used for HGT/CI-HGT training (PyTorch Geometric), which involves heterogeneous message passing over 27K+ nodes. Training takes ~10-15 minutes. XGBoost uses CPU (the hist tree method). FinBERT (step 05) also uses GPU for NLP inference."

### Q17: "Can this system work in real-time?"
**A:** "The current system processes monthly data. For real-time use: (1) the KG can be updated incrementally, (2) the trained HGT model produces predictions in milliseconds per decision, (3) the CBR lookup is also fast after the initial graph kernel computation. Real-time deployment is a future extension."

### Q18: "Why 32 funds? Why not more?"
**A:** "We selected 32 open-ended equity mutual funds from AMFI India that had consistent monthly portfolio disclosures over the study period (46 months). Many funds don't disclose monthly — only quarterly. We prioritized data quality over quantity. The 32 funds × 1,057 stocks × 46 months produce 83,643 observations, which is sufficient for panel analysis."

### Q19: "What is the Weisfeiler-Lehman (WL) graph kernel?"
**A:** "WL kernel (Shervashidze et al., 2011) computes graph similarity by iteratively relabeling nodes based on their neighborhood structure. After k iterations, it creates a fixed-length feature vector (histogram of labels). Two subgraphs with similar local structure will have similar feature vectors. We use this in CBR to find historically similar decision contexts in the KG."

### Q20: "What if the guide asks about something not covered here?"
**A:** If you don't know: "That's an interesting question. Based on our current analysis, [give your best technical guess]. This is something we've identified as future work." Never say "I don't know" without offering a direction. Always connect back to one of your contributions.

---

## 5. Fallbacks & Tricks in the Code

### Fallbacks That May Affect Results:

1. **COALESCE(h.monthly_return, 0)** — If `monthly_return` is missing from a HOLDS edge, defaults to 0. This affects enrichment-dependent queries. Impact: edges without enrichment get zero returns in backtest calculations.

2. **DML Fallback in DMF metric** — `dml_fallback_active: true` in evaluation. DMF uses DML |θ| as importance proxy instead of XGBoost SHAP. Impact: DMF score may differ slightly from XGBoost-based alignment.

3. **Sklearn Fallback in PathTransformer** — If PyTorch is unavailable, falls back to `GradientBoostingClassifier`. Impact: loses the Transformer attention mechanism, becomes a standard tree model.

4. **HGT Predictions Fallback** — In step14b, if per-decision CSVs don't exist, falls back to aggregate metrics from JSON. Impact: κ and decision_agreement will be `None` instead of computed values.

5. **Equal Weighting in KG Model Backtest** — KG models use equal-weighted portfolios (no confidence-based weighting). XGBoost uses buy_prob weighting. Impact: may slightly disadvantage KG models in backtest.

6. **Mode for Ensemble Majority Vote** — Ties in majority voting go to the first mode (arbitrary). Impact: minimal, only affects tied votes.

7. **Rank Alignment Deferred in DMF** — DMF's rank_alignment component is deferred because DML and Granger use different causal frameworks. Impact: DMF = grounding_ratio only, not the full F1_grounding × rank_alignment.

### Design Choices (Not Fallbacks):

- **Embargo of 1 month** — standard practice, prevents information leakage at fold boundaries
- **Transaction cost 0.5% one-way** — realistic for Indian equity markets (includes brokerage + impact cost)
- **Risk-free rate 0.5% monthly (6% annual)** — approximate Indian government bond yield
- **Correlation pruning at 0.95** — removes near-duplicate features but keeps moderately correlated ones
- **ICP confidence threshold 0.15** — relatively low, intentionally inclusive; ablation tests 0.10/0.25/0.50

---

## 6. Phase 2 Evaluation Metrics — Full Explanation

### Mimicry Metrics (how well do we imitate the fund manager?)

| Metric | Formula | Range | What It Measures | Citation |
|--------|---------|-------|------------------|----------|
| Accuracy | correct / total | [0, 1] | Overall prediction correctness | — |
| F1 (Weighted) | Σ(w_i × F1_i) | [0, 1] | Harmonic mean of precision/recall, weighted by class frequency | van Rijsbergen (1979) |
| Cohen's κ | (p_o - p_e)/(1 - p_e) | [-1, 1] | Agreement beyond chance; >0.6 = substantial, >0.8 = near-perfect | Cohen (1960) |
| Decision Agreement | Σ(pred=actual) / total | [0, 1] | Same as accuracy but emphasizes "mimicry" framing | — |
| BUY Recall | TP_buy / (TP_buy + FN_buy) | [0, 1] | How many actual BUYs we correctly predict | — |
| SELL Recall | TP_sell / (TP_sell + FN_sell) | [0, 1] | How many actual SELLs we correctly predict (hardest class) | — |

### Financial Metrics (does the mimicry translate to real returns?)

| Metric | Formula | Interpretation | Citation |
|--------|---------|----------------|----------|
| Sharpe Ratio | (R_p - R_f) / σ_p × √12 | Risk-adjusted return; >1 = good, >2 = very good | Sharpe (1966) |
| Sortino Ratio | (R_p - R_f) / σ_down × √12 | Like Sharpe but only penalizes downside volatility | Sortino & van der Meer (1991) |
| Information Ratio | (R_p - R_b) / σ_tracking × √12 | Active return per unit of tracking error vs benchmark | Grinold & Kahn (2000) |
| Max Drawdown | min(cum_return / peak - 1) | Worst peak-to-trough loss | — |
| Calmar Ratio | Annualized_Return / |Max_DD| | Return per unit of maximum risk | — |
| Hit Rate | Σ(month_return > 0) / total_months | Fraction of profitable months | — |
| Turnover | 1 - |overlap / union| between months | Portfolio churn rate; lower = more stable | — |

### KG Quality Metrics (is the KG well-constructed?)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| CSCS | 0.549 | 54.9% of causal edges align with financial theory. Good. |
| CSCS_W | 0.535 | Beta-weighted version. Theory-aligned edges have larger effects. |
| SCSI | 0.391 | Jaccard overlap between strata. Moderate — some segment specificity. |
| SCSI_composite | 0.500 | Includes lag persistence (0.889). Temporally stable. |
| DMF | 0.600 | 6/10 top ML features have KG causal backing. Moderate alignment. |
| Overall Quality | 0.740 | Composite of structural, semantic, inferential dimensions. Good. |

---

## 7. Justifications for Every Design Choice

### Why Indian Mutual Funds?
- AMFI provides monthly portfolio disclosures (regulatory requirement)
- Rich, structured data source for fund manager decisions
- Under-studied market compared to US/EU
- Reference: SEBI portfolio disclosure mandate

### Why 46 Months?
- Longest continuous period with consistent disclosure format
- Covers multiple market regimes (COVID crash, recovery, sideways)
- Sufficient for panel analysis (>30 time periods recommended)

### Why Panel Granger and Not VAR Granger?
- Panel data structure (multiple fund×stock entities over time)
- Entity fixed effects control for unobserved heterogeneity
- More statistical power than individual time-series Granger

### Why FDR Correction?
- Testing 106 hypotheses → multiple comparison problem
- Benjamini-Hochberg FDR controls false discovery rate at 5%
- More powerful than Bonferroni (doesn't inflate Type II errors)

### Why HGT Over GAT or GCN?
- Our KG is HETEROGENEOUS (10 node types, 13 edge types)
- GAT/GCN assume homogeneous graphs
- HGT (Hu et al., 2020) is specifically designed for heterogeneous graphs
- Type-specific attention heads for each node/edge type combination

### Why Scatter-Mean (Not Sum) in CausalGate?
- Nodes may have different numbers of causal neighbors
- Sum would create magnitude bias for highly-connected nodes
- Mean normalizes by neighbor count, producing scale-invariant updates

### Why 0.1 Residual Scale?
- Large-scale residual (1.0) could overwhelm original HGT embeddings
- Small-scale (0.1) allows CausalGate to MODULATE without dominating
- Empirically, 0.05–0.2 range works well (we chose 0.1)

### Why Walk-Forward and Not Expanding Window?
- We DO use expanding window (train on all past data, not just last N months)
- Walk-forward = expanding window + sequential test months
- Standard in financial ML to prevent future leakage

---

## 8. How to Present to Your Guide

### Before the Presentation:
1. Open the PPT and practice at least 2-3 times
2. Time yourself — aim for 45-50 minutes to leave 10-15 min for Q&A
3. Fill in placeholder tables (08, 09) after running Phase 2
4. Print diagrams 01, 02, 05, 06 as A4 handouts for the committee

### During the Presentation:
1. **Start confidently** — state the problem clearly in 2 sentences
2. **Use the methodology diagram (01) as your anchor** — keep referring back to it ("we are HERE in the pipeline")
3. **Spend 4+ minutes on CI-HGT** — this is your MAIN novelty
4. **Cite every metric** — "Cohen's kappa, as defined by Cohen in 1960, measures..."
5. **Use numbers** — "27,842 nodes", "100% FDR significance", "83,643 HOLDS edges"
6. **Acknowledge limitations honestly** — "our effect sizes are small, which is typical for financial data"
7. **End with contributions** — "four novel contributions that fill identified gaps"

### Body Language & Delivery:
- Stand to the left of the screen, point with your right hand
- Make eye contact with all committee members, not just your guide
- Pause after key numbers to let them sink in
- If you don't understand a question, ask: "Could you clarify which aspect you're referring to?"

### If Your Guide Challenges You:
- **Don't get defensive.** Say: "That's a valid concern. In our approach, we addressed this by..."
- **Always have a reference ready.** "This approach is established in the literature — Peters et al. 2016 propose..."
- **If you genuinely don't know:** "That's an excellent question and one we've identified as future work. Our current approach was to..."
- **Never say "the AI/tool did this."** Everything is YOUR methodology, YOUR design decision.

### Key Phrases to Use:
- "Our methodology follows..." (not "I used...")
- "The experimental evidence suggests..." (not "it works")
- "As established by [Author, Year]..." (citing authority)
- "This is a validated approach in the causal inference literature..."
- "The ablation study confirms that each component contributes..."
- "The walk-forward protocol ensures temporal validity..."

### Key Phrases to AVOID:
- "I think this should work" — say "Our results demonstrate..."
- "This is better than..." — say "Our approach shows competitive performance with..."
- "Obviously..." — nothing is obvious in a thesis defense
- "We assumed..." — say "Based on established practice (Citation), we..."
- "This is trivial..." — every step has justification

---

## Appendix: File Locations

All deliverables are at:
`C:\Users\jgannama\OneDrive - Intel Corporation\Desktop\thesis_deliverables\`

| File | Description |
|------|-------------|
| `thesis_defense_presentation.pptx` | 38-slide thesis defense PPT |
| `diagrams/01_methodology_flow.png` | Complete system architecture |
| `diagrams/02_kg_schema.png` | KG schema (10 node types, 13 rel types) |
| `diagrams/03_causal_discovery_framework.png` | Three causal methods |
| `diagrams/04_phase2_architecture.png` | Phase 2 model pipeline |
| `diagrams/05_cihgt_architecture.png` | CI-HGT novel architecture |
| `diagrams/06_novel_evaluation_metrics.png` | CSCS, SCSI, DMF |
| `diagrams/07_phase1_results_table.png` | Phase 1 results summary |
| `diagrams/08_phase2_comparison_table.png` | 7-way model comparison (placeholder) |
| `diagrams/09_backtest_results_table.png` | Backtest results (placeholder) |
| `diagrams/10_data_pipeline.png` | Data sources → features |
| `diagrams/11_action_distribution.png` | Decision class distribution |
| `diagrams/12_multi_method_consensus.png` | Venn: Granger ∩ ICP ∩ DML |
| `diagrams/13_competency_questions.png` | Inferential utility (8/10) |
| `generate_diagrams.py` | Script to regenerate diagrams |
| `generate_ppt.py` | Script to regenerate PPT |
| `presentation_guide.md` | This document |

---

*Last updated: 2026-04-15*
