# 📈 Stock Movement Prediction with News Sentiment Analysis

Predicting next-day price direction for India's **NIFTY Top 10 stocks** using a bidirectional LSTM enhanced with FinBERT-derived news sentiment features. The project demonstrates that incorporating NLP-based sentiment signals yields a statistically significant improvement over a technical-indicators-only baseline.

## Key Results

| Metric | Baseline (Technical Only) | Sentiment-Enhanced | Δ |
|--------|--------------------------|-------------------|---|
| **Avg. Accuracy** | 52.75% | 55.36% | **+2.32%** |
| **Avg. Precision** | 52.57% | 54.94% | +2.37% |
| **Avg. F1 Score** | 60.82% | 59.99% | −0.83% |

- **Statistically significant** — paired t-test p=0.004, Wilcoxon p=0.008
- **Large effect size** — Cohen's d = 1.19
- Improvement in **6 out of 10 stocks**, strongest in Banking and Financial Services
- Best single-stock improvement: **INFY (+7.43%)**

### Per-Stock Accuracy Comparison

| Stock | Sector | Baseline | Sentiment | Change |
|-------|--------|----------|-----------|--------|
| RELIANCE | Energy | 52.92% | 52.58% | −0.34% |
| HDFCBANK | Banking | 52.58% | 57.04% | **+4.47%** |
| TCS | IT | 56.36% | 56.36% | 0.00% |
| BHARTIARTL | Telecom | 49.83% | 52.58% | **+2.75%** |
| ICICIBANK | Banking | 53.95% | 56.70% | **+2.75%** |
| SBIN | Banking | 57.04% | 58.42% | **+1.37%** |
| INFY | IT | 50.17% | 54.64% | **+4.47%** |
| KOTAKBANK | Banking | 54.30% | 56.01% | **+1.72%** |
| BAJFINANCE | Financial Services | 47.08% | 53.95% | **+6.87%** |
| HINDUNILVR | FMCG | 53.26% | 55.33% | **+2.06%** |

## Architecture

```
                    ┌─────────────────┐
                    │   GNews API     │
                    │  (News Headlines)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    FinBERT      │
                    │ (ProsusAI/finbert)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Daily Sentiment │
                    │   Aggregation   │──────────────┐
                    └─────────────────┘              │
                                                     │
┌─────────────────┐  ┌─────────────────┐   ┌────────▼────────┐
│   yfinance      │──▶ Technical Feature│──▶│  Merged Dataset │
│ (OHLCV Prices)  │  │   Engineering   │   │ (Left Join + FFill)│
└─────────────────┘  └─────────────────┘   └────────┬────────┘
                                                     │
                                            ┌────────▼────────┐
                                            │  Bidirectional  │
                                            │   LSTM Model    │
                                            │ (2-layer, h=64) │
                                            └────────┬────────┘
                                                     │
                                            ┌────────▼────────┐
                                            │  Binary Output  │
                                            │  (Up / Down)    │
                                            └─────────────────┘
```

### LSTM Model

Both models share the same `StockLSTM` architecture:
- **2-layer bidirectional LSTM** — hidden size 64, dropout 0.3 between layers
- **Batch normalization** after LSTM output
- **FC layers** — 128 → 32 → 1 with dropout 0.5 and sigmoid activation
- **Sequence length** — 10 trading days
- **Training** — Adam optimizer (lr=0.001), BCELoss, early stopping (patience=10), chronological 80/20 split

### Feature Sets

**Baseline (~37 technical features):**
Log returns, lagged returns (1–5 days), momentum indicators, rolling volatility (5/10/20-day), volume metrics (ratio, rolling avg, change), SMA/EMA ratios, RSI, MACD, Bollinger Band position

**Sentiment-enhanced (baseline + ~12 sentiment features):**
Daily positive/negative/neutral proportions, mean & std sentiment scores, article counts, 3/5/7-day rolling sentiment averages

## Pipeline

The project runs as a sequence of Jupyter notebooks:

| Step | Notebook | Input | Output |
|------|----------|-------|--------|
| 1 | `main.ipynb` | GNews API, yfinance | `Stock News/`, `stock_data_daily/` |
| 2 | `data_processing_stock.ipynb` | Raw stock data | `stock_data_daily_processed/` |
| 3 | `data_processing_news.ipynb` | Raw news data | `daily_sentiment_data/`, `final_sentiment_features/` |
| 4 | `sentiment merge.ipynb` | Processed stock + sentiment | `merged_stock_sentiment_data/` |
| 5 | `baseline_model.ipynb` | Processed stock data | `models/*_best_model.pth`, `baseline_model_results.csv` |
| 6 | `sentiment_model.ipynb` | Merged data | `models/*_sentiment_model.pth`, comparison reports & visualizations |

Steps 2 and 3 can run in parallel. Steps 5 and 6 are independent of each other but both require step 4.

## Setup

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (optional, but recommended for FinBERT and LSTM training)

### Installation

```bash
git clone https://github.com/<your-username>/stock-sentiment-prediction.git
cd stock-sentiment-prediction
pip install torch transformers yfinance gnews scikit-learn pandas numpy matplotlib seaborn tqdm jupyter
```

### Running

```bash
jupyter notebook
```

Run the notebooks in order (1–6) as described in the [Pipeline](#pipeline) section. Each notebook is self-contained and saves its outputs to disk for the next stage.

To run headless:
```bash
jupyter nbconvert --to notebook --execute <notebook.ipynb>
```

## Output Artifacts

| File | Description |
|------|-------------|
| `baseline_model_results.csv` | Per-stock accuracy, loss, precision, recall, F1 for baseline |
| `sentiment_model_results.csv` | Same metrics for sentiment model + improvement stats |
| `final_model_comparison_report.csv` | Side-by-side comparison of both models across all stocks |
| `sentiment_analysis_summary.json` | Full statistical analysis including t-tests, effect sizes, sector breakdown |
| `baseline_vs_sentiment_comparison.png` | Accuracy comparison bar chart |
| `advanced_model_comparison.png` | Multi-metric comparison visualization |
| `sector_analysis.png` | Improvement breakdown by sector |
| `training_consistency_analysis.png` | Training stability analysis |
| `models/` | Saved PyTorch model weights (`.pth`) for all stocks |

## Stocks Covered

All 10 stocks are from the **NIFTY 50** index (NSE India):

| Stock | Sector |
|-------|--------|
| RELIANCE | Energy |
| HDFCBANK | Banking |
| TCS | IT |
| BHARTIARTL | Telecom |
| ICICIBANK | Banking |
| SBIN | Banking |
| INFY | IT |
| KOTAKBANK | Banking |
| BAJFINANCE | Financial Services |
| HINDUNILVR | FMCG |

## License

This project was developed as an academic research project (SET Project).
