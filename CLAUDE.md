# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLP-based stock movement prediction for the NIFTY Top 10 Indian stocks. The project compares a **baseline LSTM** (technical indicators only) against a **sentiment-enhanced LSTM** (technical + FinBERT news sentiment features) for binary classification of next-day price direction (Up vs Down/Neutral).

**Stocks covered:** RELIANCE, HDFCBANK, TCS, BHARTIARTL, ICICIBANK, SBIN, INFY, KOTAKBANK, BAJFINANCE, HINDUNILVR

## Pipeline & Notebook Execution Order

The project is a sequential Jupyter notebook pipeline. Run in this order:

1. **`main.ipynb`** — Fetches raw news data via GNews API and raw stock price data via `yfinance`. Outputs to `Stock News/` and `stock_data_daily/`.
2. **`data_processing_stock.ipynb`** — Engineers technical features from raw stock data: log returns, lagged returns, momentum, rolling volatility, volume features, SMA/EMA, RSI, MACD, Bollinger Bands. Binary target: `Target = 1` if next-day close > today's close. Outputs to `stock_data_daily_processed/`.
3. **`data_processing_news.ipynb`** — Runs FinBERT (`ProsusAI/finbert`) sentiment analysis on news headlines. Aggregates to daily sentiment scores. Outputs to `daily_sentiment_data/` and `final_sentiment_features/`.
4. **`sentiment merge.ipynb`** — Left-joins processed stock data with sentiment features on date. Forward-fills missing sentiment. Outputs to `merged_stock_sentiment_data/`.
5. **`baseline_model.ipynb`** — Trains per-stock LSTM models on technical features only. Saves models to `models/{TICKER}_best_model.pth` and results to `baseline_model_results.csv`.
6. **`sentiment_model.ipynb`** — Trains per-stock sentiment-enhanced LSTM models on merged data. Saves models to `models/{TICKER}_sentiment_model.pth`. Generates comparison reports, statistical tests, and visualizations.

`model.ipynb` is an earlier single-stock prototype (INFY). The multi-stock notebooks (`baseline_model.ipynb`, `sentiment_model.ipynb`) supersede it.

## Running Notebooks

```bash
# Requires Python 3.12+ with a PyTorch environment (notebooks use kernel "pytorch_env")
jupyter notebook
# Or run headless:
jupyter nbconvert --to notebook --execute <notebook.ipynb>
```

## Key Dependencies

- **PyTorch** (torch, torch.nn) — LSTM models, CUDA-aware
- **transformers** (HuggingFace) — FinBERT sentiment pipeline (`ProsusAI/finbert`)
- **yfinance** — Stock price data download
- **gnews** — News headline fetching
- **scikit-learn** — StandardScaler, train/test split, metrics
- **pandas, numpy, matplotlib, seaborn** — Data processing and visualization

## Architecture Details

### Data Flow
```
Raw News (GNews API) → FinBERT Sentiment → Daily Aggregation → Sentiment Features
Raw Prices (yfinance) → Technical Indicators → Processed Stock Data
                                    ↓                        ↓
                              Merged Dataset (left join on date, forward-fill)
                                    ↓
                    LSTM Models (baseline vs sentiment-enhanced)
```

### LSTM Model Architecture
Both models use the same `StockLSTM` class (defined in each model notebook):
- **2-layer bidirectional LSTM** with hidden_size=64
- Dropout: 0.3 between LSTM layers, 0.5 before output
- Batch normalization after LSTM output
- Fully connected layers: hidden→32→1 with sigmoid activation
- Input: sequences of `seq_length=10` trading days

### Training Configuration
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Loss: BCELoss (binary cross-entropy)
- Early stopping: patience=10, monitoring validation loss
- Train/test split: 80/20 chronological (not random — time series)
- Batch size: 32
- Max epochs: 100

### Feature Sets
- **Baseline (technical only):** ~37 features — log returns, lagged returns, momentum, volatility, volume metrics, SMA/EMA ratios, RSI, MACD, Bollinger Band position
- **Sentiment-enhanced:** Baseline features + ~12 sentiment features — daily positive/negative/neutral proportions, mean/std sentiment scores, article counts, rolling sentiment averages

### Saved Artifacts
- `models/` — Per-stock `.pth` files for both baseline and sentiment models
- `*_results.csv` and `*_comparison.csv` — Metrics per stock
- `sentiment_analysis_summary.json` — Full comparison with statistical tests (paired t-test, Wilcoxon, Cohen's d)
- PNG files — Visualization outputs (comparison charts, sector analysis, training consistency)

## Key Results

Sentiment-enhanced model shows **+2.32% average accuracy improvement** over baseline (statistically significant, p=0.004, Cohen's d=1.19). Improvement observed in 6/10 stocks, strongest in Banking and Financial Services sectors.
