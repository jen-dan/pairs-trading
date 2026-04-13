# ETF Pairs Trading: Cointegration, Walk-Forward Testing, and Regime Sensitivity

This repository contains the code and research workflow for my undergraduate capstone project on US ETF pairs trading.

## Research Question

Do cointegration-selected US ETF pairs remain profitable out of sample after realistic transaction costs?

The project investigates whether ETF pairs that appear attractive in a simple screening exercise continue to perform once they are evaluated under a stricter and more realistic framework:
- rolling walk-forward validation,
- repeated pair re-screening,
- bid-ask-based transaction costs,
- and regime-based performance analysis.

## Project Summary

Pairs trading often looks strong in a static in-sample backtest, but that can be misleading.  
This project tests whether that apparent profitability survives when:
1. pairs are re-evaluated repeatedly through time,
2. costs are included,
3. and performance is examined across different market regimes.

The empirical setting uses daily US ETF data from 2010 to 2025.

## Methodology

### Pair Selection
Pairs are screened using a two-stage procedure:
- **Correlation filter:** pairwise correlation greater than 0.80 in the formation window
- **Cointegration filter:** Engle–Granger test with p-value below 0.05

### Spread Construction
For each eligible pair:
- an OLS hedge ratio is estimated,
- the spread is constructed,
- and the spread is normalised into a z-score.

### Trading Rule
Baseline signal:
- Enter when `|z| > 2.0`
- Exit when `|z| < 0.5`

### Walk-Forward Design
The baseline backtest uses:
- 252 trading days for formation
- 63 trading days for out-of-sample trading

Pairs are re-screened in each rolling window, rather than selected once and traded forever.

### Transaction Costs
The primary cost model uses bid-ask half-spread estimates derived from Bloomberg bid and ask fields.

A fixed 5 bps cost assumption is also used as a robustness comparison.

### Performance Metrics
The analysis focuses on:
- annualised return,
- annualised volatility,
- Sharpe ratio,
- max drawdown,
- turnover,
- hit rate,
- average holding period,
- tradable window share

### Regime Analysis
Performance is also compared across:
- Pre-2020 vs 2020+
- High-VIX vs Low-VIX environments

## Main Findings

The main findings of the project are:

- Initial screening identifies several pairs that look attractive on correlation and cointegration metrics.
- However, many of these pairs weaken materially once they are tested in a rolling out-of-sample framework.
- Only a small subset of screened pairs remains mildly profitable after costs.
- For weaker pairs such as IEF–SHY, underperformance appears to be driven more by unstable tradability and regime dependence than by transaction costs alone.
- Even for stronger pairs, profitability is generally modest rather than strong, and tradable opportunities are sparse.

## Repository Structure

```text
pairs-trading/
│
├── data_raw/            # raw input files used during local development
├── data_processed/      # cleaned / transformed datasets and exported result tables
├── notebooks/           # research notebooks
├── README.md
└── .gitignore
```

## Notebook Workflow

The main workflow is currently notebook-based.

Typical sequence:

1. Load and reshape raw Bloomberg-style ETF data
2. Clean the dataset and construct mid prices / spread-cost proxies
3. Screen candidate pairs
4. Run walk-forward backtests
5. Compare baseline and robustness variations
6. Run pair-panel analysis
7. Run regime analysis
8. Export summary tables and figures

## Key Outputs

The project generates outputs such as:

- initial screening results,
- pair-level walk-forward summaries,
- trade logs,
- panel comparison tables,
- regime summary tables,
- equity curve charts,
- Sharpe comparison charts,
- tradable-window-share charts.

## Data Availability

This project relies on Bloomberg-derived ETF data.

Because Bloomberg data cannot be redistributed, the repository is intended to provide:

- the research workflow,
- the backtesting logic,
- and the analysis pipeline,

rather than a fully self-contained public data package.

If you want to reproduce the analysis, you should use your own data export with the same column structure.

## Reproducibility Notes

To reproduce the workflow, you should prepare daily ETF data with fields equivalent to:

- date
- ticker
- last price
- bid
- ask
- volume

The backtest logic assumes:

- daily frequency,
- aligned ETF price series,
- and bid/ask availability or an appropriate fallback procedure for missing spread observations.

## Current Limitations

This repository is part of an academic research project, so several limitations remain:

- no full market-impact model,
- no portfolio optimisation across pairs,
- no formal structural break model beyond regime comparison,
- and no redistribution of the underlying commercial dataset.

## Future Improvements

Potential next steps include:

- formal structural break detection,
- richer execution-cost modelling,
- portfolio construction across multiple eligible pairs,
- and extension to alternative ETF universes or higher-frequency data.