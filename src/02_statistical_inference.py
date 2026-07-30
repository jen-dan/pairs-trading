"""
02_statistical_inference.py
---------------------------------
Statistical inference on the out-of-sample Sharpe ratios of the top-10
cointegration-selected ETF pairs.

Adds the evaluation of the performance metrics themselves:
  - stationary-bootstrap 95% CI for the annualised Sharpe ratio (Politis & Romano, 1994)
  - bootstrap one-sided p-value for H0: SR <= 0
  - Mertens (2002) non-normality-adjusted standard error -> t-stat, p-value
  - Probabilistic Sharpe Ratio  PSR(0)   (Bailey & Lopez de Prado, 2014)
  - Deflated Sharpe Ratio       DSR      (adjusts for the number of trials)
  - block-length sensitivity of the CI for the best pair
  - forest plot of Sharpe +/- 95% CI

The backtest logic reproduces notebooks/01_data_cleaning.ipynb, so the point
Sharpe ratios match the dissertation's Table 1 (validated: XLB-XLI = 0.389).

Run from anywhere:  python src/02_statistical_inference.py
Outputs go to data_processed/inference/ and assets/.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.tsa.stattools import coint

# --- paths (relative to the repository, so the script is portable) ---------
REPO = Path(__file__).resolve().parents[1]          # .../code
DP = REPO / "data_processed"
ASSETS = REPO / "assets"
OUT = DP / "inference"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260730)
ANNUALIZATION = 252


# ---------------------------------------------------------------------------
# 1. Backtest logic (trimmed copy from the notebook -> daily net P&L series)
# ---------------------------------------------------------------------------
def build_signal_positions(zscores, entry_z=2.0, exit_z=0.5):
    pos, current = [], 0
    for z in zscores:
        if current == 0:
            if z >= entry_z:
                current = -1
            elif z <= -entry_z:
                current = 1
        else:
            if abs(z) <= exit_z:
                current = 0
        pos.append(current)
    return pd.Series(pos, index=zscores.index, dtype=float)


def daily_net_returns(pair_panel, formation_window=252, trading_window=63,
                      entry_z=2.0, exit_z=0.5, cost_model="bid_ask", fixed_bps=5,
                      corr_threshold=0.80, coint_pvalue_threshold=0.05):
    """Walk-forward backtest -> concatenated daily net_pair_ret series + n_trades."""
    chunks, start = [], 0
    while start + formation_window + trading_window <= len(pair_panel):
        formation = pair_panel.iloc[start:start + formation_window]
        trading = pair_panel.iloc[start + formation_window:start + formation_window + trading_window].copy()

        corr = formation["x_price"].corr(formation["y_price"])
        tradable, pvalue = True, np.nan
        if pd.isna(corr) or corr < corr_threshold:
            tradable = False
        else:
            try:
                _, pvalue, _ = coint(formation["y_price"], formation["x_price"])
            except Exception:
                pvalue = np.nan
            if pd.isna(pvalue) or pvalue >= coint_pvalue_threshold:
                tradable = False

        if tradable:
            x = formation["x_price"].values
            y = formation["y_price"].values
            beta, alpha = np.polyfit(x, y, 1)
            f_spread = formation["y_price"] - (alpha + beta * formation["x_price"])
            t_spread = trading["y_price"] - (alpha + beta * trading["x_price"])
            s_mean, s_std = f_spread.mean(), f_spread.std()
            if s_std == 0 or np.isnan(s_std):
                tradable = False

        if tradable:
            x_ret = trading["x_price"].pct_change().fillna(0)
            y_ret = trading["y_price"].pct_change().fillna(0)
            z = (t_spread - s_mean) / s_std
            sig = build_signal_positions(z, entry_z, exit_z)
            held = sig.shift(1).fillna(0)
            trade_size = held.diff().abs().fillna(held.abs())
            gross = held * ((y_ret - beta * x_ret) / (1 + abs(beta)))
            w_y, w_x = 1 / (1 + abs(beta)), abs(beta) / (1 + abs(beta))
            rate = (w_y * trading["y_cost"] + w_x * trading["x_cost"]) if cost_model == "bid_ask" else fixed_bps / 10000
            net = gross - trade_size * rate
            out = pd.DataFrame({"net": net, "held": held}, index=trading.index)
        else:
            out = pd.DataFrame({"net": 0.0, "held": 0.0}, index=trading.index)
        chunks.append(out)
        start += trading_window

    wf = pd.concat(chunks).sort_index()
    held = wf["held"].values
    n_trades = int(np.sum((held[1:] != 0) & (held[:-1] == 0))) + (1 if len(held) and held[0] != 0 else 0)
    return wf["net"], n_trades


# ---------------------------------------------------------------------------
# 2. Inference tools
# ---------------------------------------------------------------------------
def sharpe_annual(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return np.nan if sd == 0 else r.mean() / sd * np.sqrt(ANNUALIZATION)


def stationary_bootstrap_idx(n, mean_block, rng):
    """Politis & Romano (1994) stationary bootstrap index vector."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def bootstrap_sharpe(r, mean_block=21, B=10000, rng=RNG):
    r = np.asarray(r, float)
    n = len(r)
    boot = np.array([sharpe_annual(r[stationary_bootstrap_idx(n, mean_block, rng)]) for _ in range(B)])
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi, float(np.mean(boot <= 0)), boot.std()


def mertens_tstat(r):
    """Non-normality-adjusted SE of the per-observation Sharpe (Mertens, 2002)."""
    r = np.asarray(r, float)
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g = stats.skew(r, bias=False)
    k = stats.kurtosis(r, fisher=False, bias=False)
    se = np.sqrt((1 + 0.5 * sr**2 - g * sr + (k - 3) / 4 * sr**2) / n)
    t = sr / se
    return t, 2 * (1 - stats.norm.cdf(abs(t))), g, k


def psr(r, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio: P(true SR > benchmark) (Bailey & Lopez de Prado, 2014)."""
    r = np.asarray(r, float)
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g = stats.skew(r, bias=False)
    k = stats.kurtosis(r, fisher=False, bias=False)
    denom = np.sqrt(1 - g * sr + (k - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(best_r, all_sr_per_obs, n_trials):
    """DSR = PSR against the expected maximum Sharpe under N trials."""
    sr_trials = np.asarray(all_sr_per_obs, float)
    sr_trials = sr_trials[~np.isnan(sr_trials)]
    var_sr = sr_trials.var(ddof=1)
    gamma = 0.5772156649015329
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * z1 + gamma * z2)
    return psr(best_r, sr_benchmark=sr0), sr0


# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------
def main():
    clean = pd.read_csv(DP / "etf_daily_tidy.csv", parse_dates=["date"]).sort_values(["date", "ticker"])
    all_mid = clean.pivot(index="date", columns="ticker", values="mid").sort_index()
    all_hs = clean.pivot(index="date", columns="ticker", values="half_spread_rate").sort_index()
    top = pd.read_csv(DP / "pair_screening_results.csv").head(10).reset_index(drop=True)

    rows, daily_series, sr_per_obs_all = [], {}, []
    for _, row in top.iterrows():
        t1, t2 = row["ticker_1"], row["ticker_2"]
        panel = pd.concat([
            all_mid[t1].rename("x_price"), all_mid[t2].rename("y_price"),
            all_hs[t1].rename("x_cost"), all_hs[t2].rename("y_cost"),
        ], axis=1).dropna().sort_index()

        net, n_trades = daily_net_returns(panel)
        daily_series[row["pair"]] = net
        r = net.values
        sr = sharpe_annual(r)
        sr_per_obs_all.append(r.mean() / r.std(ddof=1))

        lo, hi, p_boot, _ = bootstrap_sharpe(r, mean_block=21, B=10000)
        t, p_m, g, k = mertens_tstat(r)
        rows.append({
            "pair": row["pair"].replace(" US Equity", ""),
            "n_obs": len(r), "n_trades": n_trades, "sharpe": round(sr, 3),
            "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
            "boot_p_1sided": round(p_boot, 3),
            "mertens_t": round(t, 2), "mertens_p_2sided": round(p_m, 3),
            "skew": round(g, 2), "kurtosis": round(k, 1),
            "PSR_gt0": round(psr(r), 3),
            "sig_5pct": "yes" if lo > 0 else "no",
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sharpe_inference_top10.csv", index=False)

    best_idx = int(np.nanargmax([x["sharpe"] for x in rows]))
    best_pair = rows[best_idx]["pair"]
    best_r = list(daily_series.values())[best_idx].values
    dsr10, sr0_10 = deflated_sharpe(best_r, sr_per_obs_all, n_trials=10)
    dsr300, sr0_300 = deflated_sharpe(best_r, sr_per_obs_all, n_trials=300)

    # --- forest plot ---
    rp = res.sort_values("sharpe").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#2c7fb8" if lo > 0 else "#8a8a8a" for lo in rp.ci95_low]
    for i, r in rp.iterrows():
        ax.plot([r.ci95_low, r.ci95_high], [i, i], color=colors[i], lw=2.4, zorder=2)
    ax.scatter(rp.sharpe, np.arange(len(rp)), color="#08306b", s=42, zorder=3)
    ax.axvline(0, color="#cb181d", lw=1.3, ls="--", zorder=1)
    ax.set_yticks(np.arange(len(rp)))
    ax.set_yticklabels(rp.pair)
    ax.set_xlabel("Out-of-sample annualised Sharpe ratio (after costs)")
    ax.set_title("Top-10 ETF pairs: Sharpe with 95% stationary-bootstrap CI\n(all intervals include zero)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "sharpe_ci_forest.png", dpi=150)
    fig.savefig(OUT / "sharpe_ci_forest.png", dpi=150)

    # --- block-length sensitivity for best pair ---
    sens = []
    for L in (10, 21, 42):
        lo, hi, p, _ = bootstrap_sharpe(best_r, mean_block=L, B=10000)
        sens.append({"mean_block_days": L, "ci95_low": round(lo, 3), "ci95_high": round(hi, 3), "boot_p_1sided": round(p, 3)})
    pd.DataFrame(sens).to_csv(OUT / "block_length_sensitivity_best_pair.csv", index=False)

    # --- report ---
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print("VALIDATION  XLB-XLI Sharpe (expect 0.389):",
              res.loc[res.pair.str.contains("XLB"), "sharpe"].values[0])
        print()
        print(res.to_string(index=False))
        print()
        print(f"Best pair: {best_pair}")
        print(f"Deflated Sharpe  N=10  : DSR={dsr10:.3f}  (SR0={sr0_10*np.sqrt(252):.3f} ann.)")
        print(f"Deflated Sharpe  N=300 : DSR={dsr300:.3f}  (SR0={sr0_300*np.sqrt(252):.3f} ann.)")
        print("\nBlock-length sensitivity (best pair):")
        print(pd.DataFrame(sens).to_string(index=False))
    print("\nSaved:")
    print(" ", OUT / "sharpe_inference_top10.csv")
    print(" ", OUT / "block_length_sensitivity_best_pair.csv")
    print(" ", ASSETS / "sharpe_ci_forest.png")


if __name__ == "__main__":
    main()
