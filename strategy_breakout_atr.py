"""
strategy_breakout_atr.py -- Breakout/continuation strategy filtered to a
high-ATR volatility regime.

Why this exists: AUDIT_VOL_DIV_SHORT_MES.md ran a 12-part diagnostic audit
on the best-looking mean-reversion pattern (Vol Div Short) and found no
individual pattern component cleared a Bonferroni-corrected significance
bar (~60 configs tested, needed t>=3.3, best was t=2.54). The ONE clean,
monotonic, most-defensible sub-result in the whole audit was a volatility-
regime effect: expectancy rose smoothly from a losing PF 0.60 on low-ATR
days to PF 1.81 (t=2.54) on high-ATR days (Task 11, ATR quartile). The
audit also showed the mean-reversion setup "dies near 2R" -- trailing
stops / letting winners run made it WORSE, confirming reversal rather than
continuation behavior.

This script tests the natural next hypothesis: trade WITH high-ATR days
using continuation mechanics (trend-following trailing exit, no fixed R
target) instead of AGAINST them with mean-reversion mechanics. It reuses
backtest_futures.py's data loading, day-framing, and breakout/ORB signal
detection (ON Break + Opening Range Break -- the two setup families in
that file already tagged kind="breakout"/"orb") so entry logic stays
identical/already-validated. It diverges only on trade management and
adds the regime filter.

No-lookahead guarantees (same conventions as backtest_futures.py):
  * Signals evaluated on bar close; entries fill at the NEXT bar's open,
    +1 tick adverse slippage.
  * Same-bar stop-and-trail ambiguity resolves stop-first (conservative).
  * A day's ATR-regime label is set from a TRAILING window of prior
    sessions' ATR (shift(1) before rolling) -- it never uses that day's
    own range, or any future day, to decide the threshold.
  * Walk-forward split: config grid chosen on the first 70% of sessions,
    judged on the held-out last 30% -- judge the strategy on TEST only.

Run:
    python3 strategy_breakout_atr.py                 # compare all regimes
    python3 strategy_breakout_atr.py --regime high    # just high-ATR days
    python3 strategy_breakout_atr.py --days 250       # limit to last N sessions
"""
import argparse
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_futures import (
    load_cached, build_day_frames, generate_signals, attach_market_alignment,
    SPECS, ET, COMMISSION_RT,
)

from strategy_params import PARAMS

_BA = PARAMS["breakout_atr"]

ATR_BAR_LOOKBACK = _BA["atr_bar_lookback"]       # rolling ATR window
ATR_REGIME_LOOKBACK = _BA["atr_regime_lookback"]    # trailing sessions for the regime threshold
ATR_REGIME_QUANTILE = _BA["atr_regime_quantile"]  # "high" = top quantile vs trailing history
RISK_CAP = _BA["risk_cap"]            # reject signals whose stop distance * point_value exceeds this
EOD_FLAT = __import__("strategy_params").parse_hhmm(_BA["eod_flat"])


# ════════════════════════════════════════════════════════════════════════════
# 1. ATR REGIME (walk-forward safe: trailing quantile, never today's own range)
# ════════════════════════════════════════════════════════════════════════════

def attach_atr(df: pd.DataFrame, lookback: int = ATR_BAR_LOOKBACK) -> pd.DataFrame:
    """Rolling ATR on the continuous (overnight+RTH) series, computed BEFORE
    day-slicing so it's already aligned when build_day_frames splits it --
    same pattern as sma200 in backtest_futures.generate_signals."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return df.assign(atr=tr.rolling(lookback, min_periods=lookback).mean())


def build_atr_regime(bars_by_sym: dict) -> pd.DataFrame:
    """One row per (sym, day): day's mean ATR and a trailing-quantile regime
    label. threshold for day D comes only from the ATR_REGIME_LOOKBACK
    sessions strictly before D (shift(1) precedes the rolling quantile)."""
    rows = []
    for sym, raw in bars_by_sym.items():
        df = attach_atr(raw)
        for d, day, *_ in build_day_frames(df):
            rows.append({"sym": sym, "day": d, "day_atr": float(day["atr"].mean())})
    reg = pd.DataFrame(rows).sort_values(["sym", "day"]).reset_index(drop=True)
    grp = reg.groupby("sym")["day_atr"]
    reg["thresh_hi"] = grp.transform(
        lambda s: s.shift(1).rolling(ATR_REGIME_LOOKBACK, min_periods=20).quantile(ATR_REGIME_QUANTILE)
    )
    reg["thresh_lo"] = grp.transform(
        lambda s: s.shift(1).rolling(ATR_REGIME_LOOKBACK, min_periods=20).quantile(1 - ATR_REGIME_QUANTILE)
    )
    reg["regime"] = np.select(
        [reg["day_atr"] >= reg["thresh_hi"], reg["day_atr"] <= reg["thresh_lo"]],
        ["high", "low"], default="mid",
    )
    reg.loc[reg["thresh_hi"].isna(), "regime"] = "warmup"
    return reg


# ════════════════════════════════════════════════════════════════════════════
# 2. SIGNALS -- reuse breakout/ORB detection from backtest_futures.py,
#    attach the regime label, and let the caller filter by it.
# ════════════════════════════════════════════════════════════════════════════

def build_candidates(bars: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sigs = pd.concat(
        [generate_signals(sym, df) for sym, df in bars.items()], ignore_index=True
    )
    sigs = sigs[sigs["kind"].isin(["breakout", "orb"])].reset_index(drop=True)
    if sigs.empty:
        sys.exit("No breakout/ORB signals generated -- check cached data.")
    sigs = attach_market_alignment(sigs, bars["MES"]).sort_values("ts").reset_index(drop=True)
    reg = build_atr_regime(bars)
    sigs = sigs.merge(reg[["sym", "day", "day_atr", "regime"]], on=["sym", "day"], how="left")
    return sigs, reg


# ════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION -- continuation exit: structural initial stop, then trend
#    trail (chandelier ATR or prior-bar), NO fixed profit target. This is
#    the deliberate opposite of the audit's mean-reversion scale-out, which
#    "dies near 2R" -- here winners are meant to run.
# ════════════════════════════════════════════════════════════════════════════

def simulate_trend(signals: pd.DataFrame, bars: dict, vol_min: float = _BA["vol_min"],
                    exit_mode: str = "chandelier", chandelier_mult: float = 3.0,
                    atr_lookback: int = 20, be_activate_r: float = 1.0,
                    max_daily_trades: int = 4, risk_cap: float = RISK_CAP,
                    eod_flat: time = EOD_FLAT) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    day_bars = {}
    for sym, df in bars.items():
        for d, day, *_ in build_day_frames(df):
            day_bars[(sym, d)] = day.reset_index()

    trades = []
    open_until = {}
    daily_count = {}

    for _, s in signals.sort_values("ts").iterrows():
        if s["vol_ratio"] < vol_min:
            continue
        sym, d = s["sym"], s["day"]
        if daily_count.get((sym, d), 0) >= max_daily_trades:
            continue
        if sym in open_until and s["ts"] < open_until[sym]:
            continue
        day = day_bars.get((sym, d))
        if day is None:
            continue
        i = int(s["bar_idx"])
        if i + 1 >= len(day):
            continue
        spec = SPECS[sym]
        tick, pv = spec["tick"], spec["point_value"]
        bias = int(s["bias"])
        entry_bar = day.iloc[i + 1]
        entry = float(entry_bar["open"]) + bias * tick
        stop = float(s["stop"])
        risk_pts = (entry - stop) * bias
        if risk_pts <= tick:
            continue
        risk_dollars = risk_pts * pv
        if risk_dollars > risk_cap:
            continue

        cur_stop = stop
        be_active = False
        run_extreme = entry
        exit_price = exit_ts = exit_reason = None

        for j in range(i + 1, len(day)):
            b = day.iloc[j]
            bt = b["ts"].time() if hasattr(b["ts"], "time") else b["ts"]
            if bt >= eod_flat:
                exit_price = float(b["open"]) - bias * tick
                exit_ts, exit_reason = b["ts"], "eod"
                break
            lo, hi = float(b["low"]), float(b["high"])
            run_extreme = max(run_extreme, hi) if bias == 1 else min(run_extreme, lo)

            if not be_active and (run_extreme - entry) * bias >= risk_pts * be_activate_r:
                be_active = True
                cur_stop = max(cur_stop, entry) if bias == 1 else min(cur_stop, entry)

            if exit_mode == "chandelier":
                window = day.iloc[max(0, j - atr_lookback):j]
                if len(window) >= 2:
                    atr = (window["high"] - window["low"]).abs().mean()
                    trail = run_extreme - bias * chandelier_mult * atr
                    cur_stop = max(cur_stop, trail) if bias == 1 else min(cur_stop, trail)
            elif exit_mode == "trail_prior_bar" and be_active and j > i + 1:
                prior = day.iloc[j - 1]
                trail = float(prior["low"]) if bias == 1 else float(prior["high"])
                cur_stop = max(cur_stop, trail) if bias == 1 else min(cur_stop, trail)

            stop_hit = (lo <= cur_stop) if bias == 1 else (hi >= cur_stop)
            if stop_hit:
                exit_price = cur_stop - bias * tick
                exit_reason = "be" if (be_active and cur_stop == entry) else "trail"
                exit_ts = b["ts"]
                break
        if exit_price is None:
            b = day.iloc[-1]
            exit_price, exit_ts, exit_reason = float(b["close"]), b["ts"], "eod"

        pnl = (exit_price - entry) * bias * pv - COMMISSION_RT
        trades.append(dict(
            sym=sym, day=d, setup=s["setup"], bias=bias, regime=s["regime"],
            entry_ts=entry_bar["ts"], exit_ts=exit_ts, reason=exit_reason,
            entry=round(entry, 4), exit_price=round(exit_price, 4), stop=round(stop, 4),
            risk_dollars=round(risk_dollars, 2), pnl=round(pnl, 2),
        ))
        daily_count[(sym, d)] = daily_count.get((sym, d), 0) + 1
        open_until[sym] = exit_ts
    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════════════════════
# 4. STATS -- same rigor as audit_vds.py: R-multiples, t-stat, bootstrap CI,
#    ex-top5 sensitivity check (a single fat day shouldn't carry the result).
# ════════════════════════════════════════════════════════════════════════════

def evaluate(trades: pd.DataFrame, rng_seed: int = PARAMS["stats"]["rng_seed"],
             n_boot: int = PARAMS["stats"]["n_boot"]) -> dict:
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net=0.0, exp_r=np.nan,
                    exp_r_ex_top5=np.nan, tstat=np.nan, ci_lo=np.nan, ci_hi=np.nan, maxdd=0.0)
    r = (trades["pnl"] / trades["risk_dollars"]).to_numpy()
    wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
    gross_w, gross_l = wins.pnl.sum(), -losses.pnl.sum()
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    daily = trades.groupby("day")["pnl"].sum().sort_values(ascending=False)
    top5 = set(daily.index[:5])
    ex_top5 = trades[~trades["day"].isin(top5)]
    exp_r_ex_top5 = (ex_top5["pnl"] / ex_top5["risk_dollars"]).mean() if len(ex_top5) else np.nan

    tstat = r.mean() / (r.std(ddof=1) / np.sqrt(n)) if n > 1 and r.std(ddof=1) > 0 else np.nan
    if n > 1:
        rng = np.random.default_rng(rng_seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = r[idx].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_lo, ci_hi = np.nan, np.nan

    eq = trades.sort_values("exit_ts").pnl.cumsum()
    maxdd = (eq - eq.cummax()).min()
    return dict(
        n=n, wr=round(100 * len(wins) / n, 1),
        pf=round(pf, 2) if pf != float("inf") else pf,
        net=round(trades.pnl.sum(), 2), exp_r=round(r.mean(), 4),
        exp_r_ex_top5=round(exp_r_ex_top5, 4) if not pd.isna(exp_r_ex_top5) else np.nan,
        tstat=round(tstat, 2) if not pd.isna(tstat) else np.nan,
        ci_lo=round(ci_lo, 4) if not pd.isna(ci_lo) else np.nan,
        ci_hi=round(ci_hi, 4) if not pd.isna(ci_hi) else np.nan,
        maxdd=round(maxdd, 2),
    )


# ════════════════════════════════════════════════════════════════════════════
# 5. WALK-FORWARD GRID SEARCH per regime bucket
# ════════════════════════════════════════════════════════════════════════════

GRID = dict(
    vol_min=[1.5, 2.0, 2.5],
    chandelier_mult=[2.0, 2.5, 3.0],
    atr_lookback=[14, 20],
    be_activate_r=[0.75, 1.0, 1.5],
)


def run_regime(sigs: pd.DataFrame, bars: dict, all_days: list, split, regime: str,
                quiet: bool = False) -> dict:
    """Grid-search on train days, walk-forward validate top config on test
    days, for signals restricted to one ATR regime bucket ('high','low',
    'mid', or 'all' for no filter)."""
    sub = sigs if regime == "all" else sigs[sigs["regime"] == regime]
    train = sub[sub["day"] <= split]
    test = sub[sub["day"] > split]

    keys = list(GRID)
    results = []
    import itertools
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(keys, combo))
        tr = simulate_trend(train, bars, vol_min=params["vol_min"],
                             chandelier_mult=params["chandelier_mult"],
                             atr_lookback=params["atr_lookback"],
                             be_activate_r=params["be_activate_r"])
        results.append((params, evaluate(tr)))

    min_n = max(10, len(all_days) // 4)
    ranked = sorted(
        [r for r in results if r[1]["n"] >= min_n],
        key=lambda r: (r[1]["pf"] if r[1]["pf"] != float("inf") else 1e9, r[1]["exp_r"]),
        reverse=True,
    ) or sorted(results, key=lambda r: (r[1]["n"], r[1]["exp_r"]), reverse=True)

    if not ranked or ranked[0][1]["n"] == 0:
        return dict(regime=regime, n_candidates=len(sub), **evaluate(pd.DataFrame()))

    best_params = ranked[0][0]
    test_trades = simulate_trend(test, bars, vol_min=best_params["vol_min"],
                                  chandelier_mult=best_params["chandelier_mult"],
                                  atr_lookback=best_params["atr_lookback"],
                                  be_activate_r=best_params["be_activate_r"])
    st = evaluate(test_trades)
    if not quiet:
        print(f"\n── regime={regime} ── train candidates={len(train)}  test candidates={len(test)}")
        print(f"   best train config: {best_params}")
        print(f"   TEST: n={st['n']} WR%={st['wr']} PF={st['pf']} expR={st['exp_r']} "
              f"expR(ex-top5)={st['exp_r_ex_top5']} t={st['tstat']} "
              f"95%CI=[{st['ci_lo']},{st['ci_hi']}] maxDD=${st['maxdd']}")
    return dict(regime=regime, n_candidates=len(sub), best_params=best_params,
                trades=test_trades, **st)


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=5)
    ap.add_argument("--days", type=int, default=None, help="limit to last N sessions")
    ap.add_argument("--regime", type=str, default=None,
                     help="restrict to one regime: high, low, mid, all (default: compare all)")
    args = ap.parse_args()

    bars = load_cached(args.bars)
    if len(bars) < 3:
        sys.exit(f"No cached {args.bars}-min data. Expected files in ./data/")

    sigs, reg = build_candidates(bars)
    all_days = sorted(sigs["day"].unique())
    if args.days:
        all_days = all_days[-args.days:]
        sigs = sigs[sigs["day"].isin(all_days)]
    split = all_days[int(len(all_days) * 0.7) - 1]

    print(f"Sessions: {len(all_days)}  |  train <= {split}  |  test > {split}")
    print(f"Breakout/ORB candidate signals: {len(sigs)}")
    print(f"Regime label counts: {reg['regime'].value_counts().to_dict()}")

    if args.regime:
        run_regime(sigs, bars, all_days, split, args.regime)
        return

    print("\n" + "=" * 78)
    print("COMPARISON: same breakout/ORB signals + continuation exit, filtered")
    print("by ATR regime. This is the test of the audit's regime-effect hypothesis.")
    print("=" * 78)
    summary = []
    for regime in ("all", "high", "mid", "low"):
        r = run_regime(sigs, bars, all_days, split, regime)
        summary.append(r)

    print("\n" + "=" * 78)
    print(f"{'regime':<8}{'n_cand':>8}{'test_n':>8}{'WR%':>7}{'PF':>7}{'expR':>8}"
          f"{'expR_ex5':>10}{'t':>7}{'maxDD$':>9}")
    print("-" * 78)
    for r in summary:
        pf_str = f"{r['pf']}" if r['pf'] != float("inf") else "inf"
        print(f"{r['regime']:<8}{r['n_candidates']:>8}{r['n']:>8}{r['wr']:>7}{pf_str:>7}"
              f"{r['exp_r']:>8}{r['exp_r_ex_top5']:>10}{r['tstat']:>7}{r['maxdd']:>9}")
    print("=" * 78)
    print("\nHow to read this: if 'high' clears PF>=1.3 / t>2 while 'low' and 'mid'")
    print("don't, that supports trading breakouts filtered to high-ATR days. If")
    print("'high' looks no better than 'all', the regime effect didn't transfer")
    print("from mean-reversion mechanics to continuation mechanics -- don't trade it.")

    best = next((r for r in summary if r["regime"] == "high"), None)
    if best is not None and not best.get("trades", pd.DataFrame()).empty:
        best["trades"].to_csv("breakout_atr_trades.csv", index=False)
        print("\nSaved: breakout_atr_trades.csv (high-regime test-window trades)")


if __name__ == "__main__":
    main()
