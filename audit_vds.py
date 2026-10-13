"""
audit_vds.py -- Diagnostic audit of the "Volume Divergence Short" MES strategy.

Standalone from backtest_futures.py: this file implements its own signal
generator and trade simulator so the audit has full control over intrabar
resolution, exit structure, and instrumentation (both-touch counting,
component bind rates) that the shared simulate() doesn't expose. Entry/
slippage/session conventions match backtest_futures.py exactly (next-bar-
open fill, 1 tick adverse slippage, forced flat 15:45 ET) so results are
comparable to everything already validated in that file.

Run: python3 audit_vds.py
"""
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_futures import load_cached, build_day_frames, SPECS, ET, COMMISSION_RT

from strategy_params import PARAMS

RNG_SEED = PARAMS["stats"]["rng_seed"]
N_BOOT = PARAMS["stats"]["n_boot"]

# ════════════════════════════════════════════════════════════════════════════
# 1. SIGNAL GENERATION (mirrors backtest_futures.py's Vol Div Short exactly)
# ════════════════════════════════════════════════════════════════════════════

def build_features(sym: str, df: pd.DataFrame, vol_norm: bool = False) -> dict:
    """Precompute per-day feature frames (vwap, day_open, prev_day_low, etc.)
    keyed by day. If vol_norm, also attaches a time-of-day-normalized volume
    series (bar volume / median volume at that same time-of-day over the
    prior 20 sessions, shifted -- no lookahead)."""
    out = {}
    frames = build_day_frames(df)

    # time-of-day normalization needs history across days, built once up front
    if vol_norm:
        all_bars = pd.concat([day for _, day, *_ in frames])
        all_bars = all_bars.copy()
        all_bars["tod"] = all_bars.index.time
        all_bars["day"] = all_bars.index.date
        # for each (day, tod), the median volume at that tod over the prior 20 sessions
        pivot = all_bars.pivot_table(index="day", columns="tod", values="volume", aggfunc="first")
        roll_med = pivot.rolling(20, min_periods=5).median().shift(1)
        norm_lookup = {}
        for d in pivot.index:
            for t in pivot.columns:
                m = roll_med.loc[d, t] if d in roll_med.index else np.nan
                norm_lookup[(d, t)] = m

    prev_low = None
    for d, day, on_high, on_low in frames:
        v = day["volume"].astype(float)
        vwap = (day["close"] * v).cumsum() / v.cumsum()
        vwap_slope = vwap.diff()
        avg_vol = v.rolling(12, min_periods=5).mean()
        vol_ratio = v / avg_vol
        day_open = float(day["open"].iloc[0])

        norm_vol = None
        if vol_norm:
            meds = np.array([norm_lookup.get((d, t), np.nan) for t in day.index.time])
            with np.errstate(invalid="ignore", divide="ignore"):
                norm_vol = v.to_numpy() / meds
            norm_vol = pd.Series(norm_vol, index=day.index)

        sgn = np.sign(day["close"].diff())
        whipsaw = ((sgn != sgn.shift(1)) & (sgn.shift(1) != sgn.shift(2)) & sgn.ne(0)
                   & sgn.shift(1).ne(0) & sgn.shift(2).ne(0))

        out[d] = dict(day=day, vwap=vwap, vwap_slope=vwap_slope, vol_ratio=vol_ratio,
                       day_open=day_open, prev_low=prev_low, norm_vol=norm_vol,
                       whipsaw=whipsaw)
        prev_low = float(day["low"].min())
    return out


def attach_score(cands: pd.DataFrame, feats: dict, aligned_series: "pd.Series | None" = None,
                  min_score: int = 2) -> pd.DataFrame:
    """Attach the 3 optional score components + total score.
    aligned_series: a boolean Series indexed by timestamp, True where the
    reference instrument is bearish-aligned (close below its own VWAP).
    If None, uses MES's own below_vwap column (the self-referential default
    that Task 1 checks for tautology against the weak-day gate)."""
    cands = cands.copy()
    whipsaw_lookup = {}
    for d, f in feats.items():
        for ts, w in f["whipsaw"].items():
            whipsaw_lookup[ts] = bool(w) if not pd.isna(w) else False
    cands["not_whipsaw"] = ~cands["ts"].map(whipsaw_lookup).fillna(False)
    if aligned_series is None:
        cands["aligned"] = cands["below_vwap"]
    else:
        cands["aligned"] = cands["ts"].map(aligned_series).fillna(False)
    t = cands["ts"].dt.time
    cands["time_bonus"] = ((t >= time(9, 30)) & (t <= time(10, 30))) | \
                           ((t >= time(15, 0)) & (t <= time(16, 0)))
    cands["score"] = 1 + cands["aligned"].astype(int) + cands["not_whipsaw"].astype(int) \
                        + cands["time_bonus"].astype(int)
    return cands


# ════════════════════════════════════════════════════════════════════════════
# 2. SIMULATOR (configurable intrabar resolution + exit structure)
# ════════════════════════════════════════════════════════════════════════════

def simulate_trades(cands: pd.DataFrame, bars_by_day: dict, spec: dict,
                     resolution: str = "stop_first", exit_mode: str = "fixed",
                     tgt_r: float = 2.0, eod_flat: time = time(15, 45),
                     max_daily_trades: int = 3, risk_cap: "float | None" = 750.0,
                     rng_seed: int = RNG_SEED,
                     atr_lookback: int = 20, chandelier_mult: float = 2.5) -> pd.DataFrame:
    """
    resolution: "stop_first" (default), "target_first", or "coinflip" (50/50
      when a single bar's range spans both stop and target).
    exit_mode: "fixed" (stop/target as given, tgt_r multiple), "trail_prior_bar"
      (stop trails to prior bar's high once price has moved favorably),
      "chandelier" (stop trails at chandelier_mult * 20-bar ATR from the
      running low since entry), "be_then_trail" (move stop to breakeven at 1R,
      then trail prior-bar-high from there).
    risk_cap: reject signals whose stop distance * point_value exceeds this
      (None disables the cap entirely).
    Bias is always -1 (short only) for this audit.
    """
    rng = np.random.default_rng(rng_seed)
    tick, pv = spec["tick"], spec["point_value"]
    trades = []
    daily_count = {}
    open_until = {}  # sym -> ts of open position's exit (one at a time per sym)

    for _, s in cands.sort_values("ts").iterrows():
        d = s["day"]
        if daily_count.get(d, 0) >= max_daily_trades:
            continue
        if s["sym"] in open_until and s["ts"] < open_until[s["sym"]]:
            continue
        day = bars_by_day[d]
        i = int(s["bar_idx"])
        if i + 1 >= len(day):
            continue
        entry_bar = day.iloc[i + 1]
        entry = float(entry_bar["open"]) - tick  # short: adverse = lower fill
        stop = float(s["stop"])
        risk_pts = stop - entry
        if risk_pts <= tick:
            continue
        risk_dollars = risk_pts * pv
        if risk_cap is not None and risk_dollars > risk_cap:
            continue
        target = entry - risk_pts * tgt_r

        exit_price, exit_ts, exit_reason, both_touch = None, None, None, False
        run_low = entry
        be_active = False
        cur_stop = stop
        for j in range(i + 1, len(day)):
            b = day.iloc[j]
            bt = b["ts"].time() if hasattr(b["ts"], "time") else b["ts"]
            if bt >= eod_flat:
                exit_price, exit_ts, exit_reason = float(b["open"]) + tick, b["ts"], "eod"
                break
            lo, hi = float(b["low"]), float(b["high"])
            run_low = min(run_low, lo)

            if exit_mode == "trail_prior_bar" and j > i + 1:
                prior_hi = float(day.iloc[j - 1]["high"])
                cur_stop = min(cur_stop, prior_hi)
            elif exit_mode == "chandelier":
                atr_window = day.iloc[max(0, j - atr_lookback):j]
                if len(atr_window) >= 2:
                    tr = (atr_window["high"] - atr_window["low"]).abs()
                    atr = tr.mean()
                    cur_stop = min(cur_stop, run_low + chandelier_mult * atr)
            elif exit_mode == "be_then_trail":
                if not be_active and (entry - lo) >= risk_pts:
                    be_active = True
                    cur_stop = entry
                if be_active and j > i + 1:
                    prior_hi = float(day.iloc[j - 1]["high"])
                    cur_stop = min(cur_stop, prior_hi)

            stop_hit = hi >= cur_stop
            tgt_hit = lo <= target
            if stop_hit and tgt_hit:
                both_touch = True
                if resolution == "stop_first":
                    hit_stop = True
                elif resolution == "target_first":
                    hit_stop = False
                else:  # coinflip
                    hit_stop = rng.random() < 0.5
            elif stop_hit:
                hit_stop = True
            elif tgt_hit:
                hit_stop = False
            else:
                continue

            if hit_stop:
                exit_price = cur_stop + tick  # adverse: short's buy-to-cover fills worse (higher)
                exit_reason = "be" if (exit_mode == "be_then_trail" and be_active and cur_stop <= entry) else \
                              ("trail" if exit_mode in ("trail_prior_bar", "chandelier") and cur_stop < stop else "stop")
            else:
                exit_price = target + tick
                exit_reason = "target"
            exit_ts = b["ts"]
            break

        if exit_price is None:
            b = day.iloc[-1]
            exit_price, exit_ts, exit_reason = float(b["close"]), b["ts"], "eod"

        pnl = (entry - exit_price) * pv - COMMISSION_RT  # short: profit when exit < entry
        trades.append(dict(
            sym=s["sym"], day=d, entry_ts=entry_bar["ts"], exit_ts=exit_ts,
            entry=round(entry, 4), exit_price=round(exit_price, 4), stop=round(stop, 4),
            target=round(target, 4), reason=exit_reason, both_touch=both_touch,
            risk_dollars=round(risk_dollars, 2), pnl=round(pnl, 2),
            score=s.get("score", np.nan),
        ))
        daily_count[d] = daily_count.get(d, 0) + 1
        open_until[s["sym"]] = exit_ts

    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════════════════════
# 3. STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def evaluate(trades: pd.DataFrame, rng_seed: int = RNG_SEED, n_boot: int = N_BOOT) -> dict:
    """Full stat block required by the audit's ground rules."""
    n = len(trades)
    if n == 0:
        return dict(n=0, win_pct=np.nan, pf=np.nan, net=0.0, exp_r=np.nan,
                    exp_r_ex_top5=np.nan, tstat=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    winning_days=0, losing_days=0, maxdd=0.0, pct_time_exit=np.nan,
                    both_touch_n=0)

    r = (trades["pnl"] / trades["risk_dollars"]).to_numpy()
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_pct = len(wins) / n * 100
    gross_w, gross_l = wins["pnl"].sum(), -losses["pnl"].sum()
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    net = trades["pnl"].sum()
    exp_r = r.mean()

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

    eq = trades.sort_values("exit_ts")["pnl"].cumsum()
    maxdd = (eq - eq.cummax()).min()

    return dict(
        n=n, win_pct=round(win_pct, 1), pf=round(pf, 2) if pf != float("inf") else pf,
        net=round(net, 2), exp_r=round(exp_r, 4), exp_r_ex_top5=round(exp_r_ex_top5, 4),
        tstat=round(tstat, 2), ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
        winning_days=int((daily > 0).sum()), losing_days=int((daily < 0).sum()),
        maxdd=round(maxdd, 2), pct_time_exit=round((trades["reason"] == "eod").mean() * 100, 1),
        both_touch_n=int(trades["both_touch"].sum()),
    )


def print_eval(label: str, st: dict) -> None:
    print(f"  {label}")
    print(f"    n={st['n']}  win%={st['win_pct']}  PF={st['pf']}  net=${st['net']}  "
          f"expR={st['exp_r']}  expR(ex-top5)={st['exp_r_ex_top5']}")
    print(f"    t-stat={st['tstat']}  95% CI=[{st['ci_lo']}, {st['ci_hi']}]  "
          f"win-days={st['winning_days']}  lose-days={st['losing_days']}  "
          f"maxDD=${st['maxdd']}  time-exit%={st['pct_time_exit']}  both-touch-n={st['both_touch_n']}")


def bind_rate(before: pd.DataFrame, after: pd.DataFrame) -> float:
    """% of 'before' rows removed by whatever filter produced 'after'."""
    if len(before) == 0:
        return float("nan")
    return round((1 - len(after) / len(before)) * 100, 1)


def generate_candidates(sym: str, df: pd.DataFrame, feats: dict,
                         vol_col: str = "volume", vwap_warmup: int = 0,
                         vwap_slope_gate: bool = False) -> pd.DataFrame:
    """All raw pattern candidates (higher-high + red + >=3-of-4 volume decline),
    tagged with every gate component as its own boolean/value column so
    downstream code can turn gates on/off without regenerating. vol_col:
    "volume" (raw) or "norm_vol" (time-of-day normalized, see build_features).
    vwap_warmup: bars into the session before the weak-day gate is allowed to
    fire at all (session VWAP needs bars to be meaningful)."""
    recs = []
    for d, f in feats.items():
        day, vwap, vwap_slope, vol_ratio, day_open = (
            f["day"], f["vwap"], f["vwap_slope"], f["vol_ratio"], f["day_open"])
        vseries = f["norm_vol"] if vol_col == "norm_vol" else day["volume"].astype(float)
        arr = day.reset_index()
        vwap_a, slope_a, vol_a = vwap.to_numpy(), vwap_slope.to_numpy(), vseries.to_numpy() if vseries is not None else None
        if vol_a is None:
            continue
        for i in range(4, len(arr)):
            r = arr.iloc[i]
            v5 = vol_a[i - 4:i + 1]
            if np.isnan(v5).any():
                continue
            declines = int(sum(v5[k] < v5[k - 1] for k in range(1, 5)))
            highs5 = day["high"].iloc[i - 4:i + 1]
            higher_high = bool(highs5.iloc[-1] > highs5.iloc[0])
            red = bool(r["close"] < r["open"])
            cur_vwap = vwap_a[i]
            below_vwap = bool(r["close"] < cur_vwap) if not np.isnan(cur_vwap) else False
            below_open = bool(r["close"] < day_open)
            weak_day = below_vwap and below_open and (i >= vwap_warmup)
            slope_neg = bool(slope_a[i] < 0) if not np.isnan(slope_a[i]) else False
            if vwap_slope_gate:
                weak_day = weak_day and slope_neg
            in_window = time(9, 30) <= r["ts"].time() <= time(12, 0)
            recs.append(dict(
                sym=sym, day=d, ts=r["ts"], bar_idx=i,
                open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                close=float(r["close"]), volume=float(r["volume"]),
                stop=float(highs5.max()),
                declines=declines, higher_high=higher_high, red=red,
                below_vwap=below_vwap, below_open=below_open, weak_day=weak_day,
                vwap_slope_neg=slope_neg, in_window=in_window,
                vwap=cur_vwap, day_open=day_open,
                prev_day_low=f["prev_low"],
            ))
    return pd.DataFrame(recs).sort_values("ts").reset_index(drop=True) if recs else pd.DataFrame(recs)
