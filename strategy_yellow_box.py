"""
strategy_yellow_box.py -- Time-based Value Area ("Yellow Box") breakout +
retest strategy on 15-min candles.

Strategy spec (as given):
  1. Daily setup: on each session, build a "Yellow Box" = the price range
     where the PRIOR day spent the most time (a TPO / time-based value
     area, standard Market Profile method: equal-weight-per-15min-period
     histogram, expand from the POC until 70% of periods are covered).
     Top of box = VAH, bottom of box = VAL, center = POC.
  2. Target levels: Average Range Resistance = POC + trailing-N-day avg
     daily range (long target); Average Range Support = POC - that avg
     range (short target).
  3. Entry timeframe: 15-min candles.
  4. Entry: a 15-min candle must CLOSE completely outside the box (above
     VAH for long, below VAL for short) -- the breakout candle. Then wait
     for a pullback that RETESTS the broken level (touches back to VAH/
     VAL) and REJECTS (closes back beyond it, in the breakout direction).
     Enter on the next bar's open after that rejection candle (no-
     lookahead: signal decided on bar close, filled next bar open, +1
     tick adverse slippage -- same convention as backtest_futures.py).
  5. Stop: the retest candle's own swing extreme (low for long, high for
     short), required to sit strictly inside the box (VAL < stop < VAH)
     per the spec's "stop loss inside the Yellow Box range" -- if the
     wick blew through the far side of the box, the signal is skipped
     rather than silently placing the stop outside the box.
  6. Target: TP1 = Average Range Resistance/Support. No runner -- the
     "multi-timeframe fractal" / "odds system drops to 10%" runner rules
     in the original spec have no concrete, backtestable definition, so
     this version stops at TP1 (per user decision). Forced flat at
     15:45 ET like the rest of this codebase, since nothing in the spec
     calls for holding overnight.

Data: reuses the SAME cached 5-min bars as backtest_futures.py (no new
fetch) -- resampled to 15-min here for both the box construction and the
execution chart.

This first pass runs ONE fixed default config end-to-end for a sanity
look at whether the idea has any legs; a parameter sweep (box row count,
value-area %, avg-range lookback, entry cutoff, etc.) is the deliberate
next step, which is why every tunable lives on the Config dataclass.

Run:
    python3 strategy_yellow_box.py
    python3 strategy_yellow_box.py --days 250
"""
import argparse
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_futures import load_cached, build_day_frames, SPECS, ET, COMMISSION_RT, stats

from strategy_params import PARAMS, parse_hhmm

_YB = PARAMS["yellow_box"]

RTH_START = parse_hhmm(PARAMS["session"]["rth_start"])
RTH_END = parse_hhmm(PARAMS["session"]["rth_end"])


@dataclass
class Config:
    n_rows: int = _YB["n_rows"]           # TPO histogram rows spanning the box day's range
    va_pct: float = _YB["va_pct"]         # value-area coverage (standard Market Profile convention)
    avg_range_lookback: int = _YB["avg_range_lookback"]  # trailing sessions averaged for the TP1 target
    risk_cap: float = _YB["risk_cap"]     # reject signals whose stop distance * point_value exceeds this
    eod_flat: time = parse_hhmm(_YB["eod_flat"])
    # Spec says the stop goes "above the retest swing high (for short) or below
    # the retest swing low (for long)" -- i.e. BEYOND the extreme, not on it.
    # With 0.0 the stop sits exactly on the retest bar's extreme, which is the
    # level price is actively oscillating around, so it gets picked off almost
    # immediately (55% of stop-outs landed on the entry bar itself). This pushes
    # the stop further into the box by a fraction of the box's own width; the
    # stop is still required to stay strictly inside [VAL, VAH].
    stop_buffer_frac: float = _YB["stop_buffer_frac"]
    # ── target structure ─────────────────────────────────────────────────
    # "range":      TP1 = POC +/- avg daily range. The literal spec. Median 7R
    #               away from a stop set by a 15-min wick, so only ~8% of trades
    #               ever touch it -- the target is not scaled to the risk.
    # "r_multiple": TP1 = entry +/- tgt1_r * risk. Reward now scales WITH the
    #               stop, which is the actual fix for the unreachable target.
    # "scale":      take scale_frac off at tgt1_r * risk, move the stop to
    #               breakeven, let the remainder run to the range level (the
    #               spec's "Average Range" target) as a runner. Mirrors
    #               backtest_futures.py's "scale" management.
    # NOTE: "scale" assumes a divisible position (same idealization as
    # backtest_futures.py). At 1 contract you cannot actually take half off --
    # trading it live needs >= 2 lots.
    target_mode: str = _YB["target_mode"]
    tgt1_r: float = _YB["tgt1_r"]
    scale_frac: float = _YB["scale_frac"]
    # ── position sizing ──────────────────────────────────────────────────
    # "fixed": always fixed_contracts lots -- the default, because it makes
    #   every trade equally weighted, so PF/expectancy measure the SETUP's
    #   edge rather than the sizing rule's. This is what all results in the
    #   sweep were produced under.
    # "risk": lots = risk_per_trade / (stop distance * point_value), capped
    #   at max_contracts -- equal DOLLAR risk per trade, the way it would
    #   actually be traded. Mirrors backtest_futures.simulate()'s sizing.
    # "account": size from CURRENT equity -- risk_pct of the running balance,
    #   also capped by how many lots day-trade margin allows. This is the
    #   realistic mode for a small account and it is path-dependent: a
    #   drawdown shrinks subsequent size. Trades are skipped outright when
    #   the account cannot fund even one lot.
    sizing: str = _YB["sizing"]
    fixed_contracts: int = _YB["fixed_contracts"]
    risk_per_trade: float = _YB["risk_per_trade"]
    max_contracts: int = _YB["max_contracts"]
    # ── account model (used when sizing="account") ────────────────────────
    account_start: float = _YB["account_start"]
    risk_pct: float = _YB["risk_pct"]   # fraction of CURRENT equity risked per trade
    day_margin: float = _YB["day_margin"]   # intraday margin per MES lot -- broker-dependent
                                   # (Tradovate/AMP-style day margin; overnight is
                                   # far higher, but this strategy is always flat
                                   # by 15:45 so day margin is the binding one)
    allow_min_one: bool = _YB["allow_min_one"]   # take 1 lot even when that exceeds risk_pct,
                                   # as long as margin allows -- what a small
                                   # account actually does. Set False to enforce
                                   # risk_pct strictly and skip unaffordable trades.


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA -- resample cached 5-min bars to 15-min (keep overnight bars too;
#    build_day_frames needs them to validate each RTH day has real
#    pre-open history, even though the box itself only uses RTH bars).
# ════════════════════════════════════════════════════════════════════════════

def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate the cached 5-min bars up to `minutes`. Must divide evenly into
    the 390-minute RTH session for day boundaries to line up cleanly (5, 10,
    15, 30, 65, 78... -- 60 does NOT divide 390, so the last bucket of each
    session is short; that's tolerable but worth knowing when reading results)."""
    rule = f"{minutes}min"
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.dropna()


def resample_15(df: pd.DataFrame) -> pd.DataFrame:
    """Back-compat alias -- the spec's original 15-min execution timeframe."""
    return resample_bars(df, 15)


# ════════════════════════════════════════════════════════════════════════════
# 2. YELLOW BOX -- TPO time-based value area from the PRIOR day's 15-min
#    RTH bars only. Standard Market Profile value-area algorithm: start at
#    the POC row, greedily add whichever adjacent row (above or below) has
#    the higher period-count, until >= va_pct of the day's periods are
#    covered. (This single-row greedy expansion is the common simplified
#    variant; the textbook CBOT method compares two-row groups at a time --
#    close enough for a backtest, worth knowing if this box looks off vs.
#    a charting platform's box.)
# ════════════════════════════════════════════════════════════════════════════

def compute_box(day_df: pd.DataFrame, tick: float, n_rows: int, va_pct: float) -> dict:
    lo, hi = float(day_df["low"].min()), float(day_df["high"].max())
    if hi <= lo:
        return None
    row_size = max(tick, (hi - lo) / n_rows)
    n_bins = max(1, int(round((hi - lo) / row_size)))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts = np.zeros(n_bins)
    for _, r in day_df.iterrows():
        blo, bhi = float(r["low"]), float(r["high"])
        lo_idx = max(0, min(n_bins - 1, np.searchsorted(edges, blo, side="right") - 1))
        hi_idx = max(0, min(n_bins - 1, np.searchsorted(edges, bhi, side="right") - 1))
        counts[lo_idx:hi_idx + 1] += 1

    total = counts.sum()
    if total <= 0:
        return None
    poc_idx = int(np.argmax(counts))
    lo_area = hi_area = poc_idx
    area_count = counts[poc_idx]
    while area_count < va_pct * total:
        above = hi_area + 1 if hi_area + 1 < n_bins else None
        below = lo_area - 1 if lo_area - 1 >= 0 else None
        c_above = counts[above] if above is not None else -1
        c_below = counts[below] if below is not None else -1
        if c_above <= 0 and c_below <= 0:
            break
        if c_above >= c_below:
            hi_area = above
            area_count += counts[above]
        else:
            lo_area = below
            area_count += counts[below]

    vah = float(edges[hi_area + 1])
    val = float(edges[lo_area])
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    return dict(vah=vah, val=val, poc=poc)


def build_daily_context(day15_full: pd.DataFrame, tick: float, cfg: Config):
    """Per session (sorted chronologically): the box built from the PRIOR
    session's RTH bars, and the average daily range over the
    cfg.avg_range_lookback sessions strictly before it. Both are None
    until enough history exists -- no lookahead, no box on day 1."""
    sessions = build_day_frames(day15_full)  # [(day, rth_df, on_high, on_low), ...]
    day_ranges = [float(day["high"].max() - day["low"].min()) for _, day, *_ in sessions]

    ctx = {}
    for k, (d, _day, *_ ) in enumerate(sessions):
        if k < 1 or k < cfg.avg_range_lookback:
            continue
        prev_day_df = sessions[k - 1][1]
        box = compute_box(prev_day_df, tick, cfg.n_rows, cfg.va_pct)
        if box is None:
            continue
        avg_range = float(np.mean(day_ranges[k - cfg.avg_range_lookback:k]))
        ctx[d] = dict(box=box, avg_range=avg_range)
    return sessions, ctx


# ════════════════════════════════════════════════════════════════════════════
# 3. SIGNALS -- per-day state machine: seek breakout close outside the box,
#    then seek a retest-and-rejection back through the broken level.
# ════════════════════════════════════════════════════════════════════════════

def generate_signals(sym: str, day15_full: pd.DataFrame, cfg: Config,
                      precomputed: "tuple | None" = None) -> pd.DataFrame:
    """precomputed: optional (sessions, ctx) pair from build_daily_context, so a
    parameter sweep can reuse cached box/avg-range context across configs
    instead of recomputing the TPO histogram for every grid point. Passing it
    changes nothing about the logic -- it must be the same (sessions, ctx) this
    function would have built for `cfg` itself."""
    tick = SPECS[sym]["tick"]
    sessions, ctx = precomputed if precomputed is not None else \
        build_daily_context(day15_full, tick, cfg)
    recs = []
    for d, day, *_ in sessions:
        c = ctx.get(d)
        if c is None:
            continue
        vah, val, poc = c["box"]["vah"], c["box"]["val"], c["box"]["poc"]
        avg_range = c["avg_range"]
        long_target = poc + avg_range
        short_target = poc - avg_range

        arr = day.reset_index()
        state = {1: "seek_breakout", -1: "seek_breakout"}
        for i in range(len(arr)):
            r = arr.iloc[i]
            close = float(r["close"])

            # long side
            if state[1] == "seek_breakout":
                if close > vah:
                    state[1] = "seek_retest"
            else:  # seek_retest
                if float(r["low"]) <= vah:
                    if close > vah:
                        # stop BELOW the retest swing low, deeper into the box
                        stop = float(r["low"]) - cfg.stop_buffer_frac * (vah - val)
                        # the "target must still be ahead of price" screen only
                        # applies when the range level IS the target; with an
                        # R-scaled target it would reject perfectly valid signals
                        tgt_ok = (long_target > close) if cfg.target_mode == "range" else True
                        if val < stop < vah and tgt_ok:
                            recs.append(dict(
                                sym=sym, day=d, ts=r["ts"], bar_idx=i, bias=1,
                                stop=stop, target=long_target,
                                vah=vah, val=val, poc=poc, avg_range=avg_range,
                            ))
                        state[1] = "seek_breakout"
                    else:
                        state[1] = "seek_breakout"  # failed retest, re-arm

            # short side
            if state[-1] == "seek_breakout":
                if close < val:
                    state[-1] = "seek_retest"
            else:  # seek_retest
                if float(r["high"]) >= val:
                    if close < val:
                        # stop ABOVE the retest swing high, deeper into the box
                        stop = float(r["high"]) + cfg.stop_buffer_frac * (vah - val)
                        tgt_ok = (short_target < close) if cfg.target_mode == "range" else True
                        if val < stop < vah and tgt_ok:
                            recs.append(dict(
                                sym=sym, day=d, ts=r["ts"], bar_idx=i, bias=-1,
                                stop=stop, target=short_target,
                                vah=vah, val=val, poc=poc, avg_range=avg_range,
                            ))
                        state[-1] = "seek_breakout"
                    else:
                        state[-1] = "seek_breakout"

    sig = pd.DataFrame(recs)
    return sig.sort_values("ts").reset_index(drop=True) if len(sig) else sig


# ════════════════════════════════════════════════════════════════════════════
# 4. SIMULATION -- next-bar-open fill, 1-tick adverse slippage, stop-first
#    on same-bar overlap, forced flat at cfg.eod_flat.
#
#    Target structure follows cfg.target_mode (range / r_multiple / scale).
#    When sizing="account", position size comes from the RUNNING equity, so
#    the whole simulation is path-dependent -- signals must be consumed in
#    strict chronological order, and the run halts on account ruin.
# ════════════════════════════════════════════════════════════════════════════

def simulate(signals: pd.DataFrame, bars15: dict, cfg: Config) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    day_bars = {}
    for sym, df in bars15.items():
        for d, day, *_ in build_day_frames(df):
            day_bars[(sym, d)] = day.reset_index()

    trades = []
    open_until = {}
    equity = cfg.account_start
    ruined = False
    for _, s in signals.sort_values("ts").iterrows():
        if ruined:
            break
        sym, d = s["sym"], s["day"]
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
        # The next-bar-open fill must land BEFORE the forced-flat bar. A signal
        # on the 15:30 candle would otherwise fill at 15:45 open and be flattened
        # by the eod_flat branch on that very same bar -- a guaranteed
        # -2 ticks - commission with zero chance of reaching stop or target.
        ebt = entry_bar["ts"].time() if hasattr(entry_bar["ts"], "time") else entry_bar["ts"]
        if ebt >= cfg.eod_flat:
            continue
        entry = float(entry_bar["open"]) + bias * tick
        stop = float(s["stop"])
        range_target = float(s["target"])
        risk_pts = (entry - stop) * bias
        if risk_pts <= tick:
            continue
        risk_dollars = risk_pts * pv
        if risk_dollars > cfg.risk_cap:
            continue

        # ── position sizing ──────────────────────────────────────────────
        if cfg.sizing == "risk":
            n_contracts = min(cfg.max_contracts,
                              max(1, int(cfg.risk_per_trade / risk_dollars)))
        elif cfg.sizing == "account":
            by_risk = int((equity * cfg.risk_pct) / risk_dollars)
            by_margin = int(equity / cfg.day_margin)
            if by_margin < 1:
                ruined = True          # cannot fund a single lot -- account is done
                break
            n_contracts = min(cfg.max_contracts, by_risk, by_margin)
            if n_contracts < 1:
                if cfg.allow_min_one and by_margin >= 1:
                    n_contracts = 1    # small account takes the trade anyway
                else:
                    continue           # strict risk_pct -- skip unaffordable trade
        else:
            n_contracts = cfg.fixed_contracts

        # ── target structure ─────────────────────────────────────────────
        scale_out = False
        if cfg.target_mode == "range":
            t1, t2 = range_target, None
        elif cfg.target_mode == "r_multiple":
            t1, t2 = entry + bias * cfg.tgt1_r * risk_pts, None
        elif cfg.target_mode == "scale":
            t1 = entry + bias * cfg.tgt1_r * risk_pts
            t2 = range_target
            # runner only makes sense if the range level is BEYOND T1, and
            # only if the position can actually be split
            if (t2 - entry) * bias <= (t1 - entry) * bias or n_contracts < 2:
                t2, scale_out = None, False
            else:
                scale_out = True
        else:
            raise ValueError(f"unknown target_mode {cfg.target_mode!r}")

        # ── walk the position forward ────────────────────────────────────
        # pnl_pts accumulates across partial exits. `remaining` is the fraction
        # of the position still open; with scale_out it drops to 1-scale_frac
        # at T1 and the stop moves to breakeven for the runner.
        pnl_pts = 0.0
        remaining = 1.0
        half_closed = False
        cur_stop = stop
        t1_price = None
        exit_price = exit_ts = exit_reason = None
        for j in range(i + 1, len(day)):
            b = day.iloc[j]
            bt = b["ts"].time() if hasattr(b["ts"], "time") else b["ts"]
            if bt >= cfg.eod_flat:
                px = float(b["open"]) - bias * tick
                pnl_pts += remaining * (px - entry) * bias
                exit_price, exit_ts, exit_reason = px, b["ts"], "eod"
                break
            lo, hi = float(b["low"]), float(b["high"])
            stop_hit = (lo <= cur_stop) if bias == 1 else (hi >= cur_stop)
            t1_hit = (hi >= t1) if bias == 1 else (lo <= t1)
            t2_hit = (t2 is not None) and ((hi >= t2) if bias == 1 else (lo <= t2))
            if stop_hit:                       # conservative same-bar priority
                px = cur_stop - bias * tick
                pnl_pts += remaining * (px - entry) * bias
                exit_price, exit_ts = px, b["ts"]
                exit_reason = "be" if half_closed and cur_stop == entry else "stop"
                break
            if not scale_out:
                if t1_hit:
                    px = t1 - bias * tick
                    pnl_pts += remaining * (px - entry) * bias
                    exit_price, exit_ts, exit_reason = px, b["ts"], "target"
                    break
            else:
                if not half_closed and t1_hit:
                    px = t1 - bias * tick
                    pnl_pts += cfg.scale_frac * (px - entry) * bias
                    remaining = 1.0 - cfg.scale_frac
                    half_closed = True
                    cur_stop = entry           # runner protected at breakeven
                    t1_price = px
                if half_closed and t2_hit:
                    px = t2 - bias * tick
                    pnl_pts += remaining * (px - entry) * bias
                    exit_price, exit_ts, exit_reason = px, b["ts"], "runner"
                    break
        if exit_price is None:
            b = day.iloc[-1]
            px = float(b["close"])
            pnl_pts += remaining * (px - entry) * bias
            exit_price, exit_ts, exit_reason = px, b["ts"], "eod"

        pnl = pnl_pts * pv * n_contracts - COMMISSION_RT * n_contracts
        equity += pnl
        trades.append(dict(
            sym=sym, day=d, bias=bias, entry_ts=entry_bar["ts"], exit_ts=exit_ts,
            reason=exit_reason, entry=round(entry, 4), exit_price=round(exit_price, 4),
            stop=round(stop, 4), t1=round(t1, 4),
            t2=round(t2, 4) if t2 is not None else None,
            t1_price=round(t1_price, 4) if t1_price is not None else None,
            range_target=round(range_target, 4),
            vah=round(float(s["vah"]), 4), val=round(float(s["val"]), 4),
            poc=round(float(s["poc"]), 4), avg_range=round(float(s["avg_range"]), 4),
            contracts=n_contracts, pnl_pts=round(pnl_pts, 4),
            risk_per_contract=round(risk_dollars, 2),
            risk_dollars=round(risk_dollars * n_contracts, 2),
            risk_pct_of_equity=round(100 * risk_dollars * n_contracts / (equity - pnl), 2),
            r_multiple=round(pnl_pts / risk_pts, 3),
            pnl=round(pnl, 2), equity=round(equity, 2),
        ))
        open_until[sym] = exit_ts
        if cfg.sizing == "account" and equity < cfg.day_margin:
            ruined = True                      # blown up -- no further trading
    out = pd.DataFrame(trades)
    if len(out):
        out.attrs["ruined"] = ruined
    return out


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="limit to last N sessions")
    args = ap.parse_args()

    raw = load_cached(5)
    if len(raw) < 3:
        sys.exit("No cached 5-min data. Expected files in ./data/")
    bars15 = {sym: resample_15(df) for sym, df in raw.items()}

    cfg = Config()
    all_sigs = pd.concat(
        [generate_signals(sym, df, cfg) for sym, df in bars15.items()], ignore_index=True
    )
    if all_sigs.empty:
        sys.exit("No Yellow Box signals generated -- check box/retest logic against the data.")

    all_days = sorted(all_sigs["day"].unique())
    if args.days:
        all_days = all_days[-args.days:]
        all_sigs = all_sigs[all_sigs["day"].isin(all_days)]
    split = all_days[int(len(all_days) * 0.7) - 1]

    print(f"Config: n_rows={cfg.n_rows} va_pct={cfg.va_pct} "
          f"avg_range_lookback={cfg.avg_range_lookback}")
    print(f"Sessions with signals: {len(all_days)}  |  train <= {split}  |  test > {split}")
    print(f"Candidate signals (breakout+retest+rejection): {len(all_sigs)}")
    print(all_sigs.groupby("sym").size().rename("candidates").to_string())

    trades = simulate(all_sigs, bars15, cfg)
    if trades.empty:
        sys.exit("All candidate signals were filtered out in simulation (risk cap / degenerate risk).")

    train_tr = trades[trades["day"] <= split]
    test_tr = trades[trades["day"] > split]

    print("\n── FULL HISTORY ──")
    st_full = stats(trades)
    print(f"n={st_full['n']}  WR%={st_full['wr']}  PF={st_full['pf']}  exp$={st_full['exp']}  "
          f"pnl$={st_full['pnl']}  maxDD$={st_full['maxdd']}")

    print("\n── TRAIN (<= split) vs TEST (> split) -- judge the strategy on TEST ──")
    st_tr, st_te = stats(train_tr), stats(test_tr)
    print(f"train: n={st_tr['n']}  WR%={st_tr['wr']}  PF={st_tr['pf']}  exp$={st_tr['exp']}  "
          f"pnl$={st_tr['pnl']}  maxDD$={st_tr['maxdd']}")
    print(f"test:  n={st_te['n']}  WR%={st_te['wr']}  PF={st_te['pf']}  exp$={st_te['exp']}  "
          f"pnl$={st_te['pnl']}  maxDD$={st_te['maxdd']}")

    print("\n── By symbol (full history) ──")
    by_sym = trades.groupby("sym").agg(
        n=("pnl", "size"), wr=("pnl", lambda x: round(100 * (x > 0).mean(), 1)),
        pf=("pnl", lambda x: round(x[x > 0].sum() / -x[x <= 0].sum(), 2) if (x <= 0).any() and -x[x <= 0].sum() > 0 else float("inf")),
        pnl=("pnl", "sum"),
    ).round(2)
    print(by_sym.to_string())

    print("\n── By bias (full history) ──")
    by_bias = trades.assign(side=trades["bias"].map({1: "long", -1: "short"})).groupby("side").agg(
        n=("pnl", "size"), wr=("pnl", lambda x: round(100 * (x > 0).mean(), 1)), pnl=("pnl", "sum"),
    ).round(2)
    print(by_bias.to_string())

    print("\n── Exit reasons ──")
    print(trades["reason"].value_counts().to_string())

    trades.to_csv("yellow_box_trades.csv", index=False)
    print("\nSaved: yellow_box_trades.csv")
    print("\nNOTE: this is ONE fixed default config, not yet optimized -- next step is a")
    print("parameter sweep (n_rows, va_pct, avg_range_lookback, etc.). Don't read too much")
    print("into a single-config result either way.")


if __name__ == "__main__":
    main()
