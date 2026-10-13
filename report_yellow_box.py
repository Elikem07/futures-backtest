"""
report_yellow_box.py -- Illustrative walkthrough of the Yellow Box backtest.

Prints the whole pipeline with real numbers at every stage: how many
sessions survive each filter, what a box actually looks like on a given
day, several worked trades resolved bar-by-bar, position sizing, and the
outcome distributions. Intended to make the backtest auditable by reading
it, rather than trusting a single headline PF.

Run:
    python3 report_yellow_box.py
    python3 report_yellow_box.py --sizing risk
    python3 report_yellow_box.py --examples 6
"""
import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_futures import load_cached, build_day_frames, SPECS, COMMISSION_RT
from strategy_yellow_box import (
    Config, resample_15, compute_box, build_daily_context, generate_signals, simulate,
)

SYMS = ("MES", "MNQ")
LINE = "═" * 78
THIN = "─" * 78


def hdr(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


def walk_trade(sym, tr_row, day_df, cfg):
    """Replay one trade bar-by-bar exactly as simulate() resolved it."""
    spec = SPECS[sym]
    tick, pv = spec["tick"], spec["point_value"]
    bias = int(tr_row["bias"])
    side = "LONG" if bias == 1 else "SHORT"
    entry, stop, target = tr_row["entry"], tr_row["stop"], tr_row["target"]
    n = int(tr_row["contracts"])

    print(f"\n{THIN}")
    print(f"{sym}  {tr_row['day']}  {side}   ({tr_row['reason'].upper()} exit)")
    print(f"{THIN}")
    print(f"  Yellow Box (from PRIOR session): VAL {tr_row['val']:.2f}  "
          f"POC {tr_row['poc']:.2f}  VAH {tr_row['vah']:.2f}")
    print(f"  Avg daily range (prior {cfg.avg_range_lookback} sessions): {tr_row['avg_range']:.2f}")
    tgt_label = "POC + avg_range" if bias == 1 else "POC - avg_range"
    print(f"  Target ({tgt_label}) = {target:.2f}")
    slip = "+1" if bias == 1 else "-1"
    print(f"  Entry {entry:.2f} (next bar's open {slip} tick adverse slippage)   "
          f"Stop {stop:.2f}   Risk {abs(entry-stop):.2f} pts = "
          f"${abs(entry-stop)*pv:.2f}/contract x {n} = ${tr_row['risk_dollars']:.2f}")
    print(f"\n  {'bar':<6}{'open':>10}{'high':>10}{'low':>10}{'close':>10}   what happened")

    entry_ts = pd.Timestamp(tr_row["entry_ts"])
    arr = day_df.reset_index() if day_df.index.name else day_df
    i0 = arr.index[arr["ts"] == entry_ts][0]
    for j in range(i0, len(arr)):
        b = arr.iloc[j]
        t = b["ts"].strftime("%H:%M")
        note = ""
        if j == i0:
            note = f"ENTRY fill @ {entry:.2f} ({n} contract{'s' if n>1 else ''})"
        if b["ts"].time() >= cfg.eod_flat:
            note = f"FORCED FLAT @ {tr_row['exit_price']:.2f}"
            print(f"  {t:<6}{b['open']:>10.2f}{b['high']:>10.2f}{b['low']:>10.2f}{b['close']:>10.2f}   {note}")
            break
        lo, hi = float(b["low"]), float(b["high"])
        hit_stop = (lo <= stop) if bias == 1 else (hi >= stop)
        hit_tgt = (hi >= target) if bias == 1 else (lo <= target)
        print(f"  {t:<6}{b['open']:>10.2f}{b['high']:>10.2f}{b['low']:>10.2f}{b['close']:>10.2f}   {note}")
        if hit_stop:
            print(f"  {'':<6}{'':>40}   -> STOP touched ({stop:.2f}), exit {tr_row['exit_price']:.2f}")
            break
        if hit_tgt:
            print(f"  {'':<6}{'':>40}   -> TARGET touched ({target:.2f}), exit {tr_row['exit_price']:.2f}")
            break
    gross = tr_row["pnl_pts"] * pv * n
    print(f"\n  Result: {tr_row['pnl_pts']:+.2f} pts x ${pv}/pt x {n} = ${gross:+.2f} gross"
          f"   commission ${COMMISSION_RT*n:.2f}   NET ${tr_row['pnl']:+.2f}"
          f"   ({tr_row['r_multiple']:+.2f}R)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizing", choices=["fixed", "risk"], default="fixed")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    cfg = replace(Config(), sizing=args.sizing)
    raw = load_cached(5)
    bars15 = {s: resample_15(raw[s]) for s in SYMS}

    # ── Stage 1: data ───────────────────────────────────────────────────────
    hdr("1. DATA PIPELINE")
    print("  Source: cached IBKR 5-min bars in ./data/ (no refetch).")
    print("  MGC excluded: its cache is contaminated -- every even calendar month")
    print("  has median volume 0 (illiquid odd-month contracts assigned front-month")
    print("  roll windows). MES/MNQ are quarterly contracts and are clean.\n")
    for s in SYMS:
        r5, r15 = raw[s], bars15[s]
        sess = build_day_frames(r15)
        print(f"  {s}: {len(r5):,} 5-min bars -> {len(r15):,} 15-min bars "
              f"-> {len(sess)} RTH sessions   ({r5.index.min().date()} to {r5.index.max().date()})")

    # ── Stage 2: the box ────────────────────────────────────────────────────
    hdr("2. THE YELLOW BOX (built from the PRIOR session only)")
    print("  Method: TPO / Market Profile. The prior RTH session's price range is")
    print(f"  sliced into {cfg.n_rows} rows. Each 15-min bar adds +1 to every row its")
    print("  high-low range touches. POC = busiest row. The value area grows out")
    print(f"  from the POC, repeatedly taking the busier neighbouring row, until")
    print(f"  {cfg.va_pct:.0%} of all TPO counts are enclosed. VAH/VAL = its edges.\n")
    sym = "MES"
    sess, ctx = build_daily_context(bars15[sym], SPECS[sym]["tick"], cfg)
    shown = 0
    for k, (d, day, *_) in enumerate(sess):
        if d not in ctx or shown >= 3:
            continue
        c = ctx[d]
        prev_d, prev_day = sess[k - 1][0], sess[k - 1][1]
        b = c["box"]
        print(f"  Session {d}  (box built from {prev_d})")
        print(f"    prior day range: {prev_day['low'].min():.2f} - {prev_day['high'].max():.2f}"
              f"   ({prev_day['high'].max()-prev_day['low'].min():.2f} pts, {len(prev_day)} bars)")
        print(f"    -> VAL {b['val']:.2f}   POC {b['poc']:.2f}   VAH {b['vah']:.2f}"
              f"   (box width {b['vah']-b['val']:.2f} pts = "
              f"{100*(b['vah']-b['val'])/(prev_day['high'].max()-prev_day['low'].min()):.0f}% of prior range)")
        print(f"    avg daily range (prior {cfg.avg_range_lookback} sessions): {c['avg_range']:.2f}"
              f"  -> long target {b['poc']+c['avg_range']:.2f}, short target {b['poc']-c['avg_range']:.2f}")
        shown += 1

    # ── Stage 3: signals ────────────────────────────────────────────────────
    hdr("3. SIGNAL FUNNEL")
    print("  Per session, two independent state machines (long side, short side):")
    print("    a) BREAKOUT  : a 15-min bar CLOSES outside the box (> VAH long, < VAL short)")
    print("    b) RETEST    : a later bar trades back and TOUCHES the broken edge")
    print("    c) REJECTION : that same bar CLOSES back outside again -> signal fires")
    print("  Then the state machine re-arms, so several retests per day are possible.\n")
    sigs = {s: generate_signals(s, bars15[s], cfg) for s in SYMS}
    allsig = pd.concat([sigs[s] for s in SYMS], ignore_index=True)
    for s in SYMS:
        n_days = len({d for d, *_ in build_day_frames(bars15[s])})
        _, ctx_s = build_daily_context(bars15[s], SPECS[s]["tick"], cfg)
        n_sig_days = sigs[s]["day"].nunique()
        print(f"  {s}: {n_days} sessions -> {len(ctx_s)} with a usable box "
              f"-> {n_sig_days} sessions that produced a signal "
              f"-> {len(sigs[s])} signals "
              f"({(sigs[s]['bias']==1).sum()} long / {(sigs[s]['bias']==-1).sum()} short)")
    print(f"\n  Signals rejected at simulation time and why:")
    trades = simulate(allsig, bars15, cfg)
    print(f"    total signals                     {len(allsig)}")
    print(f"    -> became trades                  {len(trades)}")
    print(f"    -> dropped ({len(allsig)-len(trades)}): position already open in that symbol,")
    print(f"       entry bar would be at/after {cfg.eod_flat} flat time, stop closer than")
    print(f"       1 tick, or per-contract risk > ${cfg.risk_cap:.0f} cap")

    # ── Stage 4: execution rules ────────────────────────────────────────────
    hdr("4. EXECUTION RULES (identical to backtest_futures.py conventions)")
    print("  * Signal is decided on BAR CLOSE; the fill is the NEXT bar's OPEN.")
    print("    No bar is ever used before it has closed -- no lookahead.")
    print("  * 1 tick of adverse slippage on entry AND on exit.")
    print(f"  * Commission ${COMMISSION_RT:.2f} round-turn per contract.")
    print("  * If a single bar's range spans BOTH stop and target, the STOP is")
    print("    assumed to fill first (conservative -- never credits the good fill).")
    print("  * One open position per symbol at a time; a new signal while in a")
    print("    trade is skipped, not stacked.")
    print(f"  * Everything is force-flattened at {cfg.eod_flat} ET. Nothing held overnight.")
    print(f"\n  Position sizing: mode='{cfg.sizing}'")
    if cfg.sizing == "fixed":
        print(f"    Every trade is exactly {cfg.fixed_contracts} contract. Chosen so that PF and")
        print(f"    expectancy measure the SETUP, not a sizing rule layered on top.")
    else:
        print(f"    contracts = min({cfg.max_contracts}, max(1, ${cfg.risk_per_trade:.0f} / per-contract risk))")
        print(f"    i.e. equal dollar risk per trade, capped at {cfg.max_contracts} lots.")
    print(f"\n    contracts actually used: {dict(sorted(trades['contracts'].value_counts().items()))}")
    print(f"    per-contract risk: median ${trades['risk_per_contract'].median():.2f}, "
          f"max ${trades['risk_per_contract'].max():.2f}")
    print(f"    total risk per trade: median ${trades['risk_dollars'].median():.2f}, "
          f"max ${trades['risk_dollars'].max():.2f}")

    # ── Stage 5: worked examples ────────────────────────────────────────────
    hdr("5. WORKED TRADES (replayed bar-by-bar)")
    day_bars = {}
    for s in SYMS:
        for d, day, *_ in build_day_frames(bars15[s]):
            day_bars[(s, d)] = day.reset_index()
    picks = []
    for reason in ("target", "stop", "eod"):
        sub = trades[trades["reason"] == reason]
        if len(sub):
            picks.append(sub.sort_values("pnl", ascending=(reason == "stop")).iloc[0])
    for t in trades[trades["reason"] == "target"].head(1).itertuples():
        pass
    for row in picks[: args.examples]:
        walk_trade(row["sym"], row, day_bars[(row["sym"], row["day"])], cfg)

    # ── Stage 6: results ────────────────────────────────────────────────────
    hdr("6. OUTCOME DISTRIBUTIONS")
    trades["hold_bars"] = [
        (pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 900
        for a, b in zip(trades["entry_ts"], trades["exit_ts"])
    ]
    print(f"  Exit reason:")
    for r, c in trades["reason"].value_counts().items():
        sub = trades[trades["reason"] == r]
        print(f"    {r:<8} {c:>5} trades ({100*c/len(trades):>4.1f}%)   "
              f"avg ${sub.pnl.mean():>8.2f}   total ${sub.pnl.sum():>10.2f}")
    print(f"\n  Holding time (15-min bars): median {trades['hold_bars'].median():.0f}, "
          f"mean {trades['hold_bars'].mean():.1f}, max {trades['hold_bars'].max():.0f}")
    print(f"  R-multiple: median {trades['r_multiple'].median():+.2f}, "
          f"mean {trades['r_multiple'].mean():+.3f}, "
          f"best {trades['r_multiple'].max():+.1f}, worst {trades['r_multiple'].min():+.1f}")
    rr = (trades["target"] - trades["entry"]).abs() / (trades["entry"] - trades["stop"]).abs()
    print(f"  Nominal reward:risk at entry: median {rr.median():.1f}R, 90th pct {rr.quantile(.9):.1f}R")
    print(f"    -> only {100*(trades['reason']=='target').mean():.1f}% of trades ever reach that target.")

    hdr("7. HEADLINE RESULT")
    for label, sub in (("ALL", trades),
                        ("MES", trades[trades.sym == "MES"]),
                        ("MNQ", trades[trades.sym == "MNQ"]),
                        ("LONG", trades[trades.bias == 1]),
                        ("SHORT", trades[trades.bias == -1])):
        w = sub[sub.pnl > 0]; l = sub[sub.pnl <= 0]
        pf = w.pnl.sum() / -l.pnl.sum() if len(l) and l.pnl.sum() < 0 else float("nan")
        print(f"  {label:<6} n={len(sub):>4}  win%={100*len(w)/len(sub):>5.1f}  "
              f"PF={pf:>5.2f}  expectancy=${sub.pnl.mean():>7.2f}  net=${sub.pnl.sum():>10.2f}")
    eq = trades.sort_values("exit_ts").pnl.cumsum()
    print(f"\n  Max drawdown: ${(eq - eq.cummax()).min():,.2f}")
    trades["year"] = pd.to_datetime(trades["day"]).dt.year
    print(f"\n  By year:")
    for y, sub in trades.groupby("year"):
        w = sub[sub.pnl > 0]; l = sub[sub.pnl <= 0]
        pf = w.pnl.sum() / -l.pnl.sum() if len(l) and l.pnl.sum() < 0 else float("nan")
        print(f"    {y}  n={len(sub):>4}  win%={100*len(w)/len(sub):>5.1f}  "
              f"PF={pf:>5.2f}  net=${sub.pnl.sum():>10.2f}")

    out = f"yellow_box_trades_{cfg.sizing}.csv"
    trades.to_csv(out, index=False)
    print(f"\n  Saved every trade with full detail -> {out}")


if __name__ == "__main__":
    main()
