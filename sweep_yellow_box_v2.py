"""
sweep_yellow_box_v2.py -- Yellow Box sweep over CANDLE TIMEFRAME x TARGET x STOP,
traded as a $1,000 MES-only account.

Why these three axes: the v1 sweep (box geometry only) found no edge, and the
detailed report identified two structural causes rather than parameter causes --
(1) the spec's POC +/- avg-range target sits a median 7R away from a stop set by
a single bar's wick, so only 8.4% of trades ever reach it, and (2) the stop sits
essentially ON the broken box edge, so 55% of stop-outs happen on the entry bar.
This sweep attacks exactly those two, plus timeframe, which changes both the
profile resolution and the wick size that sets the stop.

Account model (this is the part that usually gets skipped):
  * $1,000 starting equity, MES only, ONE position at a time.
  * Size from CURRENT equity, so a drawdown shrinks the next bet. Path-dependent.
  * Day-trade margin $50/lot caps size independently of risk. Broker-dependent.
  * Ruin is detected and the run HALTS -- no trading a blown account back.
  * Train and test are each simulated from a FRESH $1,000, so the test result
    answers "if I started this account on day one of the test window", not
    "if I inherited whatever the train window left me".

Multiple comparisons: this grid is large. The Bonferroni bar is printed and
applied. The prior Vol Div Short audit died on exactly this correction; a
result that does not clear it is not a finding.

Run:
    python3 sweep_yellow_box_v2.py
    python3 sweep_yellow_box_v2.py --quick
"""
import argparse
import itertools
import sys
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_futures import load_cached, build_day_frames, SPECS
from strategy_yellow_box import (
    Config, resample_bars, compute_box, generate_signals, simulate,
)

SYM = "MES"
from strategy_params import PARAMS

ACCOUNT = PARAMS["sweep"]["account"]

TIMEFRAMES = [5, 10, 15, 30]      # all divide the 390-min RTH session evenly
TARGETS = [                        # (target_mode, tgt1_r)
    ("range", None),
    ("r_multiple", 1.0),
    ("r_multiple", 1.5),
    ("r_multiple", 2.0),
    ("r_multiple", 3.0),
    ("scale", 1.5),
    ("scale", 2.0),
]
STOPS = [0.0, 0.10, 0.20, 0.30]

QUICK_TF = [15, 30]
QUICK_TGT = [("range", None), ("r_multiple", 2.0), ("scale", 2.0)]
QUICK_STOP = [0.0, 0.20]


def account_stats(trades: pd.DataFrame, start: float) -> dict:
    """Account-level result, not just trade-level. For a small account the
    equity path matters more than PF: a PF>1 system that hits zero on the way
    is still a blown account."""
    if trades is None or len(trades) == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net=0.0, final=start, ret_pct=0.0,
                    maxdd=0.0, maxdd_pct=0.0, tstat=np.nan, exp_r=np.nan,
                    exp_r_ex5=np.nan, ruined=False, med_lots=0)
    w, l = trades[trades.pnl > 0], trades[trades.pnl <= 0]
    gl = -l.pnl.sum()
    pf = w.pnl.sum() / gl if gl > 0 else float("inf")
    eq = trades["equity"]
    peak = eq.cummax()
    dd = eq - peak
    r = (trades["pnl"] / trades["risk_dollars"]).to_numpy()
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    daily = trades.groupby("day")["pnl"].sum().sort_values(ascending=False)
    ex5 = trades[~trades["day"].isin(set(daily.index[:5]))]
    return dict(
        n=len(trades),
        wr=round(100 * len(w) / len(trades), 1),
        pf=round(pf, 2) if pf != float("inf") else float("inf"),
        net=round(trades.pnl.sum(), 2),
        final=round(eq.iloc[-1], 2),
        ret_pct=round(100 * (eq.iloc[-1] - start) / start, 1),
        maxdd=round(dd.min(), 2),
        maxdd_pct=round(100 * (dd / peak.clip(lower=1e-9)).min(), 1),
        tstat=round(r.mean() / (sd / np.sqrt(len(r))), 2) if sd > 0 and len(r) > 1 else np.nan,
        exp_r=round(r.mean(), 4),
        exp_r_ex5=round((ex5["pnl"] / ex5["risk_dollars"]).mean(), 4) if len(ex5) else np.nan,
        ruined=bool(trades.attrs.get("ruined", False)),
        med_lots=int(trades["contracts"].median()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    tfs = QUICK_TF if args.quick else TIMEFRAMES
    tgts = QUICK_TGT if args.quick else TARGETS
    stops = QUICK_STOP if args.quick else STOPS

    raw = load_cached(5)
    bars = {tf: {SYM: resample_bars(raw[SYM], tf)} for tf in tfs}

    # split from the session calendar, identical for every config
    sess_days = sorted({d for d, *_ in build_day_frames(bars[tfs[0]][SYM])})
    split = sess_days[int(len(sess_days) * 0.7) - 1]
    print(f"Symbol: {SYM} only        Account: ${ACCOUNT:,.0f}")
    print(f"Sessions: {len(sess_days)}   train <= {split}   test > {split}")
    n_cfg = len(tfs) * len(tgts) * len(stops)
    print(f"Grid: {len(tfs)} timeframes x {len(tgts)} targets x {len(stops)} stops "
          f"= {n_cfg} configs\n")

    sig_cache = {}
    rows = []
    ci = 0
    for tf, (tmode, tr), sb in itertools.product(tfs, tgts, stops):
        ci += 1
        cfg = replace(Config(), sizing="account", account_start=ACCOUNT,
                       stop_buffer_frac=sb, target_mode=tmode,
                       tgt1_r=(tr if tr is not None else 2.0))
        # signal generation depends on timeframe, stop buffer, and only on
        # whether the target is the range level (that screen) -- cache on those
        key = (tf, sb, tmode == "range")
        if key not in sig_cache:
            sig_cache[key] = generate_signals(SYM, bars[tf][SYM], cfg)
        sigs = sig_cache[key]
        if sigs.empty:
            continue
        tr_sig = sigs[sigs["day"] <= split]
        te_sig = sigs[sigs["day"] > split]
        st_tr = account_stats(simulate(tr_sig, bars[tf], cfg), ACCOUNT)
        st_te = account_stats(simulate(te_sig, bars[tf], cfg), ACCOUNT)
        label = f"{tmode}" + (f"@{tr}R" if tr is not None else "")
        rows.append(dict(tf=tf, target=label, stop_buf=sb,
                          train_n=st_tr["n"], train_pf=st_tr["pf"],
                          train_final=st_tr["final"], train_ruined=st_tr["ruined"],
                          test_n=st_te["n"], test_pf=st_te["pf"], test_wr=st_te["wr"],
                          test_final=st_te["final"], test_ret=st_te["ret_pct"],
                          test_dd=st_te["maxdd"], test_t=st_te["tstat"],
                          test_expR=st_te["exp_r"], test_expR_ex5=st_te["exp_r_ex5"],
                          test_ruined=st_te["ruined"], lots=st_te["med_lots"]))
        print(f"  [{ci}/{n_cfg}] tf={tf:>2}m {label:<14} sb={sb:.2f} | "
              f"train PF={st_tr['pf']:<5} ${st_tr['final']:<9,.0f}"
              f"{'RUIN' if st_tr['ruined'] else '':>5} | "
              f"test PF={st_te['pf']:<5} ${st_te['final']:<9,.0f}"
              f"{'RUIN' if st_te['ruined'] else ''}")

    res = pd.DataFrame(rows)
    res.to_csv("yellow_box_sweep_v2.csv", index=False)
    L = "═" * 78

    print(f"\n{L}\nSURFACE SHAPE\n{L}")
    n = len(res)
    print(f"  configs evaluated:              {n}")
    print(f"  train PF > 1.0:                 {(res.train_pf>1).sum()}/{n}")
    print(f"  test  PF > 1.0:                 {(res.test_pf>1).sum()}/{n}")
    print(f"  BOTH > 1.0:                     {((res.train_pf>1)&(res.test_pf>1)).sum()}/{n}")
    print(f"  account RUINED in train:        {res.train_ruined.sum()}/{n}")
    print(f"  account RUINED in test:         {res.test_ruined.sum()}/{n}")
    print(f"  median train PF {res.train_pf.median():.2f}   median test PF {res.test_pf.median():.2f}")

    print(f"\n{L}\nWALK-FORWARD RESULT (chosen on TRAIN, judged on TEST)\n{L}")
    # only consider configs that did not blow the account during training
    elig = res[~res.train_ruined]
    if elig.empty:
        elig = res
        print("  NOTE: every config ruined the account in training; selecting anyway.")
    wf = elig.sort_values(["train_pf", "train_final"], ascending=False).iloc[0]
    print(f"  selected: tf={wf.tf}m  target={wf.target}  stop_buffer={wf.stop_buf}")
    print(f"    train: n={wf.train_n} PF={wf.train_pf} final=${wf.train_final:,.2f}"
          f"{'  RUINED' if wf.train_ruined else ''}")
    print(f"    TEST : n={wf.test_n} PF={wf.test_pf} WR={wf.test_wr}% "
          f"final=${wf.test_final:,.2f} ({wf.test_ret:+.1f}%) maxDD=${wf.test_dd:,.2f}")
    print(f"           expR={wf.test_expR}  expR(ex-top5)={wf.test_expR_ex5}  "
          f"t={wf.test_t}  median lots={wf.lots}"
          f"{'  RUINED' if wf.test_ruined else ''}")

    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * n))
    print(f"\n{L}\nMULTIPLE-COMPARISONS CORRECTION\n{L}")
    print(f"  {n} configs, family-wise alpha=0.05 -> per-comparison p={0.05/n:.5f}, "
          f"t bar ~= {bar:.2f}")
    print(f"  best test t anywhere:            {res.test_t.max()}")
    print(f"  walk-forward config test t:      {wf.test_t}")
    ok = (not pd.isna(wf.test_t)) and wf.test_t >= bar
    print(f"  -> walk-forward result {'CLEARS' if ok else 'DOES NOT CLEAR'} the corrected bar")

    print(f"\n{L}\nTOP 12 BY TEST PF (overfit view -- NOT tradeable)\n{L}")
    cols = ["tf", "target", "stop_buf", "train_pf", "test_n", "test_wr", "test_pf",
            "test_final", "test_dd", "test_t", "test_expR_ex5", "test_ruined"]
    print(res.sort_values("test_pf", ascending=False).head(12)[cols].to_string(index=False))

    print(f"\n{L}\nBEST PER TIMEFRAME (by train PF)\n{L}")
    for tf in tfs:
        sub = res[res.tf == tf].sort_values("train_pf", ascending=False)
        if sub.empty:
            continue
        b = sub.iloc[0]
        print(f"  {tf:>2}m: {b.target:<14} sb={b.stop_buf:.2f}  "
              f"train PF={b.train_pf:<5}  test PF={b.test_pf:<5} "
              f"test final=${b.test_final:>9,.2f}  t={b.test_t}")

    print(f"\nSaved: yellow_box_sweep_v2.csv")


if __name__ == "__main__":
    main()
