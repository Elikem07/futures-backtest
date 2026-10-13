"""
sweep_edge_only.py -- Same timeframe x target x stop grid as sweep_yellow_box_v2,
but at a FLAT 1 CONTRACT with no account constraint.

Why this exists separately: in the $1,000 account sweep, ruin halts the run.
That truncates the trade sample at the moment things go worst, so PF and t-stat
there are measured on a censored sample and cannot be read as "does the setup
have edge". This file answers the edge question cleanly -- every signal is taken,
every trade is weighted identically, nothing is censored -- and the account sweep
answers the separate question of whether $1,000 can survive it.

MES only, matching the trading decision already made.
"""
import itertools
import sys
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_futures import load_cached, build_day_frames
from strategy_yellow_box import Config, resample_bars, generate_signals, simulate
from sweep_yellow_box_v2 import TIMEFRAMES, TARGETS, STOPS

SYM = "MES"


def ev(t):
    if t is None or len(t) == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net=0.0, exp_r=np.nan,
                    exp_r_ex5=np.nan, tstat=np.nan, tgt_pct=np.nan)
    w, l = t[t.pnl > 0], t[t.pnl <= 0]
    gl = -l.pnl.sum()
    r = (t["pnl"] / t["risk_dollars"]).to_numpy()
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    daily = t.groupby("day")["pnl"].sum().sort_values(ascending=False)
    ex5 = t[~t["day"].isin(set(daily.index[:5]))]
    return dict(
        n=len(t), wr=round(100 * len(w) / len(t), 1),
        pf=round(w.pnl.sum() / gl, 2) if gl > 0 else float("inf"),
        net=round(t.pnl.sum(), 2), exp_r=round(r.mean(), 4),
        exp_r_ex5=round((ex5["pnl"] / ex5["risk_dollars"]).mean(), 4) if len(ex5) else np.nan,
        tstat=round(r.mean() / (sd / np.sqrt(len(r))), 2) if sd > 0 and len(r) > 1 else np.nan,
        tgt_pct=round(100 * t["reason"].isin(["target", "runner"]).mean(), 1),
    )


def main():
    raw = load_cached(5)
    bars = {tf: {SYM: resample_bars(raw[SYM], tf)} for tf in TIMEFRAMES}
    days = sorted({d for d, *_ in build_day_frames(bars[TIMEFRAMES[0]][SYM])})
    split = days[int(len(days) * 0.7) - 1]
    print(f"{SYM} only, flat 1 contract, no account constraint")
    print(f"Sessions {len(days)}   train <= {split}   test > {split}\n")

    cache, rows = {}, []
    for tf, (tm, tr), sb in itertools.product(TIMEFRAMES, TARGETS, STOPS):
        cfg = replace(Config(), sizing="fixed", fixed_contracts=1,
                       stop_buffer_frac=sb, target_mode=tm,
                       tgt1_r=(tr if tr is not None else 2.0))
        key = (tf, sb, tm == "range")
        if key not in cache:
            cache[key] = generate_signals(SYM, bars[tf][SYM], cfg)
        s = cache[key]
        if s.empty:
            continue
        a = ev(simulate(s[s["day"] <= split], bars[tf], cfg))
        b = ev(simulate(s[s["day"] > split], bars[tf], cfg))
        f = ev(simulate(s, bars[tf], cfg))
        lab = f"{tm}" + (f"@{tr}R" if tr is not None else "")
        rows.append(dict(tf=tf, target=lab, sb=sb,
                          tr_n=a["n"], tr_pf=a["pf"], tr_net=a["net"],
                          te_n=b["n"], te_wr=b["wr"], te_pf=b["pf"], te_net=b["net"],
                          te_expR=b["exp_r"], te_expR_ex5=b["exp_r_ex5"],
                          te_t=b["tstat"], te_tgt_pct=b["tgt_pct"],
                          full_pf=f["pf"], full_net=f["net"], full_n=f["n"]))
        print(f"  tf={tf:>2}m {lab:<16} sb={sb:.2f}  full PF={f['pf']:<5} "
              f"train PF={a['pf']:<5} test PF={b['pf']:<5} tgt-hit={b['tgt_pct']}%", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("yellow_box_edge_sweep.csv", index=False)
    L = "═" * 92
    n = len(res)

    print(f"{L}\nSURFACE SHAPE (1 lot, uncensored)\n{L}")
    print(f"  configs: {n}")
    print(f"  FULL-history PF > 1.0: {(res.full_pf>1).sum()}/{n}")
    print(f"  train PF > 1.0:        {(res.tr_pf>1).sum()}/{n}")
    print(f"  test  PF > 1.0:        {(res.te_pf>1).sum()}/{n}")
    print(f"  BOTH > 1.0:            {((res.tr_pf>1)&(res.te_pf>1)).sum()}/{n}")
    print(f"  median full PF {res.full_pf.median():.2f}  "
          f"median train PF {res.tr_pf.median():.2f}  median test PF {res.te_pf.median():.2f}")

    wf = res.sort_values(["tr_pf", "tr_net"], ascending=False).iloc[0]
    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * n))
    print(f"\n{L}\nWALK-FORWARD (chosen on train)\n{L}")
    print(f"  tf={wf.tf}m  target={wf.target}  stop_buffer={wf.sb}")
    print(f"    train n={wf.tr_n} PF={wf.tr_pf} net=${wf.tr_net:,.2f}")
    print(f"    TEST  n={wf.te_n} WR={wf.te_wr}% PF={wf.te_pf} net=${wf.te_net:,.2f} "
          f"expR={wf.te_expR} expR_ex5={wf.te_expR_ex5} t={wf.te_t}  "
          f"target-hit {wf.te_tgt_pct}%")
    print(f"  Bonferroni bar for {n} configs: t>={bar:.2f}   best test t anywhere: {res.te_t.max()}")
    print(f"  -> {'CLEARS' if wf.te_t >= bar else 'DOES NOT CLEAR'}")

    print(f"\n{L}\nDOES A RISK-SCALED TARGET FIX THE 8.4% HIT RATE?\n{L}")
    print(res.groupby("target").agg(
        target_hit_pct=("te_tgt_pct", "mean"), test_pf=("te_pf", "mean"),
        full_pf=("full_pf", "mean"), test_wr=("te_wr", "mean")).round(2).to_string())

    print(f"\n{L}\nEFFECT OF STOP BUFFER (averaged over tf and target)\n{L}")
    print(res.groupby("sb").agg(
        full_pf=("full_pf", "mean"), test_pf=("te_pf", "mean"),
        test_wr=("te_wr", "mean"), full_net=("full_net", "mean")).round(2).to_string())

    print(f"\n{L}\nEFFECT OF TIMEFRAME (averaged over target and stop)\n{L}")
    print(res.groupby("tf").agg(
        full_pf=("full_pf", "mean"), test_pf=("te_pf", "mean"),
        n_trades=("tr_n", "mean"), full_net=("full_net", "mean")).round(2).to_string())

    print(f"\n{L}\nTOP 15 BY FULL-HISTORY PF (most trades, least selection bias)\n{L}")
    print(res.sort_values("full_pf", ascending=False).head(15)[
        ["tf", "target", "sb", "tr_n", "te_n", "full_pf", "tr_pf", "te_pf",
         "te_t", "te_expR_ex5", "full_net"]].to_string(index=False))
    print("\nSaved: yellow_box_edge_sweep.csv")


if __name__ == "__main__":
    main()
