"""
sweep_yellow_box.py -- Walk-forward parameter sweep for the Yellow Box
breakout/retest strategy (strategy_yellow_box.py).

MES + MNQ ONLY. MGC is deliberately excluded: its cached 5-min file is
contaminated -- every even calendar month has median volume 0 and 54-71%
zero-volume bars (odd months are normal). Root cause is in the roll logic:
gold's actively-traded contracts are the even months, but IBKR lists MGC
monthly, and backtest_futures._list_contract_months keeps every listed
month and assigns each one a front-month window -- so the illiquid
odd-month contracts fill in roughly the preceding even calendar month with
dead data. MES/MNQ are quarterly, so every listed contract is liquid and
the bug never surfaces there. Sweeping over synthetic flat-line sessions
would manufacture parameter choices that fit nothing.

Methodology (this file exists to resist its own results):
  * Parameters are chosen on the TRAIN window only; the reported number is
    that config's performance on the held-out TEST window. Picking the
    best TEST config would be cherry-picking, so this file computes it
    too -- and prints it clearly labeled as the overfit reference point,
    for contrast with the honest walk-forward number.
  * Every config is reported, not just the winner, so the shape of the
    whole surface is visible. A single good cell in an otherwise-negative
    grid is noise; a broad plateau is a (weak) signal. This distinction
    is what AUDIT_VOL_DIV_SHORT_MES.md concluded actually matters.
  * A Bonferroni-corrected t-stat bar for the grid size is printed
    alongside the result. The prior audit's headline died on exactly this
    correction -- ~60 configs needed t>=3.3 and the best was t=2.54.

Run:
    python3 sweep_yellow_box.py
    python3 sweep_yellow_box.py --quick     # small grid, faster
"""
import argparse
import itertools
import sys
from dataclasses import replace

from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_futures import load_cached, build_day_frames, SPECS, COMMISSION_RT
from strategy_yellow_box import (
    Config, resample_15, compute_box, generate_signals, simulate,
)

SYMBOLS = ("MES", "MNQ")

GRID = dict(
    n_rows=[20, 30, 40, 50],
    va_pct=[0.60, 0.70, 0.80],
    avg_range_lookback=[5, 10, 14, 20],
)
QUICK_GRID = dict(
    n_rows=[20, 40],
    va_pct=[0.70, 0.80],
    avg_range_lookback=[10, 20],
)


# ════════════════════════════════════════════════════════════════════════════
# 1. CACHED CONTEXT -- boxes depend only on (n_rows, va_pct); avg_range only
#    on avg_range_lookback. Cache both so the grid doesn't recompute the TPO
#    histogram for every combination.
# ════════════════════════════════════════════════════════════════════════════

class ContextCache:
    def __init__(self, bars15: dict):
        self.sessions = {s: build_day_frames(df) for s, df in bars15.items()}
        self.day_ranges = {
            s: [float(day["high"].max() - day["low"].min()) for _, day, *_ in sess]
            for s, sess in self.sessions.items()
        }
        self._boxes = {}

    def boxes(self, sym: str, n_rows: int, va_pct: float) -> dict:
        """{session_index k -> box built from sessions[k-1]}"""
        key = (sym, n_rows, va_pct)
        if key not in self._boxes:
            tick = SPECS[sym]["tick"]
            sess = self.sessions[sym]
            self._boxes[key] = {
                k: compute_box(sess[k - 1][1], tick, n_rows, va_pct)
                for k in range(1, len(sess))
            }
        return self._boxes[key]

    def context(self, sym: str, cfg: Config):
        """Rebuild exactly what build_daily_context would return for cfg,
        but using cached boxes. Same guards: no box before session 1, no
        avg_range before avg_range_lookback sessions exist, and the
        avg_range window is strictly prior sessions."""
        sess = self.sessions[sym]
        ranges = self.day_ranges[sym]
        boxes = self.boxes(sym, cfg.n_rows, cfg.va_pct)
        ctx = {}
        for k, (d, *_rest) in enumerate(sess):
            if k < 1 or k < cfg.avg_range_lookback:
                continue
            box = boxes.get(k)
            if box is None:
                continue
            ctx[d] = dict(box=box,
                          avg_range=float(np.mean(ranges[k - cfg.avg_range_lookback:k])))
        return sess, ctx


# ════════════════════════════════════════════════════════════════════════════
# 2. STATS -- R-multiple based, with t-stat and an ex-top5 sensitivity check
#    (does the result survive removing the 5 best days?).
# ════════════════════════════════════════════════════════════════════════════

def evaluate(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net=0.0, exp_r=np.nan,
                    exp_r_ex5=np.nan, tstat=np.nan, maxdd=0.0)
    r = (trades["pnl"] / trades["risk_dollars"]).to_numpy()
    wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
    gl = -losses.pnl.sum()
    pf = wins.pnl.sum() / gl if gl > 0 else float("inf")

    daily = trades.groupby("day")["pnl"].sum().sort_values(ascending=False)
    ex5 = trades[~trades["day"].isin(set(daily.index[:5]))]
    exp_r_ex5 = (ex5["pnl"] / ex5["risk_dollars"]).mean() if len(ex5) else np.nan

    sd = r.std(ddof=1) if n > 1 else 0.0
    tstat = r.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    eq = trades.sort_values("exit_ts").pnl.cumsum()
    return dict(
        n=n, wr=round(100 * len(wins) / n, 1),
        pf=round(pf, 2) if pf != float("inf") else float("inf"),
        net=round(trades.pnl.sum(), 2), exp_r=round(r.mean(), 4),
        exp_r_ex5=round(exp_r_ex5, 4) if not pd.isna(exp_r_ex5) else np.nan,
        tstat=round(tstat, 2) if not pd.isna(tstat) else np.nan,
        maxdd=round((eq - eq.cummax()).min(), 2),
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. SWEEP
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    ap.add_argument("--days", type=int, default=None, help="limit to last N sessions")
    args = ap.parse_args()

    raw = load_cached(5)
    missing = [s for s in SYMBOLS if s not in raw]
    if missing:
        sys.exit(f"Missing cached data for {missing}")
    bars15 = {s: resample_15(raw[s]) for s in SYMBOLS}
    cache = ContextCache(bars15)

    grid = QUICK_GRID if args.quick else GRID
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"Symbols: {list(SYMBOLS)}   (MGC excluded -- contaminated cache)")
    print(f"Grid: {len(combos)} configs over { 'x'.join(f'{k}({len(v)})' for k, v in grid.items()) }")

    # Establish the train/test split ONCE, from the session calendar itself,
    # so every config is judged on the identical held-out window. Deriving it
    # per-config from that config's own signal days would move the boundary
    # around and make configs incomparable.
    all_days = sorted({d for s in SYMBOLS for d, *_ in cache.sessions[s]})
    if args.days:
        all_days = all_days[-args.days:]
    split = all_days[int(len(all_days) * 0.7) - 1]
    print(f"Sessions: {len(all_days)}  |  train <= {split}  |  test > {split}\n")

    rows = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        cfg = replace(Config(), **params)
        sigs = pd.concat(
            [generate_signals(s, bars15[s], cfg, precomputed=cache.context(s, cfg))
             for s in SYMBOLS],
            ignore_index=True,
        )
        if sigs.empty:
            continue
        sigs = sigs[sigs["day"].isin(all_days)]
        trades = simulate(sigs, bars15, cfg)
        if trades.empty:
            continue
        tr_train = trades[trades["day"] <= split]
        tr_test = trades[trades["day"] > split]
        st_train, st_test = evaluate(tr_train), evaluate(tr_test)
        rows.append({**params,
                     "train_n": st_train["n"], "train_pf": st_train["pf"],
                     "train_expR": st_train["exp_r"],
                     "test_n": st_test["n"], "test_pf": st_test["pf"],
                     "test_expR": st_test["exp_r"], "test_expR_ex5": st_test["exp_r_ex5"],
                     "test_t": st_test["tstat"], "test_net": st_test["net"]})
        print(f"  [{ci}/{len(combos)}] {params}  train PF={st_train['pf']} n={st_train['n']}"
              f"   test PF={st_test['pf']} n={st_test['n']}")

    if not rows:
        sys.exit("No config produced trades.")
    res = pd.DataFrame(rows)
    res.to_csv("yellow_box_sweep.csv", index=False)

    # ── Surface shape: the single most informative diagnostic here ──────────
    n_cfg = len(res)
    train_pos = (res["train_pf"] > 1.0).sum()
    test_pos = (res["test_pf"] > 1.0).sum()
    both_pos = ((res["train_pf"] > 1.0) & (res["test_pf"] > 1.0)).sum()
    print("\n" + "═" * 78)
    print("SURFACE SHAPE  (is there a plateau, or one lucky cell?)")
    print("═" * 78)
    print(f"  configs evaluated:            {n_cfg}")
    print(f"  train PF > 1.0:               {train_pos}/{n_cfg} ({100*train_pos/n_cfg:.0f}%)")
    print(f"  test  PF > 1.0:               {test_pos}/{n_cfg} ({100*test_pos/n_cfg:.0f}%)")
    print(f"  BOTH train and test PF > 1.0: {both_pos}/{n_cfg} ({100*both_pos/n_cfg:.0f}%)")
    print(f"  median train PF: {res['train_pf'].median():.2f}   "
          f"median test PF: {res['test_pf'].median():.2f}")

    # ── The honest walk-forward result ──────────────────────────────────────
    wf = res.sort_values(["train_pf", "train_expR"], ascending=False).iloc[0]
    print("\n" + "═" * 78)
    print("WALK-FORWARD RESULT  (config chosen on TRAIN, judged on TEST)")
    print("═" * 78)
    sel = {k: wf[k] for k in keys}
    print(f"  selected on train: {sel}")
    print(f"    train: n={int(wf['train_n'])}  PF={wf['train_pf']}  expR={wf['train_expR']}")
    print(f"    TEST:  n={int(wf['test_n'])}  PF={wf['test_pf']}  expR={wf['test_expR']}  "
          f"expR(ex-top5)={wf['test_expR_ex5']}  t={wf['test_t']}  net=${wf['test_net']}")

    # ── Multiple-comparisons correction ─────────────────────────────────────
    # normal approximation to the t bar -- fine at these sample sizes, and
    # avoids a scipy dependency (not installed in this env)
    alpha_bar = NormalDist().inv_cdf(1 - 0.05 / (2 * n_cfg))
    print("\n" + "═" * 78)
    print("MULTIPLE-COMPARISONS CORRECTION")
    print("═" * 78)
    print(f"  {n_cfg} configs tested at family-wise alpha=0.05 (Bonferroni)")
    print(f"  -> per-comparison threshold p={0.05/n_cfg:.5f}, two-tailed t bar ~= {alpha_bar:.2f}")
    best_t = res["test_t"].max()
    print(f"  best test t-stat anywhere in the grid: {best_t}")
    print(f"  walk-forward selected config test t:   {wf['test_t']}")
    verdict = "CLEARS" if (not pd.isna(wf["test_t"]) and wf["test_t"] >= alpha_bar) else "DOES NOT CLEAR"
    print(f"  -> walk-forward result {verdict} the corrected bar")

    # ── Overfit reference point, clearly labeled ────────────────────────────
    bt = res.sort_values(["test_pf", "test_expR"], ascending=False).iloc[0]
    print("\n" + "═" * 78)
    print("OVERFIT REFERENCE  (best TEST config -- NOT a tradeable result)")
    print("═" * 78)
    print(f"  {{{', '.join(f'{k}: {bt[k]}' for k in keys)}}}  "
          f"test PF={bt['test_pf']} n={int(bt['test_n'])} t={bt['test_t']}")
    print("  Shown only to gauge how much better cherry-picking looks than the")
    print("  walk-forward number above. The gap is the overfitting premium.")

    print("\n" + "═" * 78)
    print("FULL GRID (sorted by train PF -- the selection criterion)")
    print("═" * 78)
    show = res.sort_values("train_pf", ascending=False)
    print(show.to_string(index=False))
    print("\nSaved: yellow_box_sweep.csv")


if __name__ == "__main__":
    main()
