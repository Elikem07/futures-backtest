"""
cost_analysis_futures.py — Commission & slippage stress-test on futures backtest trades.

Reads backtest_trades.csv (produced by backtest_futures.py) and re-applies
commission/slippage scenarios on top of raw_pnl — the zero-commission,
zero-slippage pnl backtest_futures.py stores per trade specifically so this
kind of re-analysis doesn't require re-running the simulation. Mirrors
cost_analysis.py's role for the options backtest; scenario B reproduces the
backtest's own baked-in cost model exactly, as a sanity check on raw_pnl.

Run:  python3 cost_analysis_futures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_futures import SPECS, COMMISSION_RT

# ── Config ────────────────────────────────────────────────────────────────────
N_SIMS       = 5_000
STARTING_CAP = 5_000
TRADES_CSV   = Path(__file__).parent / "backtest_trades.csv"

# commission = $ round-turn per contract. ticks = slippage ticks applied on
# EACH leg (entry + exit), same convention as cost_analysis.py's TICK_SIZE*ticks*2.
SCENARIOS = {
    "A  Zero costs (raw edge)":                {"commission": 0.00,         "ticks": 0},
    "B  Current backtest model ($1.40 RT+1t)": {"commission": COMMISSION_RT, "ticks": 1},
    "C  Retail-realistic ($2.50 RT+1t)":       {"commission": 2.50,         "ticks": 1},
    "D  Stressed ($4.00 RT+2t)":                {"commission": 4.00,         "ticks": 2},
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def _max_dd_vec(paths: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - paths) / peak, 0.0)
    return dd.max(axis=1)


def cost_per_trade(row: pd.Series, commission: float, ticks: int) -> float:
    """Round-trip cost in dollars: commission + tick slippage on entry & exit legs."""
    spec = SPECS[row["sym"]]
    tick_value = spec["tick"] * spec["point_value"]
    return commission * row["contracts"] + ticks * tick_value * 2 * row["contracts"]


def run_scenario(df: pd.DataFrame, label: str, commission: float, ticks: int) -> dict:
    costs = df.apply(cost_per_trade, axis=1, commission=commission, ticks=ticks)
    pnls  = (df["raw_pnl"] - costs).to_numpy(dtype=float)

    n    = len(pnls)
    rng  = np.random.default_rng(42)
    idx  = rng.integers(0, n, size=(N_SIMS, n))
    samp = pnls[idx]

    paths        = np.empty((N_SIMS, n + 1))
    paths[:, 0]  = STARTING_CAP
    paths[:, 1:] = STARTING_CAP + np.cumsum(samp, axis=1)

    final_eq = paths[:, -1]
    max_dds  = _max_dd_vec(paths) * 100

    actual = float(STARTING_CAP + pnls.sum())

    return {
        "label":       label,
        "cost_avg":    float(costs.mean()),
        "n_trades":    n,
        "pnls":        pnls,
        "win_rate":    float((pnls > 0).mean() * 100),
        "prob_profit": float((final_eq > STARTING_CAP).mean() * 100),
        "ev_trade":    float(pnls.mean()),
        "median_pnl":  _pct(final_eq, 50) - STARTING_CAP,
        "p95_pnl":     _pct(final_eq, 95) - STARTING_CAP,
        "p5_pnl":      _pct(final_eq, 5)  - STARTING_CAP,
        "p50_dd":      _pct(max_dds, 50),
        "p95_dd":      _pct(max_dds, 95),
        "actual":      actual - STARTING_CAP,
    }


# ── Display ───────────────────────────────────────────────────────────────────
def print_table(results: list[dict], recorded_pnl: float) -> None:
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    GREEN, RED = "\033[92m", "\033[91m"
    sep = "─" * 78

    print(f"\n{BOLD}{'Futures Commission & Slippage Impact':^78}{RESET}")
    print(f"{DIM}{N_SIMS:,} bootstrap sims  ·  ${STARTING_CAP:,} start  ·  "
          f"{results[0]['n_trades']} trades  ·  breakout setup, walk-forward test window{RESET}")
    print(sep)

    labels = [r["label"] for r in results]
    col    = 22
    print(f"  {'Metric':<26}" + "".join(f"  {l[:col]:<{col}}" for l in labels))
    print(sep)

    def row(name, getter, fmt=None):
        cells = []
        for r in results:
            v = getter(r)
            if fmt == "pct":
                color = GREEN if v >= 40 else RED
                cells.append(f"  {color}{v:>+.1f}%{RESET:<{col}}")
            elif fmt == "pprob":
                color = GREEN if v >= 80 else (DIM if v >= 50 else RED)
                cells.append(f"  {color}{v:>5.1f}%{RESET:<{col}}")
            elif fmt == "dd":
                color = GREEN if v <= 20 else (DIM if v <= 40 else RED)
                cells.append(f"  {color}{v:>5.1f}%{RESET:<{col}}")
            else:
                color = GREEN if v >= 0 else RED
                cells.append(f"  {color}${v:>+,.2f}{RESET:<{col}}")
        print(f"  {name:<26}" + "".join(cells))

    row("Avg cost/trade",         lambda r: r["cost_avg"])
    row("Win rate",               lambda r: r["win_rate"],    fmt="pct")
    print(f"  {DIM}{'─'*74}{RESET}")
    row("Prob of profit",         lambda r: r["prob_profit"], fmt="pprob")
    row("Median outcome (P50)",   lambda r: r["median_pnl"])
    row("Best case (P95)",        lambda r: r["p95_pnl"])
    row("Worst case (P5)",        lambda r: r["p5_pnl"])
    row("Expected value / trade", lambda r: r["ev_trade"])
    print(f"  {DIM}{'─'*74}{RESET}")
    row("Backtest P&L (this scenario)", lambda r: r["actual"])
    print(f"  {DIM}{'─'*74}{RESET}")
    row("Median max drawdown",    lambda r: r["p50_dd"],      fmt="dd")
    row("Worst-case DD (P95)",    lambda r: r["p95_dd"],      fmt="dd")
    print(sep)

    # Sanity check: scenario B should reproduce the CSV's own recorded pnl column exactly,
    # since it re-applies the exact cost model backtest_futures.py already baked in.
    b_actual = results[1]["actual"]
    gap = b_actual - recorded_pnl
    status = f"{GREEN}OK{RESET}" if abs(gap) < 0.01 else f"{RED}MISMATCH{RESET}"
    print(f"\n  {BOLD}Sanity check{RESET}  —  scenario B vs backtest's recorded pnl column")
    print(f"    Scenario B total: ${b_actual:+,.2f}   Recorded pnl total: ${recorded_pnl:+,.2f}   "
          f"gap: ${gap:+,.2f}  [{status}]")

    print(f"\n  {BOLD}Break-even cost analysis{RESET}")
    baseline_ev = results[0]["ev_trade"]
    print(f"  {DIM}Raw (zero-cost) EV/trade = ${baseline_ev:+.2f}  →  strategy breaks even "
          f"at ${baseline_ev:.2f}/trade in total costs{RESET}")
    for r in results[1:]:
        viable = r["ev_trade"] > 0
        vstatus = f"{GREEN}VIABLE{RESET}" if viable else f"{RED}UNVIABLE{RESET}"
        print(f"    {r['label']:<40}  EV/trade = ${r['ev_trade']:+.2f}  →  {vstatus}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not TRADES_CSV.exists():
        print("backtest_trades.csv not found — run backtest_futures.py first.")
        raise SystemExit(1)

    df = pd.read_csv(TRADES_CSV)
    if "raw_pnl" not in df.columns:
        print("backtest_trades.csv has no raw_pnl column — re-run backtest_futures.py "
              "(current version) to regenerate it before running cost analysis.")
        raise SystemExit(1)

    print(f"\nLoaded {len(df)} trades from backtest_trades.csv  "
          f"({', '.join(sorted(df['setup'].unique()))})")
    print(f"Symbols: {', '.join(sorted(df['sym'].unique()))}  |  "
          f"Raw win rate: {(df['raw_pnl'] > 0).mean()*100:.1f}%  |  "
          f"Avg raw pnl: ${df['raw_pnl'].mean():+.2f}/trade\n")
    print(f"Running {N_SIMS:,} simulations per scenario...")

    results = []
    for label, params in SCENARIOS.items():
        r = run_scenario(df, label, **params)
        results.append(r)
        print(f"  {label}  done")

    print_table(results, recorded_pnl=float(df["pnl"].sum()))


if __name__ == "__main__":
    main()
