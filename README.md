# Futures Backtest Research

A backtesting and statistical-audit harness for intraday micro-futures
strategies on MES, MNQ and MGC. Built to find out whether candidate setups have
an edge — and the honest answer, for the two strategies studied here, is that
they do not.

> **Research code.** Not investment advice, and not a trading system. The
> headline finding of this repository is a pair of negative results.

## Findings

This is the point of the repo, so it goes first.

**Vol Divergence Short (MES).** A naive backtest shows n=78, PF 1.68, win rate
46.2%, net +$1,526. The audit in `AUDIT_VOL_DIV_SHORT_MES.md` does not support
it. Roughly 60 configurations were tested, so the multiple-comparisons
correction required is ≈3.3–3.4; the best result found anywhere in the audit
reaches only t=2.54. The result does not survive the correction, and it is not
robust across slices (year-level results weaken materially). **Treated as no
edge.**

**Yellow Box / Market Profile.** Documented in `YELLOW_BOX_NOTES.md`.
**PF 0.84 over n=892** across MES+MNQ+MGC — negative in every symbol and both
directions. 71.5% of trades stop out, and 55% of those stop on the entry bar
itself, which says the stop placement (the retest swing extreme) sits exactly
where price is still oscillating. The literal spec also puts TP1 a median ~7R
from that stop, so only 8.7% of trades ever reach it — the target is not scaled
to the risk. **Treated as no edge.**

Two results from the follow-up sweeps are worth more than the verdict:

- **A 48-config walk-forward sweep returned 0/48 profitable on train and 48/48
  profitable on test.** A unanimous sign flip is a regime artifact, not a
  discovery — the train/test boundary happened to land at 2025-12-15, and
  quarterly P&L is negative through 2025 and positive across 2026. Best test
  t-stat anywhere in that grid was 1.70 against a Bonferroni bar of 3.51.
- **Making the target reachable made performance worse.** Rescaling TP1 to 1.0R
  lifted the hit rate to 42% but produced the *worst* profit factor of any
  target mode (0.81), versus 0.93 for the original unreachable target. Cutting
  winners short cost more than the low hit rate did.

Both conclusions are the deliberate output of adversarial auditing rather than
a failure to find a good configuration. A harness that only ever confirms its
hypotheses is not measuring anything.

## Known issues

- **MGC cache is roughly half fabricated (unfixed).** Every *even* calendar
  month in `data/MGC_5min.csv` has median volume 0 and 54–71% zero-volume bars;
  odd months are normal. Cause: gold's actively traded contracts are the even
  months, but IBKR lists MGC monthly, and `_list_contract_months` keeps every
  listed month and assigns each a front-month roll window — so illiquid
  odd-month contracts backfill the preceding even month with dead data. MES and
  MNQ are quarterly, so every listed contract is liquid and the bug never
  surfaces there (0/26 bad months each). Fixing it needs a per-symbol
  active-months filter plus an MGC refetch. **Any MGC result in this repo
  predates that fix and is suspect**, including the audit's finding that MGC was
  a significant loser (t=−2.04).
- Two upstream scanner bugs this backtester works around rather than inherits:
  MNQ point value is **$2** (the micro), not the $20 of the E-mini NQ; and
  `GLOBEX` is no longer a valid IBKR exchange code (use `CME` / `COMEX`).

## Components

| File | Role |
|---|---|
| `backtest_futures.py` | Core engine: data loading, signal generation, scoring, simulation, grid search, walk-forward split |
| `strategy_yellow_box.py` | Market Profile / value-area "Yellow Box" strategy |
| `strategy_breakout_atr.py` | ATR breakout with volatility-regime conditioning |
| `audit_vds.py` | Statistical audit: bootstrap, multiple-comparisons correction, ablations |
| `sweep_yellow_box*.py`, `sweep_edge_only.py` | Parameter sweeps |
| `report_yellow_box.py` | Reporting for the Yellow Box runs |
| `cost_analysis_futures.py` | Commission and slippage sensitivity |

## Install

```bash
pip install -r requirements.txt
```

Python 3.11+ (`tomllib`). Fetching data requires a running IBKR TWS/Gateway
session with futures market-data entitlements.

## Configuration

**All strategy parameters are gitignored.** The committed example documents the
schema with placeholder values that generate no trades.

```bash
cp config/strategy.example.toml config/strategy.toml
# then edit config/strategy.toml
```

Override with `FUTURES_STRATEGY_CONFIG`. Sections:

- `[session]`, `[costs]` — RTH bounds, power-hour windows, commission, roll buffer
- `[signals]`, `[scoring]` — signal windows/tolerances and confluence weights
- `[sim]` — the multi-setup simulator: per-setup gates, score floor, target multiples, management mode, sizing, daily caps, cutoffs
- `[yellow_box]` — TPO rows, value-area coverage, target mode, sizing and account model
- `[breakout_atr]` — ATR windows, regime quantile, risk cap
- `[stats]` — bootstrap seed and iteration count
- `[sweep]` / `[sweep.grid]` — the grid-search space
- `[specs]` — contract specifications (exchange reference data, not strategy parameters)

## Usage

```bash
python backtest_futures.py --fetch        # cache 5-min bars to ./data/
python backtest_futures.py                # backtest + grid search + walk-forward
python backtest_futures.py --synthetic    # engine self-test on synthetic data
                                          # (no edge expected — this is the control)
python strategy_yellow_box.py             # Yellow Box strategy
python strategy_breakout_atr.py           # ATR breakout / regime study
python audit_vds.py                       # statistical audit
```

Outputs ranked configurations on the train window, top candidates validated on
a held-out test window, per-trade CSVs and `best_config.json`. Generated CSVs
and cached data are gitignored — rerun to regenerate.

## Method notes

- Train/test is a walk-forward split; grid-search winners are revalidated
  out-of-sample rather than reported from the fitted window.
- `--synthetic` runs the engine on data with no edge by construction. If it
  reports an edge, the engine is wrong — it is a control, not a demo.
- The audit applies bootstrap confidence intervals and corrects for the number
  of configurations tested. Uncorrected backtest statistics over a swept
  parameter space are not evidence.

## Limitations

- The sample is 543 sessions (roughly 2024-06 through 2026-07). That is enough
  to reject a weak edge but not enough to characterize rare regimes, and the
  per-bucket counts in the regime analysis get small fast.
- Fills are modeled, not real. Stops are assumed to fill at the stop level plus
  configured slippage, which flatters gap-through scenarios.
- Position scaling assumes a divisible position; at 1 contract you cannot take
  half off, so `scale` mode needs ≥2 lots to trade as modeled.
- Day margin is broker-dependent and only valid for strategies that stay flat
  overnight.

## License

No license granted. All rights reserved.
