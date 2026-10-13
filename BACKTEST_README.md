# Backtesting the Futures Pattern Strategy

## ⚠️ Fix these two bugs in `funded_scanner.py` first

**1. MNQ point value is $2, not $20.** The $20 figure is for the E-mini NQ; the *micro* is 1/10th. As written, the scanner under-sizes MNQ positions 10x (and any P&L math using it is 10x off).

```python
"MNQ": {..., "point_value": 2},   # was 20
```

**2. "GLOBEX" is no longer a valid IBKR exchange code.** Contracts will fail to qualify.

```python
"MES": {..., "exchange": "CME"},
"MNQ": {..., "exchange": "CME"},
"MGC": {..., "exchange": "COMEX"},
# and in the fetch code:
Future(sym, exchange=FUTURES[sym]["exchange"], currency="USD")
```

The backtester already uses the correct specs.

## Running it (on your machine, TWS open)

```bash
pip install ib_insync pandas numpy pytz

python backtest_futures.py --fetch            # caches ~45 days of 5-min bars to ./data/
python backtest_futures.py                    # backtest + grid optimization + walk-forward
python backtest_futures.py --synthetic        # engine self-test on fake data (no edge expected)
```

Outputs: ranked configs on the train window, top-3 validated on the held-out test window, `backtest_trades.csv` (every simulated trade), `best_config.json`.

## What the engine does (and why you can trust it)

- **No lookahead:** signals fire on bar close; fills happen at the *next* bar's open, ±1 tick slippage. Overnight levels use only pre-9:30 bars.
- **Conservative fills:** if a bar touches both stop and target, the stop is assumed to fill first.
- **Real costs:** $1.40 round-turn commission per contract (edit `COMMISSION_RT` to match your prop firm's routing).
- **Funded-account rules baked in:** fixed dollar risk sizing, skip trades where even 1 contract exceeds 1.5× the risk budget, max 8 trades/day, no entries after 15:00, flat by 15:45.
- **Walk-forward split:** parameters are chosen on the first 70% of sessions and judged on the last 30%. **Only the test-window numbers count.**

The grid searches: breakout volume threshold (1.5/2.0×), divergence volume (1.0/1.3×), VWAP tolerance (0.1%/0.2%), min score (2/3/4), T1 at 1.5R/2.0R, and full-exit vs. scale-out management.

## How to read the results — go/no-go criteria

| Test-window metric | Trade it | Don't trade it |
|---|---|---|
| Profit factor | ≥ 1.4 | < 1.3 |
| Trades in test window | ≥ 25 | < 15 (sample too small to conclude anything) |
| Max drawdown | < your daily loss limit × 2 | approaching your account's trailing drawdown |
| Train vs. test PF gap | small | train ≫ test (you've fit noise, not signal) |

The synthetic self-test illustrates the last row perfectly: on pure random-walk data the optimizer "found" a 2.3 PF config in-sample that promptly lost money out-of-sample. Expect some of that on real data too — that's the whole point of the held-out window.

## Honest caveats

- 45 days of 5-min data is ~30 sessions — enough for a first read, not a verdict. If the test window looks promising, re-fetch monthly and re-validate before trusting it (`--fetch-days 90` if your data subscription allows).
- Continuous-contract data stitches roll gaps; signals near roll dates (quarterly for MES/MNQ, roughly monthly liquidity shifts for MGC) can be slightly distorted.
- Backtest fills are optimistic in fast markets even with the conservative assumptions here. Paper trade the surviving config for 1–2 weeks before funded capital.
- One eval window ≠ robustness. If a config only wins with one exact parameter combo and loses with its neighbors, treat it as noise.

## Workflow

1. Fix the two scanner bugs above.
2. `--fetch`, then run the backtest.
3. If a config survives the test window, copy its values from `best_config.json` into the scanner's constants.
4. Paper trade 1–2 weeks; compare live alerts against what the backtest predicts.
5. Re-run the backtest monthly with fresh data — edges decay.
