# Yellow Box Strategy — Working Notes

**Status as of 2026-08-07: no established edge. Paused, not abandoned.**
Best configuration reaches full-history PF 1.15 with t=1.15 against a
Bonferroni-corrected bar of t≥3.51. Directionally improved a lot from where it
started (PF 0.84), but nothing here clears a significance bar and the dominant
effect in every test is a 2025-vs-2026 regime split, not the strategy.

Read this before touching the code again. The findings below are settled;
re-deriving them costs hours.

---

## 1. THE BOX — exact rules

The "Yellow Box" is a **TPO / Market Profile time-based value area**, built from
the **prior RTH session only**. It is fixed before the trading day opens and
never uses same-day or future information.

### Construction

1. **Source data**: the previous completed RTH session (09:30–16:00 ET), on the
   same bar timeframe used for execution.
2. **Range**: that session's high to low.
3. **Rows**: slice the range into `n_rows` rows (default **30**).
   - `row_size = max(tick, (high - low) / n_rows)`
   - `n_bins   = round((high - low) / row_size)`
   - row edges = `linspace(low, high, n_bins + 1)`
4. **TPO count**: for each bar in that session, add **+1 to every row its
   low–high range touches**. (Time-based: each bar counts once per row it spans,
   regardless of volume.)
5. **POC** (Point of Control) = the row with the highest count. Reported as the
   midpoint of that row.
6. **Value area**: start at the POC row. Repeatedly look at the row immediately
   above and immediately below the current area and **absorb whichever has the
   higher count**. Stop once the enclosed rows hold **≥ `va_pct` (default 70%)**
   of the session's total TPO count.
7. **VAH** = upper edge of the highest included row → **top of the box**
   **VAL** = lower edge of the lowest included row → **bottom of the box**

Notes:
- This is the common single-row greedy expansion. The textbook CBOT method
  compares rows in *pairs*. Close enough for backtesting, but expect small
  differences versus a charting platform's box.
- Boxes only exist from session 2 onward, and only once
  `avg_range_lookback` sessions of history exist.
- Verified: 0 lookahead violations across 543×2 sessions; box source session
  always ends strictly before the trading session opens.
- Typical box width on MES/MNQ: **23–80% of the prior day's range**, median 47%.

### Target levels ("Average Range")

- `avg_range` = mean of `(daily high − daily low)` over the trailing
  **`avg_range_lookback` (default 14)** sessions, **strictly before** today.
- **Average Range Resistance** (long target) = `POC + avg_range`
- **Average Range Support** (short target) = `POC − avg_range`

Both are anchored to the **prior day's POC**, not to today's open and not to the
broken box edge. This is deliberate and matches the original spec.

---

## 2. TRADE RULES

### Entry — breakout, retest, rejection

Two independent state machines run per session (long side and short side):

1. **BREAKOUT** — a bar **closes** fully outside the box
   (`close > VAH` arms the long side; `close < VAL` arms the short side).
2. **RETEST** — a later bar trades back and **touches** the broken edge
   (long: `low ≤ VAH`; short: `high ≥ VAL`).
3. **REJECTION** — that *same* bar **closes back outside** again
   (long: `close > VAH`; short: `close < VAL`). **Signal fires on this bar's close.**
4. The state machine then **re-arms**, so multiple retests per day are possible.
   A retest that fails to close back outside also re-arms it.

**Fill**: the **next bar's open**, plus **1 tick of adverse slippage**.
No bar is ever used before it has closed.

### Stop loss

```
long:  stop = retest_bar_low  −  stop_buffer_frac × (VAH − VAL)
short: stop = retest_bar_high +  stop_buffer_frac × (VAH − VAL)
```
Required to sit **strictly inside** the box (`VAL < stop < VAH`); if it doesn't,
**the signal is skipped**, never clamped.

The stop is **fixed** — it does not trail and does not move to breakeven.

⚠️ **`stop_buffer_frac` default is 0.0, which is NOT what the spec says.** The
spec says the stop goes *above* the retest swing high / *below* the swing low.
0.0 puts it exactly *on* that extreme. See §4 — this turned out to be the single
most important parameter in the entire strategy.

### Take profit

A **single fixed target** at the Average Range level (§1). No partials, no
runner, no trailing. `target_mode` also supports `r_multiple` and `scale`, but
see §4 — both tested worse than the plain range target.

### Exits — three ways a trade can end

1. **Stop** hit
2. **Target** hit
3. **Forced flat at 15:45 ET** — nothing is held overnight

Conventions (identical to `backtest_futures.py`):
- Same bar spans both stop and target → **stop assumed first** (conservative)
- 1 tick adverse slippage on entry *and* exit
- $1.40 round-turn commission per contract
- One open position per symbol; new signals while in a trade are skipped

---

## 3. DATA SITUATION — read before trusting any result

| Data | Status |
|---|---|
| `MES_5min.csv` | **Clean.** 543 sessions, 2024-06-23 → 2026-07-31 |
| `MNQ_5min.csv` | **Clean.** 543 sessions, same span |
| `MGC_5min.csv` | **CORRUPTED — do not use.** Every *even* calendar month has median volume 0 and 54–71% zero-volume bars |
| `*_3min.csv` | **CORRUPTED — do not use.** 92–99% zero-volume bars Nov 2025–Feb 2026; only Jun–Jul 2026 is real. Predates the roll fix |
| 2-minute bars | **Do not exist.** Cannot be derived from 5-min data — resampling only aggregates upward |

**The MGC bug (unfixed):** gold's actively-traded contracts are the *even*
months, but IBKR lists MGC monthly. `backtest_futures._list_contract_months`
(~line 51) keeps every listed month and gives each a front-month roll window, so
illiquid odd-month contracts fill in roughly the preceding even calendar month
with dead data. MES/MNQ are quarterly, so every listed contract is liquid and the
bug never surfaces there. **Fix requires a per-symbol active-months filter plus a
refetch with TWS running.** Any MGC result predating that fix is suspect —
including the Vol Div audit's Task 12 finding that MGC was a significant loser.

To get 2-min or clean 3-min data (needs TWS/Gateway open):
```
python3 backtest_futures.py --fetch --roll --bars 2 --symbols MES --fetch-days 760
```
Expect a long run (3s pacing per 30-day chunk across many contract months), and
IBKR may not retain 2-min history that far back on expired contracts. A 1-year
pull is more likely to complete intact.

---

## 4. WHAT WAS TESTED — settled findings

### Baseline (spec as written: 15-min, stop on the swing extreme, range target)
- MES+MNQ+MGC: **PF 0.84, n=892**, negative in every symbol and both directions
- MES+MNQ only: n=740, net **−$4,807.94**
- 71.5% of trades stopped out; **55% of stop-outs on the entry bar itself**
- Only **8.4%** ever reached the target; median nominal R:R was **~7R**

### Sweep 1 — box geometry (48 configs, `n_rows` × `va_pct` × `avg_range_lookback`)
**0/48 profitable on train, 48/48 profitable on test.** A unanimous sign flip is
a regime artifact, not an edge. Quarterly P&L confirmed it: negative through 2025
(2025Q2 −$1,480, 2025Q3 −$1,451), positive across all of 2026. The 70/30 split
lands at 2025-12-15, so "test" is simply *2026*.

### Sweep 2 — timeframe × target × stop (112 configs, MES, $1,000 account)
- **0/112 profitable on train**; **105/112 blew the $1,000 account** in training
- Best test t-stat anywhere: 1.70 vs a Bonferroni bar of **3.51**
- Every top config pinned at the maximum stop buffer tested → grid boundary, not
  an optimum

### ⚠️ FINDING 1 — risk-scaled targets made it WORSE (counterintuitive)
| target | target hit % | full PF |
|---|---|---|
| `r_multiple@1.0R` | 41.6 | **0.81** ← most reachable, worst |
| `r_multiple@1.5R` | 30.0 | 0.86 |
| `r_multiple@2.0R` | 21.4 | 0.89 |
| `r_multiple@3.0R` | 12.2 | 0.90 |
| `range` (original) | 12.6 | **0.93** ← least reachable, best |

Making the target reachable is **monotonically harmful**. The 8.4% hit rate was a
symptom, not the disease — cutting winners short costs more than the low hit rate
does. **Do not "fix" a low target-hit-rate by tightening the target.**

### ⚠️ FINDING 2 — the stop buffer is the dominant parameter
Monotonic all the way to the structural maximum (5-min, MES, range target):

| `stop_buffer_frac` | n | WR% | full PF | net |
|---|---|---|---|---|
| 0.30 | 468 | 37.6 | 1.08 | +$1,476 |
| 0.50 | 429 | 44.5 | 1.10 | +$1,888 |
| 0.70 | 405 | 48.4 | 1.13 | +$2,541 |
| 0.90 | 340 | 51.2 | **1.15** | +$2,389 |

At 0.9 the stop sits ~**104% of box width** from entry — i.e. past the far edge of
the value area. The retest swing has effectively stopped setting the stop level.
**What the sweep is really discovering is that a "reject at one box edge, stop at
the opposite edge" setup works better than the spec's retest-extreme stop.** That
is a different strategy and deserves testing as one, rather than being reached by
cranking a parameter to its limit.

### FINDING 3 — faster timeframes beat the spec's 15-min
| tf | best full PF | note |
|---|---|---|
| 5m | 1.15 | most trades (340), best t (1.15), best ex-top5 (+0.0339) |
| 10m | 1.21 | higher PF but only 231 trades |
| 15m | 1.00 | **spec's timeframe — worst; net ≈ $0 at every stop setting** |
| 30m | 0.93 | worst overall |

### FINDING 4 — a $1,000 account is not viable for this
- Median risk at the best stop setting: **$150.70/contract = 15.1% of $1,000**
- MES is already the smallest S&P instrument; 1 lot is the floor — no way to size down
- 105/112 configs ruined the account in training
- `scale` mode is untestable at this size — it needs ≥2 lots to split a position
  (verified: `scale@2R` produced byte-identical results to `r_multiple@2R` on all
  112 configs)
- Realistically needs **~$5,000–7,500** to keep per-trade risk near 2%

---

## 5. BEST CONFIGURATION FOUND (still not tradeable)

**MES, 5-min bars, `stop_buffer_frac=0.9`, `target_mode="range"`, 1 contract**

```
n=340   WR 51.2%   PF 1.15   net +$2,389   expR 0.0584   t-stat 1.15
expR excluding top-5 days: +0.0339   (the only version that stays positive here)

Exits:  eod    196 (57.6%)  avg  +$22.34   avg R +0.20   total  +$4,378
        stop    86 (25.3%)  avg −$140.78   avg R −1.01   total −$12,107
        target  58 (17.1%)  avg +$174.45   avg R +1.22   total +$10,118

Stop:   median 27.75 pts = $138.74   (104% of box width)
Target: median 43.78 pts = $218.92   median R:R 1.61  (was ~7R before the fix)
Winners avg +$107.87 (+0.77R)   Losers avg −$98.68 (−0.67R)
```

**Two things to notice:**
1. **58% of trades exit on the 15:45 clock**, not at a stop or target. The real
   profit-taking mechanism is "time ran out," not the target.
2. **Winners and losers are nearly the same size.** The system only makes money
   by winning ~51% on a coin-flip payoff — there is no payoff asymmetry to lean
   on, which is why it needs 340 trades to show +$2,389 and why t is only 1.15.

---

## 6. FILES

| File | Purpose |
|---|---|
| `strategy_yellow_box.py` | The strategy. All tunables on the `Config` dataclass |
| `report_yellow_box.py` | Illustrative walkthrough — pipeline, box examples, bar-by-bar worked trades, distributions |
| `sweep_yellow_box.py` | Sweep 1 — box geometry (MES+MNQ) |
| `sweep_yellow_box_v2.py` | Sweep 2 — timeframe × target × stop, with the $1,000 account model |
| `sweep_edge_only.py` | Same grid at flat 1 lot, no account constraint (ruin censors the sample, so this is the clean edge test) |
| `yellow_box_sweep*.csv`, `yellow_box_trades*.csv` | Results |

Everything reuses `backtest_futures.py` for data loading, day framing, specs, and
execution conventions. Treat that file as the trusted dependency.

**Verification already done** (independent re-implementation, 0 failures):
resample integrity, no-lookahead, TPO box recomputation, signal legality
(914/914 satisfy all four spec conditions), and full trade re-derivation
(740 vs 740 trades, max P&L difference $0.005).

---

## 7. WHERE TO PICK UP

The blocker is not parameters — it's that **the 2025/2026 regime split dominates
every result**. Train PF never exceeds ~1.0 while test PF runs 1.4–1.8, across
two completely independent parameter spaces. Until that's understood, any config
selected on this data is fitting the calendar.

In rough priority order:

1. **Multi-split walk-forward.** Test across many rolling split points instead of
   one 70/30 boundary. If the strategy only works after Dec 2025, that is the
   finding — and it kills the current approach regardless of PF.
2. **Investigate 2026 itself.** Both this strategy *and* the earlier Vol Div
   Short audit found 2026 to be their strongest year. Two unrelated strategies
   agreeing is either a real regime fact or a data fact — 2026 is the one segment
   coming from the current front-month contract's own native history, needing no
   roll splice. Resolve this before trusting anything that leans on 2026.
3. **Test "reject at one box edge, stop at the opposite edge"** as an explicit
   strategy (Finding 2), rather than as `stop_buffer_frac=0.9`.
4. **Fix the MGC roll bug** and refetch, if gold is ever wanted.
5. Only then revisit exits — the 58% EOD rate suggests a trailing stop or
   time-based partial might do real work. Note the Vol Div audit found trailing
   *hurt* mean-reversion setups; this is a continuation setup, so it's untested
   here, not known-bad.

**House rules for this project:** judge on the TEST window, apply a
multiple-comparisons correction to any sweep, and check whether a result survives
removing the top 5 days. Report PF near 1.0 or small n as inconclusive rather
than spinning it positive.
