# Diagnostic Audit: "Volume Divergence Short" MES Strategy

Audit of the strategy as specified: 5-bar window divergence pattern, hard weak-day gate,
09:30–12:00 ET window, stop-first 5-bar-high stop, 2.0R full-exit target, min score 2,
1 contract, max 3 trades/day, $750 risk cap. Baseline reproduces the reported walk-forward
result exactly: **n=78, PF 1.68, win% 46.2, net $1,525.80** (`baseline_trades.csv`). Most
diagnostics below run on the full 544-session history (n=221 trades, `baseline_trades_full_history.csv`)
for statistical power; the 78-trade slice is the walk-forward-honest subset.

Standalone simulator (`audit_vds.py`) built and verified to reproduce the original
`backtest_futures.py` `simulate()` output exactly (same entry/exit prices, same $1.40/contract
commission) before any diagnostic was run.

## The three things that would most change my conclusion

1. **Multiple-comparisons correction eats the whole result.** ~60 configurations were tested
   across this audit (see close-out below). The Bonferroni-adjusted t-stat bar for that many
   comparisons is ≈3.3–3.4. The single best result found anywhere in the audit — the high-ATR
   regime quartile — hits t=2.54. Nothing clears the corrected bar. The originally reported
   headline (t≈1.95 on the 78-trade slice) isn't close.

2. **The edge is a volatility regime effect wearing a pattern's clothes.** Expectancy rises
   monotonically and cleanly from ATR Q1 (expR −0.22, a loser) to Q4 (expR +0.51, t=2.54) — the
   only genuinely clean, non-spiky, significant-on-its-own-terms result in the whole audit. This
   also plausibly explains the year-to-year swings better than "year" itself does.

3. **It does not survive on a second instrument.** MGC, same unmodified config: PF 0.77, expR
   −0.23, **t=−2.04 — significantly negative**, opposite sign from MES. MNQ is directionally
   consistent but not itself significant (t=1.76). One instrument agrees, one flatly disagrees.

## Per-experiment results

### Task 1 — Score gate inertness
| Reference | Gate-passing bars | "Aligned" fire rate | Score≥2 pass rate |
|---|---|---|---|
| MES vs itself (as coded) | 267 | **100.0%** | 100.0% |
| SPY (independent, 59-day yfinance cap) | 38 matched | 94.7% | 97.4% |

Confirmed exactly as hypothesized: self-referencing MES makes the "aligned" score component a
tautology with the weak-day gate — literally 100%, rejects nothing. Swapping in an independent
reference doesn't meaningfully fix this: SPY and MES are correlated enough that the component
still fires 94.7% of the time. The gate does not bind under either reference. Sample caveat:
yfinance only serves 60 days of 5-min history, so the SPY check is n=38, not the full period.

### Task 2 — Intrabar sequencing
| Resolution | n | Win% | PF | expR | t-stat |
|---|---|---|---|---|---|
| Stop-first | 221 | 41.6 | 1.36 | 0.152 | 1.54 |
| Target-first | 221 | 42.5 | 1.37 | 0.179 | 1.81 |
| Coinflip | 221 | 42.1 | 1.36 | 0.166 | 1.67 |

Only 2 of 221 trades ever had a single bar touch both stop and target. Resolution choice is
functionally irrelevant here — the headline result is not a fill-order artifact.

### Task 3 — Ablation
| Variant | n_cands | n | Win% | PF | expR | t-stat |
|---|---|---|---|---|---|---|
| (a) red only, no gate | 6,912 | 1,328 | 32.7 | 0.92 | **−0.164** | **−4.47** |
| (b) gate + red | 3,327 | 542 | 40.6 | 1.12 | 0.037 | 0.68 |
| (c) gate+red+higher-high | 650 | 416 | 36.5 | 1.11 | 0.003 | 0.04 |
| (d) gate+red+vol-divergence | 1,089 | 400 | 40.8 | 1.16 | 0.077 | 1.13 |
| (e) full pattern | 267 | 221 | 41.6 | 1.36 | 0.152 | 1.54 |

Shorting any red morning bar with no filter is a significant loser (t=−4.47). The weak-day gate
alone flips the sign but isn't significant. **Higher-high alone adds nothing** (c is *worse* than
b). Volume-divergence alone adds a bit more (d). Only the full combination looks best, but it's
also the smallest sample of the five and still doesn't clear significance — this reads as
"needs both conditions together, weakly," not "volume divergence is the active ingredient."

### Task 4 — VWAP warm-up / slope
| Warm-up | n_cands | n | PF | expR | t-stat |
|---|---|---|---|---|---|
| 0/2/4 bars | 267 | 221 | 1.36 | 0.152 | 1.54 |
| 6 bars | 254 | 210 | 1.42 | 0.167 | 1.64 |
| 8 bars | 243 | 203 | 1.53 | 0.194 | 1.86 |
| 12 bars | 212 | 179 | 1.65 | 0.214 | **1.92** |

Warm-up 0/2/4 are identical because the pattern's own 5-bar requirement already forces i≥4.
Beyond that, longer warm-up produces a gradual, plausible-looking improvement — but it never
clears 1.96, and n keeps shrinking as the threshold tightens, so this could equally be "later
trades in the window are just better" (consistent with the cutoff-sweep finding below) rather
than a VWAP-maturity effect specifically.

**VWAP slope < 0**: bind rate **0.0%** against the weak-day gate. This is expected, not
surprising — a cumulative VWAP's slope is mechanically pulled negative by a bar that closes
below it. Redundant by construction.

### Task 5 — Volume as a clock
Time-of-day-normalized volume (median over prior 20 sessions, shifted) cuts the candidate pool
from 267 → 178 (33%), fairly uniformly across the session (half-hour buckets: 13→6, 42→28,
60→34, 69→52, 73→52, 10→6 — no single bucket disproportionately vanishes). Simulated result:
n=146, PF 1.36, expR 0.136, **t=1.10** — weaker than the raw-volume baseline (t=1.54), not
stronger. Normalizing doesn't uncover a cleaner signal; it just shrinks the sample.

### Task 6 — Dollar cap vs. ATR band
$750 cap bind rate: **0.0%** (n=267 before and after). Max stop distance in the entire dataset
is $315 (63 points) — confirmed dead code, as hypothesized.

| ATR band | n_cands | bind% | n | PF | expR | t-stat |
|---|---|---|---|---|---|---|
| [0.5, 1.5] | 214 | 19.9 | 182 | 1.48 | 0.203 | 1.84 |
| [0.5, 1.0] | 153 | 42.7 | 135 | 1.38 | 0.160 | 1.25 |
| [0.75, 1.5] | 152 | 43.1 | 136 | 1.39 | 0.204 | 1.61 |
| [0.25, 2.0] | 257 | 3.7 | 215 | 1.40 | 0.162 | 1.60 |

A genuine plateau — every band lands in a tight PF 1.38–1.48 range, no standout.

### Task 7 — Score decomposition (min_score=1, unfiltered pool n=267)
| Component | Value | n | PF | expR | t-stat |
|---|---|---|---|---|---|
| aligned | True | 267 | 1.36 | 0.152 | 1.54 |
| aligned | False | 0 | — | — | degenerate (Task 1) |
| not_whipsaw | True | 178 | 1.48 | 0.154 | 1.39 |
| not_whipsaw | False | 51 | 1.00 | 0.069 | 0.34 |
| time_bonus | True | 58 | **1.00** | **0.039** | 0.20 |
| time_bonus | False | 172 | **1.57** | **0.213** | 1.88 |

`not_whipsaw` points the expected direction (cleaner price action → better trades), though not
significant. **`time_bonus` is inverted**: bars that get the score's "first hour" bonus point
perform *worse* (PF 1.00) than bars that don't (PF 1.57). The score system is rewarding exactly
the wrong sub-window within this window.

### Task 8 — Exit structure
| Exit | n | Win% | PF | expR | expR(ex-top5) | t-stat |
|---|---|---|---|---|---|---|
| Fixed 2.0R (baseline) | 221 | 41.6 | 1.36 | 0.152 | 0.064 | 1.54 |
| Trail prior-bar-high | 241 | 32.4 | 0.97 | −0.033 | −0.096 | −0.50 |
| Chandelier (2.5×ATR) | 223 | 39.0 | 1.19 | 0.088 | −0.002 | 0.93 |
| Breakeven-at-1R-then-trail | 234 | **25.6** | **0.65** | **−0.275** | −0.327 | **−4.73** |
| Fixed 1.5R | 226 | 46.0 | 1.22 | 0.059 | −0.013 | 0.71 |
| Fixed 3.0R | 210 | 31.9 | 1.17 | 0.099 | −0.012 | 0.81 |
| Fixed 4.0R | 207 | 28.5 | 1.27 | 0.117 | −0.015 | 0.83 |

Every alternative exit underperforms the fixed 2R baseline; breakeven-then-trail is a
significant loser. Win rate falls as expected when trying to let winners run, but expectancy
falls too, not rises — **the moves die near 2R; this is mean-reversion behavior, not
continuation**, exactly per the audit's own framing. Also notable: expR-ex-top5 is negative or
near-zero at *every* target horizon except the exact 2.0R already in use — the apparent edge is
fragile everywhere else on this axis.

### Task 9 — Daily cap
| Cap | n | PF | expR | t-stat |
|---|---|---|---|---|
| 1/day | 175 | 1.24 | 0.081 | 0.73 |
| 2/day | 213 | 1.34 | 0.132 | 1.31 |
| 3/day | 221 | 1.36 | 0.152 | 1.54 |
| 5/day | 221 | 1.36 | 0.152 | 1.54 |
| Conditional (extra only if prior won & still below VWAP) | 193 | 1.37 | 0.123 | 1.16 |

Cap 3 and cap 5 are identical — no day ever produced more than 3 qualifying signals, so the cap
is only partially binding (matters between 1–3, not beyond). The conditional variant doesn't
clearly help.

### Task 10 — Plateau sweeps
**Volume decline threshold** (of 4 step-changes):
| ≥k | n_cands | n | PF | expR | t-stat |
|---|---|---|---|---|---|
| ≥1 | 647 | 414 | 1.11 | 0.000 | 0.00 |
| ≥2 | 559 | 378 | 1.20 | 0.051 | 0.68 |
| ≥3 (current) | 267 | 221 | 1.36 | 0.152 | 1.54 |
| ≥4 | 42 | 40 | 1.19 | 0.114 | 0.48 |

Gradual climb through ≥3, then falls off at ≥4 — but n=40 there, and its expR-ex-top5 is
strongly negative (−0.149), so that's a thin-sample wobble, not continued improvement. Plateau
shape, not a spike.

**Window size** (lookback bars, decline threshold scaled ∝60% of window — a real limitation,
two parameters move together):
| W | n_cands | n | PF | expR | t-stat |
|---|---|---|---|---|---|
| 3 | 239 | 215 | 1.01 | 0.060 | 0.60 |
| 4 | 473 | 357 | 1.19 | 0.070 | 0.91 |
| 5 (current) | 267 | 221 | 1.36 | 0.152 | 1.54 |
| **6** | 134 | 111 | **1.67** | **0.320** | **2.27** |
| 7 | 246 | 178 | 1.26 | 0.170 | 1.56 |

**W=6 is a spike, not a trend.** It's the only cell in this entire audit to clear 1.96, and it's
flanked on both sides by meaningfully lower values (5→1.54, 7→1.56). Per the audit's own ground
rule: call this out as a fit. Do not treat it as validated.

**Entry cutoff time**:
| Cutoff | n_cands | n | PF | expR | t-stat |
|---|---|---|---|---|---|
| 10:30 | 64 | 58 | 1.00 | 0.039 | 0.20 |
| 11:00 | 122 | 106 | 1.15 | 0.107 | 0.75 |
| **12:00 (current)** | 267 | 221 | **1.36** | **0.152** | 1.54 |
| 13:00 | 397 | 311 | 1.18 | 0.054 | 0.66 |
| 16:00 (full day) | 705 | 503 | 1.08 | −0.057 | −0.92 |

This is a genuine plateau/peak, not a spike — smooth rise to 12:00, smooth fall after, going
negative by end of day. The one sweep dimension where the current setting sits at a defensible,
non-arbitrary-looking local maximum.

### Task 11 — Regime and year
**ATR quartile** (day's mean 20-bar ATR):
| Quartile | n | Win% | PF | expR | expR(ex-top5) | t-stat |
|---|---|---|---|---|---|---|
| Q1 (low) | 56 | 30.4 | 0.60 | −0.220 | −0.429 | −1.19 |
| Q2 | 56 | 37.5 | 0.93 | 0.034 | −0.195 | 0.18 |
| Q3 | 54 | 46.3 | 1.70 | 0.292 | −0.014 | 1.43 |
| **Q4 (high)** | 55 | 52.7 | **1.81** | **0.514** | **0.222** | **2.54** |

**By year**:
| Year | n | PF | expR | t-stat |
|---|---|---|---|---|
| 2024 | 48 | 1.63 | 0.184 | 0.85 |
| 2025 | 104 | 1.03 | 0.012 | 0.08 |
| 2026 | 69 | 1.73 | 0.342 | 1.88 |

The ATR-quartile gradient is monotonic and clean, and Q4 is the single strongest, most
independently-supported result in the audit (survives ex-top5). Year-level results are weaker
and noisier by comparison — 2025 essentially flat, matching this session's earlier findings.

**Signal-frequency check** (the audit brief assumed an ~8x 2025-vs-2026 gap): **not found in
the current data.** Gate-passing signals per session: 2024 ≈1.26/session, 2025 ≈1.32/session,
2026 ≈1.31/session — consistent. Raw session counts differ (136/257/150) because 2025 is a full
calendar year in this dataset and 2024/2026 are partial. Median 5-min bar volume does rise
across years (875→1,263→1,777) but signal rate per session does not track it proportionally.
Whatever produced an apparent 8x gap in earlier analysis most likely reflects the pre-2026
zero-volume data-corruption bug already found and fixed earlier in this session (the original
naive contract-rolling logic spliced in each contract's thin pre-front-month history); this
audit runs entirely on the corrected, front-month-windowed dataset.

### Task 12 — Cross-instrument
| Symbol | n_cands | n | Win% | PF | expR | t-stat |
|---|---|---|---|---|---|---|
| MES (reference) | 267 | 221 | 41.6 | 1.36 | 0.152 | 1.54 |
| MNQ | 285 | 227 | 41.0 | 1.41 | 0.169 | 1.76 |
| MGC | 170 | 135 | 28.1 | **0.77** | **−0.233** | **−2.04** |

MNQ agrees in direction, not itself significant. **MGC disagrees in direction and is
significant** — the unmodified config is a real loser on gold.

## Configuration count and multiple-comparisons correction

Total distinct configurations evaluated across Tasks 2–12 (excluding Task 1's non-trade-level
diagnostics): **≈60** (3 + 5 + 7 + 1 + 4 + 5 + 6 + 5 + 14 + 7 + 2, plus the baseline reference).

For m≈60 comparisons at family-wise α=0.05 (Bonferroni), the per-comparison threshold is
0.05/60 ≈ 0.00083, corresponding to a two-tailed **t-stat bar of ≈3.3–3.4**.

The best result anywhere in this audit — ATR Q4 — reaches **t=2.54**. The window=6 spike reaches
2.27. The originally reported headline (78-trade walk-forward slice) sits at **t≈1.95**. None of
these clear an uncorrected 1.96 bar convincingly, and *none* clear the Bonferroni-adjusted bar.
The gap between "best observed" and "corrected significance" is roughly 0.8–1.0 t-units — not
close.

## Definition of done

**1. Is expectancy still positive with the top 5 P&L days removed?**
Yes, barely, on the full-history baseline (expR 0.152 → 0.064) and on the original 78-trade
walk-forward slice. But it goes negative or near-zero on almost every variant tested elsewhere
in this audit (every fixed-target horizon except 2.0R, most ATR bands' lower quartiles, several
ablation stages). The positive result is not robust to which slice you look at — it survives
in exactly the configuration already chosen, and largely fails elsewhere.

**2. Do the sweeps show plateaus or spikes?**
Mixed, and it matters which axis: decline-threshold, ATR-band, and cutoff-time sweeps are all
genuine plateaus with smooth local peaks. The window-size sweep produced one clear **spike**
(W=6, t=2.27, flanked by lower values on both sides) that should not be trusted despite being
the best-looking number in that table.

**3. Does the direction survive on MNQ and MGC?**
Partially. MNQ: same direction, not significant. MGC: **opposite direction, significant**
(t=−2.04). One instrument doesn't contradict, one actively does.

**4. Which single component is carrying the edge, or is there no identifiable one?**
No single pattern component carries it. The score gate's "aligned" component is dead
(tautological). "Higher-high" alone adds nothing (Task 3, ablation b→c). "Volume divergence"
alone adds a little, not significantly. "Time bonus" is miscalibrated in the wrong direction.
The clearest, most defensible signal in the whole audit isn't a pattern-recognition component
at all — it's **volatility regime** (Task 11, ATR quartile, t=2.54, monotonic, survives
ex-top5, the closest thing to a real effect here). If there is a real edge in this data, the
evidence points at "shorting weak setups on high-ATR days," not at the specific volume-
divergence/higher-high mechanics the strategy is built around.

## Bottom line

The headline PF/win-rate is not evidence of anything on its own, per the brief — and across
~60 configurations tested here, nothing clears a corrected significance bar. The strongest
individual signal found (ATR-quartile regime effect) is not what the strategy's rules select
for. This reads as a strategy without an identified, component-level edge, sitting on top of a
plausible-but-unproven volatility-regime effect that the current rules don't isolate. A
negative result, as specified.
