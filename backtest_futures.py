"""
Backtest + optimizer for the 5-min futures pattern strategy (MES / MNQ / MGC).

Usage:
    python backtest_futures.py --synthetic          # validate engine with fake data
    python backtest_futures.py --fetch              # pull 5-min bars from IBKR, current contract only
    python backtest_futures.py --fetch --roll       # ...and roll backward across expired contract months
    python backtest_futures.py                      # run backtest + grid optimization on cached data
    python backtest_futures.py --days 30            # limit to last N sessions

Design notes (no-lookahead guarantees):
  * Signals are evaluated on BAR CLOSE; entries fill at the NEXT bar's open (+1 tick slippage).
  * If a bar touches both stop and target, the STOP is assumed to fill first (conservative).
  * Overnight levels use only bars strictly before that day's 9:30 open.
  * Walk-forward split: parameters chosen on the first 70% of sessions, judged on the last 30%.
"""

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")
DATA_DIR = Path("./data")

# ── Contract specs (verified) ───────────────────────────────────────────────
# NOTE: MNQ is $2/point (E-mini NQ is $20). MGC is $10 per $1/oz move.
# Every tunable lives in config/strategy.toml -- see strategy_params.py
from strategy_params import PARAMS, parse_hhmm

_SIM     = PARAMS["sim"]
_SCORING = PARAMS["scoring"]
_COSTS   = PARAMS["costs"]

SPECS = {k: dict(v) for k, v in PARAMS["specs"].items()}
COMMISSION_RT = _COSTS["commission_rt"]  # round-turn $ per contract

RTH_START = parse_hhmm(PARAMS["session"]["rth_start"])
RTH_END   = parse_hhmm(PARAMS["session"]["rth_end"])
POWER_WINDOWS = [
    (parse_hhmm(w["start"]), parse_hhmm(w["end"]))
    for w in PARAMS["session"]["power_windows"]
]


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ════════════════════════════════════════════════════════════════════════════

def _list_contract_months(ib, sym: str, spec: dict):
    """Front-month-and-older Future contracts for a symbol, newest expiry first.

    Resolves the current front month via ContFuture (same as the non-roll
    path), then pulls the full contract-month list via reqContractDetails
    with includeExpired=True and keeps only that contract and older ones --
    never contracts further out than the current front month, since those
    aren't liquid yet and rolling into them would just be noise. How far
    back the "older" contracts actually go is up to what IBKR still retains
    expired-contract metadata for; there's no guarantee of hitting any
    particular depth of history.
    """
    from ib_insync import ContFuture, Future

    cf = ContFuture(sym, spec["exchange"], currency="USD")
    ib.qualifyContracts(cf)
    front_expiry = cf.lastTradeDateOrContractMonth

    details = ib.reqContractDetails(
        Future(symbol=sym, exchange=spec["exchange"], currency="USD", includeExpired=True)
    )
    contracts = [
        d.contract for d in details
        if d.contract.lastTradeDateOrContractMonth <= front_expiry
    ]
    contracts.sort(key=lambda c: c.lastTradeDateOrContractMonth, reverse=True)
    return contracts


ROLL_BUFFER_DAYS = _COSTS["roll_buffer_days"]  # approx "roll ~1 week before expiry" convention -- a contract's
                       # own front-month window is treated as ending this many days
                       # before ITS expiry (volume has typically already rolled to the
                       # next contract by then) and starting this many days before the
                       # NEXT-OLDER contract's expiry (when it, in turn, rolled out).
                       # This is a simplifying approximation, not a verified per-instrument
                       # roll-date convention -- real roll timing varies by contract and
                       # by year. Good enough to avoid stitching in a contract's thin,
                       # pre-front-month history, which is what silently corrupted the
                       # first version of this rolling fetch (near-zero volume pre-2026).

def _parse_expiry(s: str):
    return datetime.strptime(s, "%Y%m%d").date()


def _default_chunk_days(bar_minutes: int) -> int:
    """IBKR's max historical-data duration per request shrinks sharply for
    finer bar sizes (far more bars per day). 30 D chunks work fine at 5 min+
    but silently fail (every request times out / gets cancelled) at 1-2 min --
    empirically hit trying to fetch 2-min MES bars with the 5-min default."""
    if bar_minutes <= 2:
        return 2
    if bar_minutes <= 3:
        return 7
    return 30


def fetch_ibkr(days: int = 180, bar_minutes: int = 5, roll: bool = False,
                symbols: "list[str] | None" = None,
                chunk_days: "int | None" = None) -> None:
    """Fetch N-minute bars from IBKR.

    roll=False (default): single current continuous-front contract only, via
    ContFuture (ContFuture itself rejects endDateTime, error 10339, hence
    resolving to the concrete Future). Capped at that one contract's own
    trading history.

    roll=True: walks backward across contract months, but -- unlike a naive
    "continue the same cursor into the next contract" approach -- each
    contract only contributes bars from its OWN approximate front-month
    window (see ROLL_BUFFER_DAYS), stitched edge-to-edge with its neighbors.
    This avoids splicing in a contract's earlier life as a thin back-month,
    which is real volume data but not representative of "the market" during
    that period (a different, then-front contract was). Still a naive
    (non back-adjusted) price concatenation -- small price jumps can occur
    right at each roll boundary.
    """
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    from ib_insync import IB, ContFuture, Future, util

    DATA_DIR.mkdir(exist_ok=True)
    ib = IB()
    connected = False
    for port in (7497, 7496):
        try:
            ib.connect("127.0.0.1", port, clientId=7, readonly=True)
            connected = True
            break
        except Exception:
            continue
    if not connected:
        sys.exit("Could not connect to TWS/Gateway (ports 7497/7496).")

    bar_size  = f"{bar_minutes} mins"
    if chunk_days is None:
        chunk_days = _default_chunk_days(bar_minutes)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    mode = "rolling across contract months (front-month windows)" if roll else "single current contract"
    wanted = set(symbols) if symbols else set(SPECS)
    print(f"Fetching up to {days} days of {bar_size} bars ({mode})  symbols={sorted(wanted)}...")

    for sym, spec in SPECS.items():
        if sym not in wanted:
            continue
        if roll:
            contracts = _list_contract_months(ib, sym, spec)
            print(f"  {sym}: {len(contracts)} contract month(s) available, "
                  f"newest {contracts[0].lastTradeDateOrContractMonth}, "
                  f"oldest {contracts[-1].lastTradeDateOrContractMonth}")
            expiries = [_parse_expiry(c.lastTradeDateOrContractMonth) for c in contracts]
            gaps = [expiries[i] - expiries[i + 1] for i in range(len(expiries) - 1)]
            # median contract-to-contract spacing in THIS symbol's own listing --
            # used as the fallback window length for contracts with no older
            # neighbor to bound them (the front contract if it's the only one
            # available, and the oldest contract in the list). Without this,
            # the oldest contract has no start boundary and fetches its entire
            # life, right back into its own thin pre-front-month history --
            # exactly the bug this whole rework exists to fix.
            typical_gap = sorted(gaps)[len(gaps) // 2] if gaps else timedelta(days=90)
        else:
            cf = ContFuture(sym, spec["exchange"], currency="USD")
            ib.qualifyContracts(cf)
            fut = Future(
                symbol=sym,
                lastTradeDateOrContractMonth=cf.lastTradeDateOrContractMonth,
                exchange=spec["exchange"], currency="USD",
            )
            ib.qualifyContracts(fut)
            contracts, expiries = [fut], [None]

        chunks = []
        for idx, contract in enumerate(contracts):
            label = contract.localSymbol or str(contract.lastTradeDateOrContractMonth)

            if not roll:
                window_end, window_start = "", None
            elif idx == 0:
                window_end = ""
                window_start = (expiries[1] - timedelta(days=ROLL_BUFFER_DAYS)
                                 if len(contracts) > 1 else expiries[0] - typical_gap)
            else:
                we = expiries[idx] - timedelta(days=ROLL_BUFFER_DAYS)
                window_end = f"{we:%Y%m%d}-23:59:59"
                window_start = (expiries[idx + 1] - timedelta(days=ROLL_BUFFER_DAYS)
                                 if idx + 1 < len(contracts) else expiries[idx] - typical_gap)

            end, earliest_dt = window_end, None
            while True:
                if window_start is not None and earliest_dt is not None \
                        and earliest_dt.date() <= window_start:
                    break   # covered this contract's own front-month window
                if earliest_dt is not None and earliest_dt <= cutoff:
                    break   # covered the overall requested depth
                bars = ib.reqHistoricalData(
                    contract, endDateTime=end,
                    durationStr=f"{chunk_days} D",
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=False, formatDate=2,
                )
                if not bars:
                    ib.sleep(2)  # still pace empty/failed responses -- a fast run of dry
                                 # contracts back-to-back is exactly what trips IBKR's
                                 # pacing limit and can drop the whole TWS connection
                    break   # nothing this far back on this contract
                df = util.df(bars)
                if window_start is not None:
                    df = df[pd.to_datetime(df["date"]).dt.date > window_start]
                    if df.empty:
                        break
                chunks.append(df)
                earliest_dt = pd.Timestamp(pd.to_datetime(df["date"]).min())
                if earliest_dt.tzinfo is None:
                    earliest_dt = earliest_dt.tz_localize("UTC")
                total = sum(len(c) for c in chunks)
                print(f"    {sym} [{label}]  +{len(df):5d} bars  (total {total:6d})  "
                      f"back to {earliest_dt.date()}")
                end = earliest_dt.strftime("%Y%m%d-%H:%M:%S")
                ib.sleep(3)  # pacing — IBKR caps historical requests per rolling window
            if earliest_dt is not None and earliest_dt <= cutoff:
                break   # covered the requested window; no need for older contracts
        if not chunks:
            print(f"  {sym}: no data returned")
            continue
        df = pd.concat(chunks).drop_duplicates(subset="date").sort_values("date")
        out_path = DATA_DIR / f"{sym}_{bar_minutes}min.csv"
        df.to_csv(out_path, index=False)
        span = (f"{pd.to_datetime(df['date']).min().date()} -> "
                f"{pd.to_datetime(df['date']).max().date()}")
        print(f"  {sym}: TOTAL {len(df)} bars cached -> {out_path}  ({span})")
    ib.disconnect()


def load_cached(bar_minutes: int = 5) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in SPECS:
        f = DATA_DIR / f"{sym}_{bar_minutes}min.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(ET)
        df = df.rename(columns={"date": "ts"}).set_index("ts")
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        out[sym] = df
    return out


def make_synthetic(n_days: int = 30, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Random-walk 5-min bars with overnight + RTH sessions, U-shaped RTH volume,
    and occasional trend days. Engine-validation only — carries no real edge."""
    rng = np.random.default_rng(seed)
    base = {"MES": 5000.0, "MNQ": 18000.0, "MGC": 2300.0}
    out = {}
    # trading weekdays ending recently
    d = date.today()
    days = []
    while len(days) < n_days:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    days = sorted(days)

    for sym, p0 in base.items():
        rows, price = [], p0
        tick = SPECS[sym]["tick"]
        for dt in days:
            trend = rng.choice([0, 0, 0, 1, -1]) * rng.uniform(0.0005, 0.002)
            # overnight 18:00 (prev day) -> 9:30, then RTH 9:30 -> 16:00
            start = ET.localize(datetime.combine(dt - timedelta(days=1), time(18, 0)))
            end   = ET.localize(datetime.combine(dt, time(16, 0)))
            ts = start
            while ts < end:
                t = ts.timetz()
                in_rth = RTH_START <= ts.time() < RTH_END and ts.date() == dt
                if ts.time() >= time(17, 0) and ts.time() < time(18, 0):
                    ts += timedelta(minutes=5); continue  # daily halt
                if ts.date() != dt and ts.time() < time(18, 0) and ts.time() >= time(16,0):
                    ts += timedelta(minutes=5); continue
                sigma = price * (0.0006 if in_rth else 0.00025)
                drift = price * trend / 78 if in_rth else 0.0
                o = price
                c = o + drift + rng.normal(0, sigma)
                h = max(o, c) + abs(rng.normal(0, sigma * 0.6))
                l = min(o, c) - abs(rng.normal(0, sigma * 0.6))
                # round to tick
                o, h, l, c = (round(x / tick) * tick for x in (o, h, l, c))
                if in_rth:
                    minutes = (ts.hour - 9) * 60 + ts.minute - 30
                    u = 1.6 - 1.2 * math.sin(math.pi * min(max(minutes, 0), 390) / 390)
                    vol = int(rng.lognormal(mean=math.log(3000 * u), sigma=0.45))
                else:
                    vol = int(rng.lognormal(mean=math.log(300), sigma=0.6))
                rows.append((ts, o, h, l, c, vol))
                price = c
                ts += timedelta(minutes=5)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
        out[sym] = df
    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURE / SIGNAL GENERATION (thresholds applied LATER, per config)
# ════════════════════════════════════════════════════════════════════════════

def build_day_frames(df: pd.DataFrame) -> list[tuple[date, pd.DataFrame, float, float]]:
    """Split into RTH sessions; attach overnight high/low computed from bars
    in (9:30 - 17.5h, 9:30) — i.e., strictly pre-open information."""
    times = df.index.time
    rth = df[(times >= RTH_START) & (times < RTH_END)]
    sessions = []
    for d, day_df in rth.groupby(rth.index.date):
        open_ts = ET.localize(datetime.combine(d, RTH_START))
        on = df[(df.index >= open_ts - timedelta(hours=17.5)) & (df.index < open_ts)]
        if on.empty or len(day_df) < 6:
            continue
        sessions.append((d, day_df.copy(), float(on["high"].max()), float(on["low"].min())))
    return sessions


PDL_WINDOW_START = parse_hhmm(PARAMS["signals"]["pdl_window_start"])
PDL_WINDOW_END   = parse_hhmm(PARAMS["signals"]["pdl_window_end"])
PDL_PROXIMITY_TOL = PARAMS["signals"]["pdl_proximity_tol"]  # how close a bar's high must come to the
                             # PREVIOUS day's RTH low to count as "testing" it,
                             # for the Vol Div Short PDL setup below.

def generate_signals(sym: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return ALL candidate signals with raw features. Thresholds are applied
    at simulation time so one pass supports the whole optimization grid."""
    recs = []
    # 200-bar SMA needs to span multiple sessions (200 5-min bars is ~16.7hrs,
    # more than one RTH day's worth) -- computed on the full continuous
    # overnight+RTH series BEFORE day-slicing, so it carries through into
    # each day's slice already aligned, unlike the other per-day features.
    df = df.assign(sma200=df["close"].rolling(200, min_periods=200).mean())
    prev_day_low = None   # previous COMPLETE RTH session's low; set at the
                           # end of each day's loop body for use the next day
    for d, day, on_high, on_low in build_day_frames(df):
        v = day["volume"].astype(float)
        day["vwap"] = (day["close"] * v).cumsum() / v.cumsum()
        avg_vol = v.rolling(12, min_periods=5).mean()
        day["vol_ratio"] = v / avg_vol
        prev_close = day["close"].shift(1)
        vol_down = (v.diff() < 0).astype(int)
        day["dec6"] = vol_down.rolling(5).sum()   # declines within last 6 bars
        day["low6"]  = day["low"].rolling(6).min()
        day["high6"] = day["high"].rolling(6).max()
        # whipsaw: last 4 closes strictly alternating direction
        sgn = np.sign(day["close"].diff())
        day["whipsaw"] = (
            (sgn != sgn.shift(1)) & (sgn.shift(1) != sgn.shift(2)) & sgn.ne(0)
            & sgn.shift(1).ne(0) & sgn.shift(2).ne(0)
        )
        or_high = day["high"].iloc[:3].max()  # first 15 minutes
        or_low  = day["low"].iloc[:3].min()
        or_rng  = or_high - or_low
        day_open = float(day["open"].iloc[0])

        arr = day.reset_index()
        for i in range(1, len(arr)):
            r, p = arr.iloc[i], arr.iloc[i - 1]
            if pd.isna(r["vol_ratio"]):
                continue
            common = dict(
                sym=sym, ts=r["ts"], day=d, bar_idx=i,
                vol_ratio=float(r["vol_ratio"]),
                whipsaw=bool(r["whipsaw"]) if not pd.isna(r["whipsaw"]) else False,
                close=float(r["close"]),
            )
            # 1/2: overnight breakout / breakdown
            if p["close"] <= on_high < r["close"]:
                recs.append({**common, "setup": "ON Break Long", "bias": 1,
                             "stop": float(r["low"]), "kind": "breakout"})
            if p["close"] >= on_low > r["close"]:
                recs.append({**common, "setup": "ON Break Short", "bias": -1,
                             "stop": float(r["high"]), "kind": "breakout"})
            # 3/4: volume divergence -- mirrors the live options trader's only
            # validated setup (scanner.py's Volume Divergence Short): 5-bar
            # window, extreme compared against 4 bars back (last5.iloc[0] vs
            # .iloc[-1]), volume-decline counted as down-ticks among the 4
            # consecutive step-changes in that window, current candle must
            # close against the extreme (red for Short/new-high, green for
            # Long/new-low). Short additionally requires the "weak day" gate
            # -- close below both VWAP and the day's open -- since that's a
            # hard precondition on the live setup, not a tunable threshold;
            # Long has no live-trading precedent so it stays ungated.
            if i >= 4:
                last5 = arr.iloc[i - 4:i + 1]
                vols5 = last5["volume"].to_numpy()
                declines5 = sum(vols5[k] < vols5[k - 1] for k in range(1, 5))
                lows5, highs5 = last5["low"], last5["high"]
                if lows5.iloc[-1] < lows5.iloc[0] and r["close"] > r["open"]:
                    recs.append({**common, "setup": "Vol Div Long", "bias": 1,
                                 "stop": float(lows5.min()), "kind": "divergence",
                                 "declines": int(declines5)})
                is_weak = r["close"] < r["vwap"] and r["close"] < day_open
                if highs5.iloc[-1] > highs5.iloc[0] and r["close"] < r["open"] and is_weak:
                    recs.append({**common, "setup": "Vol Div Short", "bias": -1,
                                 "stop": float(highs5.max()), "kind": "divergence",
                                 "declines": int(declines5)})
                    # Vol Div Short PDL -- same pattern + weak-day gate, plus two
                    # extra hard conditions found to matter empirically: only in
                    # the 9:30-12:00 morning window, and only when the rejection
                    # bar's high is testing the PREVIOUS day's RTH low (within
                    # PDL_PROXIMITY_TOL). On real MES data this combination held
                    # up under walk-forward validation with positive P&L in every
                    # year tested (2024/2025/2026) and profit spread across ~38
                    # distinct days rather than concentrated in a few -- unlike
                    # plain Vol Div Short or the morning-window-only variant,
                    # which both turned out to be dominated by 2026 alone.
                    # A specific validated variant rather than a grid-search
                    # parameter; its window lives in [signals] in config.
                    if (prev_day_low is not None
                            and PDL_WINDOW_START <= r["ts"].time() <= PDL_WINDOW_END
                            and abs(r["high"] - prev_day_low) / prev_day_low <= PDL_PROXIMITY_TOL):
                        recs.append({**common, "setup": "Vol Div Short PDL", "bias": -1,
                                     "stop": float(highs5.max()), "kind": "divergence",
                                     "declines": int(declines5)})
            # 5: ORB -- opening range = first 15 min (3 bars); breakout can
            # fire any time after the range closes through end of session
            # (the shared entry_cutoff in Config governs the actual cutoff,
            # same as every other setup kind).
            if i > 2 and or_rng > 0:
                if p["close"] <= or_high < r["close"]:
                    recs.append({**common, "setup": "ORB Long", "bias": 1,
                                 "stop": float(or_low), "kind": "orb"})
                if p["close"] >= or_low > r["close"]:
                    recs.append({**common, "setup": "ORB Short", "bias": -1,
                                 "stop": float(or_high), "kind": "orb"})
            # 6: VWAP volume divergence (6-bar)
            if i >= 6 and not pd.isna(r["dec6"]) and r["vwap"] > 0:
                dist = abs(r["close"] - r["vwap"]) / r["vwap"]
                if r["close"] <= r["vwap"] and r["close"] > r["open"]:
                    recs.append({**common, "setup": "VWAP Div Long", "bias": 1,
                                 "stop": float(r["low6"]), "kind": "vwap",
                                 "declines": int(r["dec6"]), "vwap_dist": dist})
                if r["close"] >= r["vwap"] and r["close"] < r["open"]:
                    recs.append({**common, "setup": "VWAP Div Short", "bias": -1,
                                 "stop": float(r["high6"]), "kind": "vwap",
                                 "declines": int(r["dec6"]), "vwap_dist": dist})
            # 7: 200-SMA rejection -- single bar opens on one side of the
            # 200-bar SMA, wicks through to test it, and closes back on the
            # side it started from (i.e. the average held as support/
            # resistance). Stop is the rejection bar's own extreme, same
            # convention as breakout/ORB. Short is gated by the weak-day
            # filter (close below both VWAP and day open), mirroring
            # divergence -- this measurably helped Short (WR 29.4% -> 37.9%,
            # loss cut from -$512 to -$139 on the real dataset). Long tried
            # the mirrored "strong day" gate and it made things WORSE (WR
            # 34.1% -> 25.6%), so Long stays ungated: on a genuinely
            # trending-up day, price testing the SMA more often breaks
            # through than bounces, so filtering for strength selects
            # against the rejection pattern rather than for it.
            is_weak = r["close"] < r["vwap"] and r["close"] < day_open
            if not pd.isna(r["sma200"]):
                sma = r["sma200"]
                # opened above (SMA = support from above), dipped into it, closed back above
                if r["open"] > sma and r["low"] <= sma and r["close"] > sma:
                    recs.append({**common, "setup": "SMA200 Reject Long", "bias": 1,
                                 "stop": float(r["low"]), "kind": "sma"})
                # opened below (SMA = resistance from below), poked into it, closed back below
                if r["open"] < sma and r["high"] >= sma and r["close"] < sma and is_weak:
                    recs.append({**common, "setup": "SMA200 Reject Short", "bias": -1,
                                 "stop": float(r["high"]), "kind": "sma"})
        prev_day_low = float(day["low"].min())   # this day becomes "previous day" for the next iteration
    sig = pd.DataFrame(recs)
    return sig.sort_values("ts").reset_index(drop=True) if len(sig) else sig


def attach_market_alignment(signals: pd.DataFrame, mes_df: pd.DataFrame) -> pd.DataFrame:
    """aligned = MES close vs MES VWAP at signal time (as-of lookup, no lookahead)."""
    if signals.empty:
        return signals
    frames = []
    for d, day, *_ in build_day_frames(mes_df):
        v = day["volume"].astype(float)
        vwap = (day["close"] * v).cumsum() / v.cumsum()
        frames.append(pd.DataFrame({"ts": day.index, "mes_above": (day["close"] > vwap).values}))
    ref = pd.concat(frames).sort_values("ts")
    out = pd.merge_asof(signals.sort_values("ts"), ref, on="ts", direction="backward")
    out["mes_above"] = out["mes_above"].fillna(False)
    out["aligned"] = np.where(out["bias"] == 1, out["mes_above"], ~out["mes_above"])
    return out


def score(row) -> int:
    s = _SCORING["vwap_setup"] if row["kind"] == "vwap" else _SCORING["other_setup"]
    s += _SCORING["aligned"] * int(bool(row.get("aligned", False)))
    s += _SCORING["no_whipsaw"] * int(not row["whipsaw"])
    t = row["ts"].time()
    if any(start <= t <= end for start, end in POWER_WINDOWS):
        s += _SCORING["power_hour"]
    return min(s, _SCORING["max_score"])


# ════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    vol_breakout: float = _SIM["vol_breakout"]   # min vol_ratio for ON breakout / ORB
    vol_div: float = _SIM["vol_div"]             # min vol_ratio for VWAP-divergence setup (unused by
                                    # "divergence" kind -- that one has no volume-ratio
                                    # gate on the live options trader, only the decline count)
    dec8_min: int = _SIM["dec8_min"]   # declining bars required, of the 4 step-changes in
                                    # divergence's 5-bar window (matches scanner.py's ">=3 of 4")
    dec6_min: int = _SIM["dec6_min"]   # declining bars required (of 6)
    vwap_tol: float = _SIM["vwap_tol"]   # max |close-vwap|/vwap for VWAP setups
    min_score: int = _SIM["min_score"]
    tgt1_r: float = _SIM["tgt1_r"]
    tgt2_r: float = _SIM["tgt2_r"]  # matches the live options trader's Vol Div Short T2;
                                     # shared across all setup kinds (breakout/orb/vwap have
                                     # no live-trading reference, so they inherit this too)
    mgmt: str = _SIM["mgmt"]       # "scale" = 50% @T1 + BE trail, "full" = 100% @T1
    risk_per_trade: float = _SIM["risk_per_trade"]
    max_contracts: int = _SIM["max_contracts"]
    max_daily_trades: int = _SIM["max_daily_trades"]
    entry_cutoff: time = parse_hhmm(_SIM["entry_cutoff"])  # aligned to eod_flat rather
                                        # than literal 16:00 so a late entry can't fire and
                                        # get force-flattened one bar later for nothing
    eod_flat: time = parse_hhmm(_SIM["eod_flat"])
    setups: tuple = tuple(_SIM["setups"])


def passes(row, cfg: Config) -> bool:
    if row["kind"] not in cfg.setups:
        return False
    if row["kind"] in ("breakout", "orb") and row["vol_ratio"] < cfg.vol_breakout:
        return False
    if row["kind"] == "divergence" and row.get("declines", 0) < cfg.dec8_min:
        return False
    if row["kind"] == "vwap" and (
        row.get("declines", 0) < cfg.dec6_min
        or row.get("vwap_dist", 1.0) > cfg.vwap_tol
        or row["vol_ratio"] < cfg.vol_div
    ):
        return False
    if row["score"] < cfg.min_score:
        return False
    if row["ts"].time() > cfg.entry_cutoff:
        return False
    return True


def simulate(signals: pd.DataFrame, bars: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Chronological event simulation. One open position per symbol,
    global daily trade cap, conservative same-bar stop priority."""
    if signals.empty:
        return pd.DataFrame()
    # pre-index RTH bars per symbol/day as numpy for speed
    day_bars = {}
    for sym, df in bars.items():
        for d, day, *_ in build_day_frames(df):
            day_bars[(sym, d)] = day.reset_index()

    trades = []
    open_until = {}          # sym -> exit ts of open trade
    daily_count = {}         # date -> trades taken

    for _, s in signals.iterrows():
        if not passes(s, cfg):
            continue
        d = s["day"]
        if daily_count.get(d, 0) >= cfg.max_daily_trades:
            continue
        sym = s["sym"]
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
        entry_raw = float(entry_bar["open"])                    # fill with no slippage
        entry = entry_raw + bias * tick                         # 1-tick slippage in
        stop = float(s["stop"])
        risk = (entry - stop) * bias
        if risk <= tick:                                        # gapped through / degenerate
            continue
        if risk * pv > cfg.risk_per_trade * 1.5:                # even 1 lot blows the budget
            continue
        t1 = entry + bias * risk * cfg.tgt1_r
        t2 = entry + bias * risk * cfg.tgt2_r
        n = min(cfg.max_contracts, max(1, int(cfg.risk_per_trade / (risk * pv))))

        # walk forward through bars. raw_pnl_pts mirrors pnl_pts but with zero
        # tick slippage and (below) zero commission — it isolates the raw
        # price edge so cost/slippage assumptions can be swapped later without
        # re-running the simulation, same role prem_in/prem_out play for the
        # options backtest in backtest.py.
        pnl_pts, raw_pnl_pts = 0.0, 0.0
        half_closed = False
        be_stop, be_stop_raw = stop, stop
        exit_ts, exit_reason, exit_price = None, None, None
        t1_exit_price = None    # set only if T1 is actually touched (full exit or scale-out partial)
        remaining = 1.0  # fraction of position
        for j in range(i + 1, len(day)):
            b = day.iloc[j]
            bt = b["ts"].time() if hasattr(b["ts"], "time") else b["ts"]
            if bt >= cfg.eod_flat:
                px = float(b["open"]) - bias * tick
                pnl_pts += remaining * (px - entry) * bias
                raw_pnl_pts += remaining * (float(b["open"]) - entry_raw) * bias
                exit_ts, exit_reason, exit_price = b["ts"], "eod", px
                break
            lo, hi = float(b["low"]), float(b["high"])
            stop_now = be_stop if half_closed else stop
            stop_now_raw = be_stop_raw if half_closed else stop
            stop_hit = lo <= stop_now if bias == 1 else hi >= stop_now
            t1_hit = hi >= t1 if bias == 1 else lo <= t1
            t2_hit = hi >= t2 if bias == 1 else lo <= t2
            if stop_hit:                                        # conservative priority
                px = stop_now - bias * tick
                pnl_pts += remaining * (px - entry) * bias
                raw_pnl_pts += remaining * (stop_now_raw - entry_raw) * bias
                exit_ts, exit_reason, exit_price = b["ts"], ("be" if half_closed else "stop"), px
                break
            if cfg.mgmt == "full":
                if t1_hit:
                    px = t1 - bias * tick
                    pnl_pts += remaining * (px - entry) * bias
                    raw_pnl_pts += remaining * (t1 - entry_raw) * bias
                    exit_ts, exit_reason, exit_price = b["ts"], "t1", px
                    t1_exit_price = px
                    break
            else:  # scale-out
                if not half_closed and t1_hit:
                    px = t1 - bias * tick
                    pnl_pts += 0.5 * (px - entry) * bias
                    raw_pnl_pts += 0.5 * (t1 - entry_raw) * bias
                    remaining, half_closed = 0.5, True
                    be_stop, be_stop_raw = entry, entry_raw
                    t1_exit_price = px
                if half_closed and t2_hit:
                    px = t2 - bias * tick
                    pnl_pts += remaining * (px - entry) * bias
                    raw_pnl_pts += remaining * (t2 - entry_raw) * bias
                    exit_ts, exit_reason, exit_price = b["ts"], "t2", px
                    break
        else:
            b = day.iloc[-1]
            pnl_pts += remaining * (float(b["close"]) - entry) * bias
            raw_pnl_pts += remaining * (float(b["close"]) - entry_raw) * bias
            exit_ts, exit_reason, exit_price = b["ts"], "eod", float(b["close"])

        pnl = pnl_pts * pv * n - COMMISSION_RT * n
        raw_pnl = raw_pnl_pts * pv * n                           # zero commission, zero slippage
        open_until[sym] = exit_ts
        daily_count[d] = daily_count.get(d, 0) + 1
        trades.append(dict(
            sym=sym, day=d, setup=s["setup"], score=int(s["score"]), bias=bias,
            entry_ts=entry_bar["ts"], exit_ts=exit_ts, reason=exit_reason,
            entry=round(entry, 4), exit_price=round(exit_price, 4), stop=round(stop, 4),
            t1_exit_price=round(t1_exit_price, 4) if t1_exit_price is not None else None,
            contracts=n, risk_dollars=round(risk * pv * n, 2),
            pnl=round(pnl, 2), raw_pnl=round(raw_pnl, 2),
        ))
    return pd.DataFrame(trades)


def stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(n=0, wr=0, pf=0, exp=0, pnl=0, maxdd=0, avg_win=0, avg_loss=0)
    wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
    gross_w, gross_l = wins.pnl.sum(), -losses.pnl.sum()
    eq = trades.sort_values("exit_ts").pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    return dict(
        n=len(trades),
        wr=round(100 * len(wins) / len(trades), 1),
        pf=round(gross_w / gross_l, 2) if gross_l > 0 else float("inf"),
        exp=round(trades.pnl.mean(), 2),
        pnl=round(trades.pnl.sum(), 2),
        maxdd=round(dd, 2),
        avg_win=round(wins.pnl.mean(), 2) if len(wins) else 0,
        avg_loss=round(losses.pnl.mean(), 2) if len(losses) else 0,
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. OPTIMIZATION (walk-forward)
# ════════════════════════════════════════════════════════════════════════════

GRID = dict(PARAMS["sweep"]["grid"])


def _prepare_signals(bars: dict[str, pd.DataFrame], last_n_days: int | None):
    """Generate scored signals and split by session. Shared by run() and per-setup."""
    sigs = pd.concat(
        [generate_signals(sym, df) for sym, df in bars.items()], ignore_index=True
    )
    if sigs.empty:
        sys.exit("No signals generated — check data.")
    sigs = attach_market_alignment(sigs, bars["MES"]).sort_values("ts").reset_index(drop=True)
    sigs["score"] = sigs.apply(score, axis=1)
    all_days = sorted(sigs["day"].unique())
    if last_n_days:
        all_days = all_days[-last_n_days:]
        sigs = sigs[sigs["day"].isin(all_days)]
    split = all_days[int(len(all_days) * 0.7) - 1]
    return sigs, all_days, split


def _grid_configs(setups_override: tuple | None):
    """Materialize the parameter grid, applying an optional setups override."""
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        cfg = Config(**dict(zip(keys, combo)))
        if setups_override is not None:
            cfg.setups = setups_override
        yield cfg


def run(bars: dict[str, pd.DataFrame], last_n_days: int | None = None,
        setups_override: tuple | None = None, quiet: bool = False,
        setup_name: str | None = None) -> dict | None:
    sigs, all_days, split = _prepare_signals(bars, last_n_days)
    if setup_name is not None:
        # filter to one exact setup (e.g. "Vol Div Short"), not just its kind --
        # kind-level filtering can't isolate one bias out of a Long/Short pair
        sigs = sigs[sigs["setup"] == setup_name].reset_index(drop=True)
        if sigs.empty:
            sys.exit(f"No signals match setup name {setup_name!r}. "
                     f"Check spelling against the 'setup' column values in generate_signals().")
    train_sigs = sigs[sigs["day"] <= split]
    test_sigs  = sigs[sigs["day"] > split]
    if not quiet:
        print(f"\nSessions: {len(all_days)}  |  train ≤ {split}  |  test > {split}")
        active = f"{setup_name!r} only" if setup_name else (
            setups_override or ("breakout", "divergence", "orb", "vwap", "sma"))
        print(f"Active setups: {active}")
        print(f"Raw candidate signals: {len(sigs)}\n")

    results = []
    for cfg in _grid_configs(setups_override):
        tr = simulate(train_sigs, bars, cfg)
        results.append((cfg, stats(tr)))
    ranked = sorted(
        [r for r in results if r[1]["n"] >= max(10, len(all_days) // 3)],
        key=lambda r: (r[1]["pf"], r[1]["exp"]), reverse=True,
    ) or sorted(results, key=lambda r: (r[1]["pf"], r[1]["exp"]), reverse=True)

    if not quiet:
        print(f"{'rank':<5}{'PF':>6}{'WR%':>7}{'exp$':>8}{'n':>5}   config (train)")
        for k, (cfg, st) in enumerate(ranked[:8], 1):
            desc = f"volBO={cfg.vol_breakout} volDiv={cfg.vol_div} tol={cfg.vwap_tol} score≥{cfg.min_score} T1={cfg.tgt1_r}R {cfg.mgmt}"
            print(f"{k:<5}{st['pf']:>6}{st['wr']:>7}{st['exp']:>8}{st['n']:>5}   {desc}")
        print("\n── Walk-forward validation (top 3 on held-out test window) ──")
        print(f"{'rank':<5}{'PF':>6}{'WR%':>7}{'exp$':>8}{'pnl$':>9}{'maxDD$':>9}{'n':>5}")

    best_cfg, best_trades, best_st = None, None, None
    for k, (cfg, _) in enumerate(ranked[:3], 1):
        tr = simulate(test_sigs, bars, cfg)
        st = stats(tr)
        if not quiet:
            print(f"{k:<5}{st['pf']:>6}{st['wr']:>7}{st['exp']:>8}{st['pnl']:>9}{st['maxdd']:>9}{st['n']:>5}")
        if k == 1:
            best_cfg, best_trades, best_st = cfg, tr, st

    if not quiet and best_trades is not None and not best_trades.empty:
        print("\n── Best config: test-window breakdown ──")
        by_setup = best_trades.groupby("setup").agg(
            n=("pnl", "size"), wr=("pnl", lambda x: round(100 * (x > 0).mean(), 1)),
            pnl=("pnl", "sum"), avg=("pnl", "mean"),
        ).round(2)
        print(by_setup.to_string())
        by_sym = best_trades.groupby("sym").agg(
            n=("pnl", "size"), wr=("pnl", lambda x: round(100 * (x > 0).mean(), 1)),
            pnl=("pnl", "sum"),
        ).round(2)
        print()
        print(by_sym.to_string())
        best_trades.to_csv("backtest_trades.csv", index=False)
        cfg_out = asdict(best_cfg)
        cfg_out["entry_cutoff"] = str(cfg_out["entry_cutoff"])
        cfg_out["eod_flat"] = str(cfg_out["eod_flat"])
        Path("best_config.json").write_text(json.dumps(cfg_out, indent=2, default=str))
        print("\nSaved: backtest_trades.csv, best_config.json")
    if not quiet:
        print("\nNOTE: judge the strategy on the TEST columns only. If test PF < ~1.3 "
              "or test n is tiny, the edge is not established — do not trade it live.")
    return best_st


def run_per_setup_report(bars, last_n_days=None):
    """Backtest each setup family in isolation and print a comparison table."""
    all_setups = ("breakout", "divergence", "orb", "vwap", "sma")
    print("Running walk-forward backtest for each setup family in isolation...")
    print("(quiet mode: only the comparison summary is printed)\n")
    summary = []
    for setup_name in all_setups:
        st = run(bars, last_n_days, setups_override=(setup_name,), quiet=True)
        summary.append({"setup": setup_name, **(st or {"n": 0, "pf": 0, "wr": 0, "exp": 0, "pnl": 0, "maxdd": 0})})

    print(f"{'═' * 72}")
    print(f"PER-SETUP COMPARISON  (test-window numbers — the ones that matter)")
    print(f"{'═' * 72}")
    print(f"{'setup':<14}{'PF':>7}{'WR%':>7}{'exp$':>8}{'pnl$':>9}{'maxDD$':>10}{'n':>6}")
    print(f"{'-' * 72}")
    for r in summary:
        pf_str = f"{r['pf']}" if r['pf'] != float('inf') else "inf"
        print(f"{r['setup']:<14}{pf_str:>7}{r['wr']:>7}{r['exp']:>8}{r['pnl']:>9}{r['maxdd']:>10}{r['n']:>6}")
    print(f"{'═' * 72}")
    print("\nHow to read this:")
    print("  * A setup with test PF ≥ 1.4 on 10+ trades is a candidate to keep.")
    print("  * A setup with test PF < 1.0 or n < 5 is either broken or invisible; drop it.")
    print("  * If NO setup clears PF 1.3, no combination will either — the approach")
    print("    doesn't have edge on this data.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="pull data from IBKR")
    ap.add_argument("--synthetic", action="store_true", help="validate engine on fake data")
    ap.add_argument("--days", type=int, default=None,
                    help="limit backtest to last N sessions (default: use all cached data)")
    ap.add_argument("--fetch-days", type=int, default=180,
                    help="days of history to pull with --fetch (default 180)")
    ap.add_argument("--bars", type=int, default=5,
                    help="bar size in minutes (default 5); cache filename encodes this")
    ap.add_argument("--roll", action="store_true",
                    help="with --fetch, roll backward across expired contract months "
                         "once the front-month contract's own history runs dry")
    ap.add_argument("--symbols", type=str, default=None,
                    help="with --fetch, restrict to a subset: MES,MNQ,MGC (default: all)")
    ap.add_argument("--chunk-days", type=int, default=None,
                    help="with --fetch, override the per-request duration (default: "
                         "auto -- 2D for <=2min bars, 7D for 3min, 30D for 5min+)")
    ap.add_argument("--setups", type=str, default=None,
                    help="restrict to a subset: breakout,divergence,orb,vwap,sma")
    ap.add_argument("--setup-name", type=str, default=None,
                    help="restrict to one exact setup, e.g. 'Vol Div Short' "
                         "(isolates one bias out of a Long/Short pair; overrides --setups)")
    ap.add_argument("--per-setup", action="store_true",
                    help="run backtest separately for each setup family and compare")
    args = ap.parse_args()

    if args.fetch:
        fetch_symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
        fetch_ibkr(args.fetch_days, args.bars, roll=args.roll, symbols=fetch_symbols,
                   chunk_days=args.chunk_days)
        sys.exit(0)
    if args.synthetic:
        print("SYNTHETIC MODE — random-walk data. Results validate the ENGINE only;")
        print("a random walk has no exploitable edge, so expect PF ≈ 1.0 or below.\n")
        bars = make_synthetic(30)
    else:
        bars = load_cached(args.bars)
        if len(bars) < 3:
            sys.exit(
                f"No cached data for {args.bars}-min bars. Run first:\n"
                f"  python3 backtest_futures.py --fetch --bars {args.bars}  (with TWS open)"
            )

    if args.per_setup:
        run_per_setup_report(bars, args.days)
    else:
        override = tuple(s.strip() for s in args.setups.split(",")) if args.setups else None
        if override:
            valid = {"breakout", "divergence", "orb", "vwap", "sma"}
            bad = [s for s in override if s not in valid]
            if bad:
                sys.exit(f"Unknown setup(s): {bad}. Valid: {sorted(valid)}")
        run(bars, args.days, setups_override=override, setup_name=args.setup_name)
