"""gta6-trophy-rarity-forecast-01: share-logit damped-trend forecast.

For each trophy, the earn rate (percent of tracked players) is modelled on
the logit scale. A recency- and earner-weighted least-squares line through
the read history gives a slope; the slope is shrunk toward its tier's
median slope, clamped, and projected forward with a damped horizon so a
launch-night spike cannot run away over the ~4.5-day final horizon. The
spread combines a random-walk term, the slope's own uncertainty over the
damped horizon, and a small-count term, so flat histories stay close to
persistence while fast movers get honest width.

Run by the grader as `python3 -P -s /bot/main.py` (python:3.13-slim,
standard library only, no network). Reads every <id>.in in the input
directory and writes <id>.out per case:
    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
with one row per trophy, ids once, five finite non-decreasing numbers
0..100, file at most 65,536 bytes. Outside the grader, MAKEFUN_INPUT_DIR
and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Damping time constant (days) for the trend projection. Tuned: 1.4 bends
# too late for S-curve adoption, 2.2 over-projects saturating curves.
TAU = 1.7
# Recency half-life (days) for the history weights.
HALF_LIFE = 0.75
# Slope clamp, logit/day.
SLOPE_MIN = -0.10
SLOPE_MAX = 1.10
# Shrinkage strength: w = e_last / (e_last + 20).
SHRINK_E0 = 20.0
# Spread: random-walk volatility per sqrt-day on the logit scale.
SIGMA_RW = 0.30
# Spread floor/ceiling on the logit scale.
SIGMA_MIN = 0.18
SIGMA_MAX = 1.6
# Base spread term (logit units).
SIGMA_BASE = 0.10
# Never-seen trophy: median rate (percent) and logit spread.
UNSEEN_MEDIAN = 0.05
UNSEEN_SIGMA = 1.2
# Tier ceilings (percent): the projected median never exceeds these.
TIER_CEILING = {
    "platinum": 10.0,
    "gold": 65.0,
    "silver": 75.0,
    "bronze": 95.0,
}
# Fallback ceiling for an unknown tier string.
DEFAULT_CEILING = 95.0


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fit_slope(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Weighted least squares of y (logit rate) on t (days).

    points are (t, y, weight). Returns (slope, slope standard error).
    With fewer than two points there is no slope; with no time spread the
    slope is unidentified and the error is set wide.
    """
    if len(points) < 2:
        return 0.0, 1.0
    w_sum = sum(w for _, _, w in points)
    t_bar = sum(w * t for t, _, w in points) / w_sum
    y_bar = sum(w * y for _, y, w in points) / w_sum
    sxx = sum(w * (t - t_bar) ** 2 for t, _, w in points)
    if sxx <= 0.0:
        return 0.0, 1.0
    sxy = sum(w * (t - t_bar) * (y - y_bar) for t, y, w in points)
    slope = sxy / sxx
    intercept = y_bar - slope * t_bar
    n = len(points)
    if n > 2:
        resid_var = sum(w * (y - (intercept + slope * t)) ** 2
                        for t, y, w in points) / (n - 2)
    else:
        resid_var = 0.0
    se = math.sqrt(resid_var / sxx) if resid_var > 0.0 else 0.0
    return slope, se


def quantiles(logit_med: float, sigma: float) -> list[float | int]:
    out: list[float | int] = []
    prev = -1.0
    for z in Z:
        value = round4(inv_logit(logit_med + z * sigma))
        # Rounding can only move a value by <0.0001; force monotonicity.
        if isinstance(value, (int, float)) and value < prev:
            value = prev  # type: ignore[assignment]
        out.append(value)
        prev = float(value)
    return out


def forecast_trophy(
    reads: list[tuple[float, float, float]],
    tier_median_slope: float,
    tier: str,
    horizon_days: float,
) -> list[float | int]:
    """Forecast one trophy.

    reads: (t_days, rate_pct, players_tracked) for reads where the trophy
    was present, oldest first. Returns the five quantiles.
    """
    if not reads:
        # Never seen in any read: a low median with a wide spread.
        return quantiles(logit(UNSEEN_MEDIAN), UNSEEN_SIGMA)

    t_last, rate_last, players_last = reads[-1]
    earners_last = rate_last / 100.0 * players_last

    points: list[tuple[float, float, float]] = []
    for t, rate, players in reads:
        earners = rate / 100.0 * players
        recency = 0.5 ** ((t_last - t) / HALF_LIFE)
        weight = recency * math.sqrt(earners + 1.0)
        points.append((t, logit(rate), weight))
    slope, slope_se = fit_slope(points)

    # Shrink toward the tier median; a trophy with many earners keeps
    # its own slope, a thin one borrows the tier's.
    w = earners_last / (earners_last + SHRINK_E0)
    slope = w * slope + (1.0 - w) * tier_median_slope
    slope = min(SLOPE_MAX, max(SLOPE_MIN, slope))

    # Damped projection: the trend runs for an effective horizon shorter
    # than the real one, so saturating curves stop climbing.
    h_eff = TAU * (1.0 - math.exp(-horizon_days / TAU))
    logit_med = logit(rate_last) + slope * h_eff

    # Tier ceiling, never below the current rate (a trophy at 100% stays
    # ~100; the ceiling must not claw earners back).
    ceiling = logit(TIER_CEILING.get(tier, DEFAULT_CEILING))
    logit_med = min(logit_med, ceiling)
    logit_med = max(logit_med, logit(rate_last))

    sigma = math.sqrt(
        SIGMA_RW ** 2 * horizon_days
        + (slope_se * h_eff) ** 2
        + 1.0 / (earners_last + 4.0)
        + SIGMA_BASE ** 2
    )
    sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
    return quantiles(logit_med, sigma)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    t0 = parse_time(history[0]["at"])

    # Per-trophy read series: (days since first read, rate, players).
    series: list[list[tuple[float, float, float]]] = [
        [] for _ in case["trophies"]
    ]
    for read in history:
        t = (parse_time(read["at"]) - t0).total_seconds() / 86400.0
        players = read["players_tracked"]
        for i, rate in enumerate(read["rates"]):
            if rate is None:
                continue
            series[i].append((t, float(rate), float(players)))

    # Raw slopes first, so each tier's median can shrink its trophies.
    raw_slopes: list[float | None] = []
    for reads in series:
        if len(reads) < 2:
            raw_slopes.append(None)
            continue
        t_last = reads[-1][0]
        points = [
            (t, logit(rate), 0.5 ** ((t_last - t) / HALF_LIFE)
             * math.sqrt(rate / 100.0 * players + 1.0))
            for t, rate, players in reads
        ]
        raw_slopes.append(fit_slope(points)[0])

    tier_slopes: dict[str, list[float]] = {}
    for trophy, slope in zip(case["trophies"], raw_slopes):
        if slope is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(slope)
    tier_median = {
        tier: median(slopes) for tier, slopes in tier_slopes.items()
    }

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        med_slope = tier_median.get(trophy["tier"], 0.0)
        q = forecast_trophy(series[i], med_slope, trophy["tier"], horizon_days)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
