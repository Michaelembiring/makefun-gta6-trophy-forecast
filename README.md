# Share-logit damped trend — a GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Michaelembiring/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Michaelembiring/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI trophy's Exophase earn rate one week after launch,
as five quantiles (5/25/50/75/95) on the rate, scored by pinball loss on
the logit scale. The program is `main.py` (Python 3.13, standard library
only): it reads `/input/<id>.in` and writes `/output/<id>.out`.

## Model

Work on logit(rate), per trophy:

- **Trend:** recency-weighted (half-life 0.75 d) and earner-weighted
  (`sqrt(earners + 1)`) least squares of logit(rate) against days since
  the first read, keeping the slope and its standard error.
- **Shrinkage:** the slope is shrunk toward its tier's median slope
  (platinum/gold/silver/bronze) with weight `e_last / (e_last + 20)`,
  then clamped to [-0.10, 1.10] logit/day — thin trophies borrow the
  tier's trend, thick ones keep their own.
- **Projection:** `logit_med = logit(last rate) + slope * tau * (1 - exp(-H/tau))`,
  `tau = 1.7` days, `H` the horizon in days. The damping stops a
  launch-night spike from running away over the ~4.5-day final horizon.
  The median is capped at a tier ceiling (platinum 10%, gold 65%,
  silver 75%, bronze 95%), never below the current rate — a trophy at
  100% stays ~100%.
- **Spread:** `sigma = sqrt(0.30^2 * H + (slope_se * H_eff)^2 +
  1/(e_last + 4) + 0.10^2)`, clamped to [0.18, 1.6] logit units.
  Quantiles are `inv_logit(logit_med + z * sigma)`, rounded half-up to
  4 decimals. A trophy never seen in any read gets median 0.05% with
  sigma 1.2.

Deterministic, no network, ~0.06 s per case. Handles irregular read
spacing and changing `players_tracked` (earners use each read's own
count).

## Validation (redone in-session)

- Kit practice cases p00/p01: 55 rows, monotone quantiles, 2,694 bytes;
  flat launch-night history reproduces persistence (max |q50 − last
  rate| = 0.01, median q95−q05 spread 0.96 pp); byte-identical rerun.
- Synthetic worlds (2.5 d of 30-min reads, target day 7), mean pinball
  loss on the logit scale vs the starter's persistence:
  - logistic adoption: **0.1575 vs 0.2505** (0.63x)
  - probit S-curve adoption: **0.4306 vs 0.6333** (0.68x)
- Edge cases: a trophy at 100% stays ~100
  (q50 99.99); a never-seen trophy gets median 0.05 with a wide spread;
  a trophy missing from the last reads anchors on its last seen rate.
