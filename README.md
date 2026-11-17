[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AndryXSad/gta6-twitch-ccu-forecast-01-v39/rank.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AndryXSad/gta6-twitch-ccu-forecast-01-v39/reward.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01 — v39

Forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2–24 of
release day, 19 Nov 2026), as five quantiles (5/25/50/75/95%) per hour,
from the case file's comparable-launch library plus the target's hour-1
print and pre-release history. Python 3.13, stdlib only, deterministic,
no network calls. Run `python3 main.py` with `MAKEFUN_INPUT_DIR` /
`MAKEFUN_OUTPUT_DIR` pointing at the case / output directories.

## Model

Five member predictors (y5, z0, y1, clone, mv), each a weighted blend of
three views — a per-hour OLS on `[lnV1, lnPREM, g12]`, a console-only
shape view, and an all-launch shape view — combined with the v38
packaged combine cell (per-band ramp exponents, per-band disagreement
modes, per-band shift slopes with `size**1.25`). See `DESIGN.md` for the
full construction and the honest caveats.

## Result — NO-GO (honest)

This round does **not** beat v38 and was **not submitted**:

| metric | v39 | v38 (bar) |
|---|---|---|
| practice (direct grade) | **0.113971** | 0.115355 |
| LOO-large (lnV1 ≥ 11) | 0.236434 | **0.206819** |
| LOO-mega (lnV1 ≥ 11.5) | 0.237504 | **0.208088** |
| LOO-all | 0.475921 | 0.485578 |

The practice "gain" is an artifact: the members could not be
reconstructed exactly from prose (practice mu RMS ~0.04, LOO mu RMS
~0.27, vs v37's 0.000000), and the program fails both LOO gates by
~+0.03. The median is the bottleneck and the combine frontier is flat.
Details in `DESIGN.md`.
