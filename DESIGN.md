# DESIGN.md — gta6-twitch-ccu-forecast-01 v39

## What this program does

Reads one case (comparable-launch library + target hour-1 print +
pre-release history) and writes five quantiles (5/25/50/75/95%) for
hours 2–24 of the release day. Score = pinball loss on the log scale
(lower is better). Python 3.13, stdlib only, deterministic, offline.

## Members

Five members, each a weighted blend of three views:

- **OLS view** — per-hour OLS on `[lnV1, lnPREM, g12]` over comparables
  within ±2.5 log units of the target's lnV1; whole-library fallback
  when the window holds <10 launches; persistence fallback `(lnV1, 0.5)`
  when <8 rows remain. Spread = residual sd × 0.8. No recentering.
- **Shape view** — per-hour median log-ratio template (hour / hour 1)
  over a pool; each pool launch fits level `c` + amplitude `s` on the
  template over all hours 2–24; `(c, s)` are regressed on the member's
  shape features and the target's `(c, s)` predicted (amplitude clamped
  `[0, 2]`). Spread = per-hour sd of each launch's own-fit residuals.
  When the windowed pool holds fewer than `min_fits` launches, the
  whole (unwindowed) pool is used instead.
- **mv** — per-hour median across four views (OLS, console shape,
  all shape with 3 features, and the clone's own forecast); spread =
  mean of the four views' spreads.

| member | EA filter | shape features | fit_hours | min_fits | sd ddof | weights (OLS, console, all) | spread | shape pool |
|---|---|---|---|---|---|---|---|---|
| y5 | tier `comparable` | [lnV1, g12] | 10 | 6 | population | (0.5, 0.2, 0.3) | sqrt-variance | windowed |
| z0 | naive (no `early access`) | [lnV1, g12] | 20 | 5 | sample | (0.5, 0.2, 0.3) | sqrt-variance | windowed |
| y1 | naive | [lnV1, lnPREM] | 20 | 5 | sample | (0.5, 0.2, 0.3) | sqrt-variance | windowed |
| clone | hyphen (no `early-access`) | [lnV1, lnPREM, g12] | 20 | 5 | population | (0.6, 0.2, 0.2) | 0.81 × linear mean | whole library |
| mv | hyphen, 3-feature shape | [lnV1, lnPREM, g12] | 20 | 5 | population | — | mean of 4 view spreads | windowed |

Features: `lnV1` = log hour-1 viewers; `lnPREM` = log of the pooled
mean of hourly viewers in `[day_start−7d, day_start)`; `g12` =
`lnV1 − log viewers` at `day_start−12h` (library-median fallback when
missing). `size = max(0, lnV1 − 11.5)`.

## Combine — the v38 packaged cell (unchanged this round)

Bands: early hours 2–6, mid 7–14, late 15–24. Median sources: early Y1,
mid `median(Y5, clone, mv)`, late `median(Y5, Z0, Y1)`. Ramps
(+0.070 / −0.19 / +0.05) with per-band exponents (2.0 / 1.25 / 1.5).
Size shifts (−0.06 / −0.04 / −0.05). Location shift
`(base + slope·size**1.25)·loc_scale·sd` with bases (0.60 / 0.80 / 0.45),
slopes (0.23 / 0.38 / 0.43), loc scales (0.8 / 1.1 / 1.0). Disagreement
`sd` = sample sd of the five member mus in early/late bands,
`(max−min)/2` in the mid band. Spread =
`0.975·(0.45·sigY5 + 0.45·sigZ0 + 0.05·sigClone + 0.05·sigMV)`, with the
late band's base ramping 0.975 → 0.84 (replacing the 0.975 factor),
`×(1 + 0.8·0.9·sd)`, `×(1 + 0.9·size·sd)`, asymmetric factors
(0.90,1.10)/(1.10,0.85)/(0.95,0.85), `sig_lo ×(1 + 0.075·size)`,
cap 0.315 on the final sigmas. Quantile z-values are the standard
normal quantiles for 0.05/0.25/0.5/0.75/0.95.

## Result — NO-GO

| metric | v39 | v38 (bar) |
|---|---|---|
| practice (direct grade, exact RULES scorer) | 0.113971 (p00 0.104073 / p01 0.121572 / p02 0.073955 / p03 0.156284) | 0.115355 |
| LOO-large (lnV1 ≥ 11) | 0.236434 | 0.206819 |
| LOO-mega (lnV1 ≥ 11.5) | 0.237504 | 0.208088 |
| LOO-all | 0.475921 | 0.485578 |

## Honest caveats

1. **The practice "gain" is not a real improvement.** The members
   could not be reconstructed exactly from the prose specification:
   practice mu RMS vs the cached member rows is ~0.04 (max ~0.09–0.14)
   and LOO mu RMS is ~0.27–0.31 (clone 0.12), versus v37's 0.000000
   on practice AND LOO. The program's practice score (0.113971) is
   better than v38's (0.115355) only because the member mismatch flips
   the combine's median-of-3 source per band — it does not come from a
   verified combine improvement.
2. **Both LOO gates fail by ~+0.03.** The LOO gates exist precisely to
   catch this: a program that looks better on the 4 practice cases but
   generalizes worse. v39 is such a program. It was not submitted.
3. **Where the reconstruction breaks.** The discrepancy is
   concentrated in leave-one-out edge cases: the smallest launches
   (Silksong lnV1 5.1, Kingdom Come 6.03, HELLDIVERS 2 6.04, Marathon
   7.24 — windowed shape pools of 0–4 launches) and the biggest
   (Cyberpunk 2077 / ELDEN RING lnV1 ~13.7 — the OLS window holds 9
   launches, triggering the whole-library fallback, whose prediction
   is ~1.4 log too low at late hours). Tested and rejected as the
   missing detail: OLS fallback whole/keep/nearest-10/expand and
   thresholds <3…<10; OLS features shared-3/subset/ratio; OLS target
   abs/ratio; recentering none/median/mean; fit modes all/first/
   from1/all1; template median/mean; spread sqrt/lin81; nested vs
   flat weights; shape-small whole/expand; console-fallback to ALL;
   regression-input clamping none/[min, max+0.5]/[min−0.5, max+0.5].
   The remaining gap is a detail of the original harness that the
   prose does not pin down; recovering it would require reading that
   harness's code, which this round's authorship rules forbid.
4. **The frontier is flat.** v30→v38 gains were 0.118710 → 0.115355
   over eight rounds, each step noise-level and increasingly fitted to
   the 4 practice cases. The median is the bottleneck (best
   median-only practice ~0.177); per-hour OLS residual sd is 0.35–0.45
   log. 0.100 is practically unreachable honestly.
5. **`size**1.25` extrapolates hardest at the final** (size ~2.5+ for
   GTA VI's expected scale) — do not chase practice numbers by
   pushing it further without LOO evidence.
6. **Solver.** `_solve_linear` is full Gauss-Jordan (eliminates all
   other rows at each pivot) and was tested on known 2×2 and 3×3
   systems before any member validation (lesson of v37's silent
   forward-elimination-only bug).
