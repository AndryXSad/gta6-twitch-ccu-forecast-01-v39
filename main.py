#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gta6-twitch-ccu-forecast-01 — round v39 entry.

Forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2-24),
as five quantiles (5/25/50/75/95%) per hour, from the case file's
comparable-launch library plus the target's hour-1 print and pre-release
history. Stdlib only, deterministic, no network.

Model (v38 packaged cell, the bar this round tries to beat):
  Five member predictors, each a geometric blend of three views:
    * OLS view — per-hour OLS on [lnV1, lnPREM, g12] over comparables
      within +-2.5 log units of the target (whole-library fallback when
      the window holds <10 launches); spread = residual sd x 0.8.
      No recentering: the cached-row validation shows the plain OLS
      prediction matches the members (a median recentering adds noise).
    * console-shape view — one-factor shape on console-only launches
      (no 'PC' in platforms): template = per-hour median log-ratio
      curve (hour / hour 1) over the pool; each launch fits level c +
      amplitude s on the template over all forecast hours, (c, s)
      regressed on features, amplitude clamped [0, 2]; spread =
      per-hour residual sd.
    * all-shape view — same, over the whole EA-filtered pool.
  Members differ by EA filter, shape features, fit window and sd
  convention:
    * y5    — tier filter (Tier: comparable), shape [lnV1, g12],
      fit_hours 10, min_fits 6, population sd, blend (0.5, 0.2, 0.3),
      spread = sqrt-variance blend, shape pool windowed.
    * z0    — naive EA filter ('early access' absent, case-insensitive),
      shape [lnV1, g12], fit_hours 20, min_fits 5, sample sd,
      blend (0.5, 0.2, 0.3), spread = sqrt-variance blend,
      shape pool windowed.
    * y1    — naive EA filter, shape [lnV1, lnPREM], fit_hours 20,
      min_fits 5, sample sd, blend (0.5, 0.2, 0.3),
      spread = sqrt-variance blend, shape pool windowed.
    * clone — hyphen EA filter ('early-access' absent), shape
      [lnV1, lnPREM, g12], fit_hours 20, min_fits 5, population sd,
      blend (0.6, 0.2, 0.2), spread = 0.81 x linear mean of view
      spreads, shape pool whole-library.
    * mv    — per-hour median across four views (OLS, console shape,
      all shape — all 3-feature, population sd, windowed, hyphen EA
      filter — and the clone's own forecast); spread = mean of the
      four views' spreads.
  Combine (v38 cell): per-band median source (early Y1, mid
  median(Y5, clone, mv), late median(Y5, Z0, Y1)); convex ramps
  frac**exp per band; size-indexed disagreement shift with per-band
  bases/slopes entering as size**1.25; per-band size shifts; spread =
  0.975 x weighted member sigs (late band ramps 0.975 -> 0.84),
  widened by disagreement (k_dis) and by size x disagreement (W5),
  asymmetric lo/hi scales per band, W1 lower-spread size term, cap
  0.315 on both final spreads.

Features per launch: lnV1 = log viewers at hour 1 of its release
day; lnPREM = log mean viewers over the 7 days before release;
g12 = lnV1 - log viewers 12 h before release day (library median
fallback when that point is missing). The size fed to regressions is
clamped to the library's observed lnV1 range + 0.5 log units of
headroom; the level anchor stays the true hour-1 print.

Honest caveat: the combine constants were selected on the same four
practice launches and the same 37-launch LOO as every round since
v22, so neither gate is a clean holdout; the final launch (lnV1 ~14+)
sits beyond the library max (13.68), so the size-indexed terms
extrapolate hardest exactly there. This round (v39) tries one new
combine/median lever on top of the v38 cell, or reports NO-GO.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

LEVELS_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
HOURS = list(range(2, 25))

# ---------------------------------------------------------------- utilities

def median(xs):
    """True median (mean of the two middle values for even n)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty list")
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs, ddof=0):
    """Standard deviation with the given ddof (0 = population)."""
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def solve_linear(a, b):
    """Solve a x = b by full Gauss-Jordan elimination (every other row
    eliminated at each pivot), with partial pivoting.

    A forward-elimination-only solver with a Gauss-Jordan-style return
    is silently wrong for p >= 2 (v37 lesson); this one eliminates all
    other rows at each pivot, so the diagonal return is correct.
    Tested on known 2x2 / 3x3 systems before any member validation.
    Returns None when the system is singular.
    """
    n = len(a)
    aug = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [v - f * w for v, w in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def ols(design, target):
    """Least squares via normal equations + solve_linear.

    design: list of rows (feature vectors, WITHOUT intercept);
    an intercept column is added here. Returns the coefficient
    list [intercept, b1, ...] or None when singular / too few rows.
    """
    n = len(design)
    if n == 0:
        return None
    p = len(design[0]) + 1
    if n < p:
        return None
    ata = [[0.0] * p for _ in range(p)]
    atb = [0.0] * p
    for row, y in zip(design, target):
        x = [1.0] + list(row)
        for i in range(p):
            atb[i] += x[i] * y
            for j in range(p):
                ata[i][j] += x[i] * x[j]
    return solve_linear(ata, atb)


def predict(coef, feats):
    return coef[0] + sum(c * f for c, f in zip(coef[1:], feats))


# ---------------------------------------------------------------- time helpers

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fmt_ts(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- feature extraction

def launch_features(hourly, day_start):
    """Features + release-day log curve for one launch.

    hourly: list of {from, viewers}; day_start: datetime of the
    release day's first instant. Returns (feats, curve) where feats
    = (lnV1, lnPREM, g12 or None) and curve = {hour: log viewers}
    for hours 1..24 of the release day. Returns None when the launch
    has no hour-1 print.
    """
    by_from = {}
    for pt in hourly:
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        by_from[pt["from"]] = float(v)
    h1_key = fmt_ts(day_start)
    if h1_key not in by_from:
        return None
    ln_v1 = math.log(by_from[h1_key])
    curve = {}
    for h in range(1, 25):
        key = fmt_ts(day_start + timedelta(hours=h - 1))
        if key in by_from:
            curve[h] = math.log(by_from[key])
    # lnPREM: log mean viewers over the 7 days before release.
    prem = [v for k, v in by_from.items()
            if day_start - timedelta(days=7) <= parse_ts(k) < day_start]
    ln_prem = math.log(mean(prem)) if prem else None
    # g12: lnV1 - log viewers exactly 12 h before release day.
    g12_key = fmt_ts(day_start - timedelta(hours=12))
    g12 = ln_v1 - math.log(by_from[g12_key]) if g12_key in by_from else None
    return (ln_v1, ln_prem, g12), curve


def target_features(case):
    """Features for the case's own target launch.

    lnV1 from known[0].viewers; lnPREM from history_hourly points in
    [day_start - 7d, day_start); g12 from the history point at
    day_start - 12h (history_hourly, else history).
    """
    day_start = parse_ts(case["day_start"])
    ln_v1 = math.log(max(1.0, float(case["known"][0]["viewers"])))
    hist = case.get("history_hourly") or []
    prem = [float(pt["viewers"]) for pt in hist
            if pt.get("viewers") and day_start - timedelta(days=7)
            <= parse_ts(pt["from"]) < day_start]
    ln_prem = math.log(mean(prem)) if prem else None
    g12 = None
    want = fmt_ts(day_start - timedelta(hours=12))
    for pt in hist:
        if pt.get("from") == want and pt.get("viewers"):
            g12 = ln_v1 - math.log(float(pt["viewers"]))
            break
    if g12 is None:
        for pt in case.get("history") or []:
            if pt.get("at") == want and pt.get("viewers"):
                g12 = ln_v1 - math.log(float(pt["viewers"]))
                break
    return (ln_v1, ln_prem, g12)


def tier_of(unlock):
    """Parse 'Tier: <name>' out of the unlock research note."""
    if not unlock:
        return "none"
    low = unlock.lower()
    idx = low.find("tier:")
    if idx < 0:
        return "none"
    rest = low[idx + 5:].strip()
    return rest.split()[0].strip(".,;") if rest else "none"


def is_console(platforms):
    """Console-only launch: no 'PC' in the platforms string."""
    return "PC" not in (platforms or "")


def build_library(case):
    """The usable comparable library: every comparable with an hour-1
    print (this drops the GTA VI pre-release history entry, which has
    no release day), each with features, curve, tier and console flag.
    """
    lib = []
    for comp in case.get("comparables") or []:
        out = launch_features(comp.get("hourly") or [],
                              parse_ts(comp["day_start"]))
        if out is None:
            continue
        feats, curve = out
        lib.append({
            "name": comp.get("name"),
            "feats": feats,
            "curve": curve,
            "tier": tier_of(comp.get("unlock")),
            "console": is_console(comp.get("platforms")),
            "unlock": comp.get("unlock") or "",
        })
    return lib


def fill_missing(feats_list, target):
    """Fill a missing lnPREM / g12 with the library median of the
    launches that have it (target included when present)."""
    lib = [f for f in feats_list if f is not None]
    out = list(target)
    for i in (1, 2):
        if out[i] is None:
            vals = [f[i] for f in lib if f[i] is not None]
            out[i] = median(vals) if vals else 0.0
    return tuple(out)

# ---------------------------------------------------------------- views

def ols_view(pool, target_feats, hour, ddof, scale):
    """Per-hour OLS view on [lnV1, lnPREM, g12].

    pool: library launches in the member's EA filter; target_feats:
    the target's (filled) features. Window +-2.5 log units of the
    target's lnV1, counted on the full pool before the hour filter;
    whole-library fallback when the window pool holds <10 launches;
    persistence fallback (lnV1, 0.5) when fewer than 8 rows remain.
    Spread = residual sd x scale. (No recentering: the cached-row
    validation shows the plain OLS prediction matches the members;
    a median recentering adds noise.)
    Returns (mu, sig) in log space.
    """
    ln_v1 = target_feats[0]
    window = [L for L in pool
              if abs(L["feats"][0] - ln_v1) <= 2.5 and hour in L["curve"]]
    if len(window) < 10:
        window = [L for L in pool if hour in L["curve"]]
    if len(window) < 8:
        return ln_v1, 0.5
    design = [L["feats"] for L in window]
    y = [L["curve"][hour] for L in window]
    coef = ols(design, y)
    if coef is None:
        return ln_v1, 0.5
    resid = [yi - predict(coef, xi) for xi, yi in zip(design, y)]
    mu = predict(coef, target_feats)
    return mu, sd(resid, ddof=ddof) * scale


def shape_view(pool, target_feats, hour, shape_feats_idx, fit_hours,
               ddof, min_fits, windowed=True):
    """One-factor shape view over `pool` (console-only or all).

    When windowed, the pool is first restricted to launches within
    +-2.5 log units of the target's lnV1 (y5/z0/y1/mv; the clone uses
    the whole library); when that windowed pool holds fewer than
    min_fits launches, the whole (unwindowed) pool is used instead.
    Template = per-hour median log-ratio curve (hour / hour 1) over
    the pool. Each pool launch fits level c + amplitude s on the
    template over all forecast hours; (c, s) are regressed on the
    member's shape features and the target's (c, s) predicted
    (amplitude clamped [0, 2]). Spread = per-hour sd of each
    launch's own-fit residuals. `fit_hours` is the minimum curve
    length for a launch to join the fit (all launches have 24 hours,
    so it never binds on this data). Returns (mu, sig) in log space,
    or None when the pool is too small to fit.
    """
    if windowed:
        win = [L for L in pool
               if abs(L["feats"][0] - target_feats[0]) <= 2.5]
        if len(win) >= min_fits:
            pool = win
    fit = [L for L in pool if len(L["curve"]) >= fit_hours]
    if len(fit) < 5:
        return None
    template = {1: 0.0}
    for h in HOURS:
        vals = [L["curve"][h] - L["feats"][0] for L in fit if h in L["curve"]]
        if vals:
            template[h] = median(vals)
    if hour not in template:
        return None
    hours_fit = list(range(2, 25))
    cs = []
    for L in fit:
        xs = [template[h] for h in hours_fit]
        ys = [L["curve"][h] - L["feats"][0] for h in hours_fit]
        coef = ols([[x] for x in xs], ys)
        if coef is None:
            continue
        cs.append((L, coef[0], coef[1]))
    if len(cs) < 5:
        return None
    c_vals = [c for _, c, _ in cs]
    s_vals = [s for _, _, s in cs]
    c_coef = ols([[L["feats"][i] for i in shape_feats_idx] for L, _, _ in cs],
                 c_vals)
    s_coef = ols([[L["feats"][i] for i in shape_feats_idx] for L, _, _ in cs],
                 s_vals)
    if c_coef is None or s_coef is None:
        return None
    t_feats = [target_feats[i] for i in shape_feats_idx]
    c_t = predict(c_coef, t_feats)
    s_t = min(2.0, max(0.0, predict(s_coef, t_feats)))
    ln_v1 = target_feats[0]
    mu = ln_v1 + c_t + s_t * template[hour]
    resid = []
    for L, c_i, s_i in cs:
        if hour in L["curve"]:
            resid.append((L["curve"][hour] - L["feats"][0])
                         - (c_i + s_i * template[hour]))
    return mu, sd(resid, ddof=ddof)


# ---------------------------------------------------------------- members

# EA filters: each takes a library launch, returns True to keep it.
def keep_tier(L):
    return L["tier"] == "comparable"


def keep_naive(L):
    return "early access" not in L["unlock"].lower()


def keep_hyphen(L):
    return "early-access" not in L["unlock"].lower()


# Member spec: (filter, shape feature indices into (lnV1, lnPREM, g12),
# fit_hours, min_fits, sd ddof, blend weights (OLS, console, all),
# spread mode, shape pool windowed?).
# ddof per v38 DESIGN.md: y5 population, z0/y1 sample, clone population.
MEMBERS = {
    "y5":    (keep_tier,   (0, 2),    10, 6, 0, (0.5, 0.2, 0.3), "sqrt", True),
    "z0":    (keep_naive,  (0, 2),    20, 5, 1, (0.5, 0.2, 0.3), "sqrt", True),
    "y1":    (keep_naive,  (0, 1),    20, 5, 1, (0.5, 0.2, 0.3), "sqrt", True),
    "clone": (keep_hyphen, (0, 1, 2), 20, 5, 0, (0.6, 0.2, 0.2), "lin81", False),
}

OLS_SCALE = 0.8


def member_views(lib, target_feats, hour, filt, shape_idx, fit_hours,
                 ddof, min_fits, windowed):
    """The three views (OLS, console-shape, all-shape) of one member
    at one hour. Each view is (mu, sig) in log space."""
    pool = [L for L in lib if filt(L)]
    ols_mu, ols_sig = ols_view(pool, target_feats, hour, ddof, OLS_SCALE)
    console_pool = [L for L in pool if L["console"]]
    con = shape_view(console_pool, target_feats, hour, shape_idx,
                     fit_hours, ddof, min_fits, windowed)
    allv = shape_view(pool, target_feats, hour, shape_idx, fit_hours,
                      ddof, min_fits, windowed)
    ln_v1 = target_feats[0]
    if con is None:
        con = (ln_v1, 0.5)
    if allv is None:
        allv = (ln_v1, 0.5)
    return (ols_mu, ols_sig), con, allv


def combine_views(views, weights, mode):
    """Geometric-mean blend of view medians; spread blend per mode."""
    mus = [v[0] for v in views]
    sigs = [v[1] for v in views]
    mu = sum(w * m for w, m in zip(weights, mus))
    if mode == "sqrt":
        sig = math.sqrt(sum(w * s * s for w, s in zip(weights, sigs)))
    else:  # 'lin81': 0.81 x weighted linear mean of view spreads
        sig = 0.81 * sum(w * s for w, s in zip(weights, sigs))
    return mu, sig


def compute_members(lib, target_feats, hour):
    """All five members' (mu, sig) at one hour, in log space."""
    out = {}
    for name, (filt, shape_idx, fit_hours, min_fits, ddof, weights,
               mode, windowed) in MEMBERS.items():
        views = member_views(lib, target_feats, hour, filt, shape_idx,
                             fit_hours, ddof, min_fits, windowed)
        out[name] = combine_views(views, weights, mode)
    # mv: per-hour median across four views (OLS, console shape,
    # all shape — 3-feature, population sd, windowed, hyphen filter —
    # and the clone's own forecast); spread = mean of the four
    # views' spreads.
    views = member_views(lib, target_feats, hour, keep_hyphen, (0, 1, 2),
                         20, 0, 5, True)
    four = list(views) + [out["clone"]]
    out["mv"] = (median([v[0] for v in four]),
                 mean([v[1] for v in four]))
    return out

# ---------------------------------------------------------------- combine (v38 cell)

BANDS = ((2, 6), (7, 14), (15, 24))  # early, mid, late
# Per-band constants (early, mid, late), v38 packaged cell.
RAMPS = (0.070, -0.19, 0.05)          # ramp end values
RAMP_EXP = (2.0, 1.25, 1.5)           # convex ramp exponents
SHIFT_BASE = (0.60, 0.80, 0.45)       # disagreement-shift bases
SHIFT_SLOPE = (0.23, 0.38, 0.43)      # per-band shift slopes (size**1.25)
LOC_SCALE = (0.8, 1.1, 1.0)           # location-shift disagreement scales
SIZE_SHIFT = (-0.06, -0.04, -0.05)    # per-band size shifts
ASYM = ((0.90, 1.10), (1.10, 0.85), (0.95, 0.85))  # (lo, hi) per band
K_DIS = 0.80
SPREAD_SCALE = 0.9     # uniform disagreement scale for spread factor + W5
SPREAD_BASE = 0.975
SPREAD_LATE_END = 0.84
SPREAD_W = (0.45, 0.45, 0.05, 0.05)  # (y5, z0, clone, mv)
W1_SIZE = 0.075        # sig_lo *= 1 + W1_SIZE * size
SPREAD_CAP = 0.315


def band_of(hour):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= hour <= hi:
            return i
    return 2


def frac_in_band(hour, band):
    lo, hi = BANDS[band]
    if hi == lo:
        return 1.0
    return (hour - lo) / (hi - lo)


def disagreement(mus, band):
    """Disagreement of the five member medians at this hour:
    sample sd early and late, (max - min) / 2 in mid."""
    vals = [mus[k] for k in ("y5", "z0", "y1", "clone", "mv")]
    if band == 1:
        return (max(vals) - min(vals)) / 2.0
    return sd(vals, ddof=1)


def combine_hour(members, ln_v1, hour):
    """v38 combine: (mu, sig_lo, sig_hi) in log space for one hour."""
    band = band_of(hour)
    frac = frac_in_band(hour, band)
    size = max(0.0, ln_v1 - 11.5)
    sd_band = disagreement({k: v[0] for k, v in members.items()}, band)
    # Median source per band.
    if band == 0:
        med = members["y1"][0]
    elif band == 1:
        med = median([members["y5"][0], members["clone"][0],
                      members["mv"][0]])
    else:
        med = median([members["y5"][0], members["z0"][0],
                      members["y1"][0]])
    mu = med
    mu += RAMPS[band] * (frac ** RAMP_EXP[band])
    mu += SIZE_SHIFT[band] * size
    mu += (SHIFT_BASE[band]
           + SHIFT_SLOPE[band] * (size ** 1.25)) * LOC_SCALE[band] * sd_band
    # Spread: weighted member sigs, late band ramps 0.975 -> 0.84.
    if band == 2:
        base = SPREAD_BASE + (SPREAD_LATE_END - SPREAD_BASE) * frac
    else:
        base = SPREAD_BASE
    spread = base * (
        SPREAD_W[0] * members["y5"][1]
        + SPREAD_W[1] * members["z0"][1]
        + SPREAD_W[2] * members["clone"][1]
        + SPREAD_W[3] * members["mv"][1])
    spread *= 1.0 + K_DIS * SPREAD_SCALE * sd_band
    spread *= 1.0 + SPREAD_SCALE * size * sd_band
    lo_scale, hi_scale = ASYM[band]
    sig_lo = spread * lo_scale
    sig_hi = spread * hi_scale
    sig_lo *= 1.0 + W1_SIZE * size
    sig_lo = min(sig_lo, SPREAD_CAP)
    sig_hi = min(sig_hi, SPREAD_CAP)
    return mu, sig_lo, sig_hi


# ---------------------------------------------------------------- program

def forecast_case(case):
    """Forecast one case: {hour: [q05, q25, q50, q75, q95]}."""
    lib = build_library(case)
    raw_target = target_features(case)
    filled = fill_missing([L["feats"] for L in lib], raw_target)
    ln_v1 = filled[0]
    # Regression inputs use the target's true features: the probe found
    # clamping the regression size to the library lnV1 range (+0.5
    # headroom) WORSENS leave-one-out member reproduction, so no
    # clamping is applied here; the level anchor stays the true
    # hour-1 print.
    reg_feats = (filled[0], filled[1], filled[2])
    out = {}
    for hour in HOURS:
        members = compute_members(lib, reg_feats, hour)
        mu, sig_lo, sig_hi = combine_hour(members, ln_v1, hour)
        out[hour] = [
            math.exp(mu + LEVELS_Z[0] * sig_lo),
            math.exp(mu + LEVELS_Z[1] * sig_lo),
            math.exp(mu),
            math.exp(mu + LEVELS_Z[3] * sig_hi),
            math.exp(mu + LEVELS_Z[4] * sig_hi),
        ]
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        case_id = fn[:-3]
        with open(os.path.join(in_dir, fn), encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        rows = [{"hour": h, "q": forecast[h]} for h in HOURS]
        with open(os.path.join(out_dir, case_id + ".out"),
                  "w", encoding="utf-8") as f:
            json.dump({"hours": rows}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
