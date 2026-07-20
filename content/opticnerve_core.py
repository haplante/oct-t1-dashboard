"""Shared source of truth for the OCT–T1 figures: params, stats, geometry,
resolve_view, and the Plotly builders. Imported by the Dash app, the Flask
route, and the three notebooks."""

import warnings
from functools import lru_cache
from pathlib import Path
import base64

import numpy as np
import pandas as pd
from scipy.stats import linregress
import statsmodels.formula.api as smf
from plotly.colors import sample_colorscale

warnings.simplefilter("ignore")   # silence statsmodels convergence chatter

# ---- state model -----------------------------------------------------------
DEFAULTS = {"exclude": (), "stat": "R2m", "band": "T1_mean_015",
            "mac": "All_1_3_gcc", "disc": "All_um_", "mode": "avg"}

_STATS = ("R2m", "Rm")
_BANDS = ("T1_mean_015", "T1_mean_05", "T1_mean_510", "T1_mean_1015")
_MODES = ("avg", "lme")
_ORDER = ("exclude", "stat", "band", "mac", "disc", "mode")


def parse_params(mapping):
    """Normalize a dict-like of raw query values into the 6 canonical params."""
    g = mapping.get
    exclude = g("exclude", "") or ""
    exclude = tuple(s for s in (x.strip() for x in exclude.split(",")) if s)
    pick = lambda key, allowed: (g(key) if g(key) in allowed else DEFAULTS[key])
    return {
        "exclude": exclude,
        "stat": pick("stat", _STATS),
        "band": pick("band", _BANDS),
        "mac": g("mac") or DEFAULTS["mac"],
        "disc": g("disc") or DEFAULTS["disc"],
        "mode": pick("mode", _MODES),
    }


def serialize_params(params):
    """Canonical query string, fixed key order, exclude joined by commas."""
    parts = []
    for k in _ORDER:
        v = params[k]
        v = ",".join(v) if k == "exclude" else v
        parts.append(f"{k}={v}")
    return "&".join(parts)


# ============================================================================
# CONFIGURATION  (mirrors the figure notebooks / the HTML dashboard)
# ============================================================================
_HERE = Path(__file__).resolve().parent          # works whether the app sits at the
DATA = next((p / "data" for p in (_HERE, _HERE.parent)   # repo root or in content/
             if (p / "data").exists()), _HERE / "data")
T1_SCALE = 1000.0          # T1 stored in seconds -> ms
N_SLICES, MIN_N = 46, 3    # Figure 1 profile

C_OD, C_OS = "rgb(34,139,94)", "rgb(59,130,246)"   # right eye green, left eye blue

# the four T1 bands: (column, label, colour)
T1_BANDS = [("T1_mean_015", "0–15 mm", "#2B2D42"), ("T1_mean_05", "0–5 mm", "#3B82F6"),
            ("T1_mean_510", "5–10 mm", "#228B5E"), ("T1_mean_1015", "10–15 mm", "#8B5CF6")]
T1_COLS = ["T1_mean_05", "T1_mean_510", "T1_mean_1015", "T1_mean_015"]
BAND_LABEL = {k: lbl for k, lbl, _ in T1_BANDS}

# Figure 2 default panels: (title, sector, x-axis label)
DEF_MAC, DEF_DISC = "All_1_3_gcc", "All_um_"
PANELS = [("Macula", DEF_MAC, "GCC thickness (µm)"),
          ("Optic Disc", DEF_DISC, "RNFL thickness (µm)")]

# Figure 3 metric sets (FDR is computed within each set)
DISC_METRICS = ["TS_um_", "ST_um_", "SN_um_", "NS_um_", "NI_um_", "IN_um_",
                "IT_um_", "TI_um_", "All_um_"]
MAC_METRICS = ["Center_1_gcc", "T_1_3_gcc", "S_1_3_gcc", "N_1_3_gcc", "I_1_3_gcc",
               "All_1_3_gcc", "T_3_6_gcc", "S_3_6_gcc", "N_3_6_gcc", "I_3_6_gcc",
               "All_3_6_gcc", "All_field_gcc"]

# averages table under the profile: (metric, display name)
MAC_AVG = [("All_1_3_gcc", "GCC All (1–3 mm)"), ("All_3_6_gcc", "GCC All (3–6 mm)"),
           ("All_field_gcc", "GCC All Field")]
DISC_AVG = [("All_um_", "RNFL Average")]

NAMES = {"All_um_": "RNFL Overall", "TS_um_": "RNFL TS", "ST_um_": "RNFL ST",
         "SN_um_": "RNFL SN", "NS_um_": "RNFL NS", "NI_um_": "RNFL NI",
         "IN_um_": "RNFL IN", "IT_um_": "RNFL IT", "TI_um_": "RNFL TI",
         "Center_1_gcc": "GCC Center (1 mm)",
         "N_1_3_gcc": "GCC N (1–3 mm)", "S_1_3_gcc": "GCC S (1–3 mm)",
         "T_1_3_gcc": "GCC T (1–3 mm)", "I_1_3_gcc": "GCC I (1–3 mm)",
         "N_3_6_gcc": "GCC N (3–6 mm)", "S_3_6_gcc": "GCC S (3–6 mm)",
         "T_3_6_gcc": "GCC T (3–6 mm)", "I_3_6_gcc": "GCC I (3–6 mm)",
         "All_1_3_gcc": "GCC All (1–3 mm)", "All_3_6_gcc": "GCC All (3–6 mm)",
         "All_field_gcc": "GCC All Field"}
sector_name = lambda m: NAMES.get(m, m)

# ---- Figure 3 map geometry ----
PI, LIM = np.pi, 1.32
rC, rI, rO = LIM / 6.6, LIM * 3 / 6.6, LIM * 6 / 6.6
mac_q, mac_ang = ["N", "S", "T", "I"], [k * PI / 4 for k in (-1, 1, 3, 5, 7)]
MAC_SECTORS = [dict(m="Center_1_gcc", th1=0, th2=2 * PI, ri=0, ro=rC)]
for k, q in enumerate(mac_q):
    MAC_SECTORS.append(dict(m=f"{q}_1_3_gcc", th1=mac_ang[k], th2=mac_ang[k + 1], ri=rC, ro=rI))
    MAC_SECTORS.append(dict(m=f"{q}_3_6_gcc", th1=mac_ang[k], th2=mac_ang[k + 1], ri=rI, ro=rO))
disc_short = ["NS", "SN", "ST", "TS", "TI", "IT", "IN", "NI"]
disc_order = ["NS_um_", "SN_um_", "ST_um_", "TS_um_", "TI_um_", "IT_um_", "IN_um_", "NI_um_"]
disc_th = [d * PI / 180 for d in (0, 70, 115, 150, 190, 220, 255, 305, 360)]
DISC_SECTORS = [dict(m=disc_order[k], th1=disc_th[k], th2=disc_th[k + 1], ri=0.5, ro=1.05)
                for k in range(8)]

# ============================================================================
# DATA  (read once at startup — small CSVs bundled in data/)
# ============================================================================
MERGED = pd.read_csv(DATA / "data_merged.csv")
MERGED[T1_COLS] *= T1_SCALE
PROFILE = pd.read_csv(DATA / "data_profile.csv")
SLICE_COLS = [f"T1_slice_{i:02d}" for i in range(N_SLICES)]
SUBJECTS = sorted(MERGED["MRI_ID"].unique())

_uri = lambda p: "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()
MAC_URI, DISC_URI = _uri(DATA / "macula_OD.jpg"), _uri(DATA / "disc_OD.jpg")

AX = dict(color="black", linecolor="black", showline=True, mirror=False, showgrid=False,
          zeroline=False, ticks="outside", tickcolor="black", fixedrange=True)

# ============================================================================
# STATISTICS
# ============================================================================
def bh_fdr(p):
    """Benjamini–Hochberg FDR adjustment (NaNs pass through)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    prev = 1.0
    for k in range(m - 1, -1, -1):
        prev = min(prev, p[order[k]] * m / (k + 1))
        out[order[k]] = prev
    return out


@lru_cache(maxsize=8192)
def fit(excluded, sector, band, mode):
    """Fit T1(band) ~ OCT(sector) for the active subjects.

    excluded : tuple of MRI_IDs to drop (hashable, so results are cached).
    Returns a dict of results, or None if there are too few points.
      b0,b1  intercept/slope       R2,R  (marginal) R-squared / signed R
      p      slope p-value         x,y   points to plot in Figure 2
      eye    per-point eye (LME only) — None in average mode
    """
    rows = MERGED[~MERGED["MRI_ID"].isin(excluded)]

    if mode == "avg":                       # average both eyes -> one point per subject -> OLS
        d = (rows.groupby("MRI_ID", as_index=False)[[sector, band]].mean()
                 .dropna(subset=[sector, band]))
        if len(d) < 5:
            return None
        f = linregress(d[sector], d[band])
        return dict(b0=f.intercept, b1=f.slope, R2=f.rvalue ** 2, R=f.rvalue, p=f.pvalue,
                    x=tuple(d[sector]), y=tuple(d[band]), eye=None, subj=tuple(d["MRI_ID"]))

    # mode == "lme": per-eye data, random intercept per subject
    d = rows[["MRI_ID", "Eye", sector, band]].dropna()
    if len(d) < 6 or d["MRI_ID"].nunique() < 5:
        return None
    try:
        res = smf.mixedlm(f"{band} ~ {sector}", d, groups=d["MRI_ID"]).fit(reml=True)
    except Exception:
        return None
    b0, b1 = res.fe_params["Intercept"], res.fe_params[sector]
    u = d["MRI_ID"].map(lambda g: float(res.random_effects[g].iloc[0]))   # subject intercepts
    y_marg = (d[band] - u).values                                        # marginal T1
    ybar = d[band].mean()
    yhat = b0 + b1 * d[sector]
    sst = float(((d[band] - ybar) ** 2).sum())
    ssr = float(((yhat - ybar) ** 2).sum())                             # fixed-effect variance
    R2 = ssr / sst if sst > 0 else np.nan
    R = np.sign(b1) * np.sqrt(max(R2, 0)) if not np.isnan(R2) else np.nan
    return dict(b0=b0, b1=b1, R2=R2, R=R, p=float(res.pvalues[sector]),
                x=tuple(d[sector]), y=tuple(y_marg), eye=tuple(d["Eye"]), subj=tuple(d["MRI_ID"]))


def fit_family(excluded, metrics, band, mode):
    """Fit every metric in a family against one band; return fits + FDR p-values."""
    fits = {m: fit(excluded, m, band, mode) for m in metrics}
    pf = bh_fdr([fits[m]["p"] if fits[m] else np.nan for m in metrics])
    return fits, {m: pf[i] for i, m in enumerate(metrics)}


# ---- small formatters ----
stat_val = lambda r, stat: (r["R2"] if stat == "R2m" else r["R"]) if r else np.nan
stat_lbl = lambda stat: "R²" if stat == "R2m" else "R"
fmt2 = lambda v: "—" if (v is None or np.isnan(v)) else f"{v:.2f}".replace("0.", ".").replace("-0.", "-.")


def jet(v, a=0.5, vmin=0.0):
    if v is None or np.isnan(v):
        return "rgba(153,153,153,0.5)"
    frac = float(np.clip((v - vmin) / (1 - vmin), 0, 1))
    r, g, b = sample_colorscale("Jet", [frac], colortype="tuple")[0]
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a})"


def wedge(th1, th2, ri, ro, n=60):
    th = np.linspace(th1, th2, n)
    if ri == 0 and abs((th2 - th1) - 2 * PI) < 1e-9:
        x, y = ro * np.cos(th), ro * np.sin(th)
    else:
        x = np.r_[ri * np.cos(th), ro * np.cos(th[::-1])]
        y = np.r_[ri * np.sin(th), ro * np.sin(th[::-1])]
    return np.r_[x, x[0]], np.r_[y, y[0]]


# regression lines are drawn in the notebook's band order (0-15 first)
T1_COLS_ORDER = [b[0] for b in T1_BANDS]


def resolve_view(exclude=(), stat="R2m", band="T1_mean_015",
                 mac="All_1_3_gcc", disc="All_um_", mode="avg"):
    """Single source of truth: filter data, run every fit needed to draw
    Fig 1/2/3 + the averages table for this state. Cheap thanks to fit()'s
    lru_cache; returns plain dicts so it JSON-serializes and unit-tests easily."""
    params = parse_params({"exclude": ",".join(exclude) if exclude else "",
                           "stat": stat, "band": band, "mac": mac,
                           "disc": disc, "mode": mode})
    excluded = params["exclude"]
    included = [s for s in SUBJECTS if s not in excluded]

    # Fig 3 / averages-table fits for the SELECTED band (used to colour wedges)
    mac_fits, mac_pf = fit_family(excluded, MAC_METRICS, params["band"], params["mode"])
    disc_fits, disc_pf = fit_family(excluded, DISC_METRICS, params["band"], params["mode"])

    # Fig 2 panels fit every band against the selected sector; FDR within panel
    panel_fits = {}
    for key, sector in (("mac", params["mac"]), ("disc", params["disc"])):
        fits = [fit(excluded, sector, b, params["mode"]) for b in T1_COLS_ORDER]
        pf = bh_fdr([f["p"] if f else float("nan") for f in fits])
        panel_fits[key] = {"sector": sector,
                           "fits": {b: f for b, f in zip(T1_COLS_ORDER, fits)},
                           "pf": {b: pf[i] for i, b in enumerate(T1_COLS_ORDER)}}

    # averages table needs FDR per family per band
    avg_pf = {}
    for b, _, _ in T1_BANDS:
        _, m_pf = fit_family(excluded, MAC_METRICS, b, params["mode"])
        _, d_pf = fit_family(excluded, DISC_METRICS, b, params["mode"])
        avg_pf[b] = {"mac": m_pf, "disc": d_pf}

    return {
        "params": params, "excluded": excluded,
        "subjects": included, "n": len(included),
        "stat": params["stat"], "band": params["band"], "mode": params["mode"],
        "mac": params["mac"], "disc": params["disc"],
        "mac_fits": mac_fits, "mac_pf": mac_pf,
        "disc_fits": disc_fits, "disc_pf": disc_pf,
        "panel_fits": panel_fits, "avg_pf": avg_pf,
    }
