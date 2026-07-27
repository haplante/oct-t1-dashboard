"""Shared source of truth for the OCT–T1 figures: params, stats, geometry,
resolve_view, and the Plotly builders. Imported by the Dash app, the Flask
route, and the three notebooks."""

import base64
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
import statsmodels.formula.api as smf
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIGURATION  (mirrors the figure notebooks / the HTML dashboard)
# ============================================================================
_HERE = Path(__file__).resolve().parent          # works whether the app sits at the
DATA = next((p / "data" for p in (_HERE, _HERE.parent)   # repo root or in content/
             if (p / "data").exists()), _HERE / "data")
T1_SCALE = 1000.0          # T1 stored in seconds -> ms
N_SLICES, MIN_N = 46, 3    # Figure 1 profile

C_OD, C_OS = "rgb(34,139,94)", "rgb(59,130,246)"   # right eye green, left eye blue

# the four T1 bands: (column, label, colour). This order is also the order
# Figure 2 draws its regression lines in.
T1_BANDS = [("T1_mean_015", "0–15 mm", "#2B2D42"), ("T1_mean_05", "0–5 mm", "#3B82F6"),
            ("T1_mean_510", "5–10 mm", "#228B5E"), ("T1_mean_1015", "10–15 mm", "#8B5CF6")]
T1_COLS = [k for k, _, _ in T1_BANDS]
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

# Exclusion is always resolved to EYE level; a bare subject token is shorthand
# for "both of its eyes". Keeping both spellings in `exclude` lets the URL
# round-trip the sidebar's two checklists (subject-level and eye-level) exactly.
EYE_SEP = "."
EYES_OF = {s: tuple(sorted(g)) for s, g in MERGED.groupby("MRI_ID")["Eye"]}
ALL_EYES = tuple(f"{s}{EYE_SEP}{e}" for s in SUBJECTS for e in EYES_OF[s])
VALID_EXCLUDE = frozenset(SUBJECTS) | frozenset(ALL_EYES)


def split_excluded(excluded):
    """-> (subjects fully out, set of excluded 'SUBJ.EYE' tokens)."""
    eyes = {t for t in excluded if EYE_SEP in t}
    for s in excluded:
        if EYE_SEP not in s:
            eyes.update(f"{s}{EYE_SEP}{e}" for e in EYES_OF[s])
    gone = {s for s in SUBJECTS if all(f"{s}{EYE_SEP}{e}" in eyes for e in EYES_OF[s])}
    return gone, eyes


# ============================================================================
# STATE MODEL — the 6 params that fully determine every figure
# ============================================================================
# `band` is an ORDERED, non-empty tuple: Figure 2 overlays every band in it, and
# band[0] — the primary — is the one Figure 3, the reference line and the
# averages table follow. Order is selection order, so it is NOT sorted (unlike
# `exclude`, where order carries no meaning).
DEFAULTS = {"exclude": (), "stat": "R2m", "band": ("T1_mean_015",),
            "mac": "All_1_3_gcc", "disc": "All_um_", "mode": "avg"}
PARAM_ORDER = ("exclude", "stat", "band", "mac", "disc", "mode")
_ALLOWED = {"stat": ("R2m", "Rm"), "mode": ("avg", "lme"),
            "mac": MAC_METRICS, "disc": DISC_METRICS}


def _ordered(raw, valid):
    """Comma list -> tuple of the known entries, first-seen order, no repeats."""
    out = []
    for t in (s.strip() for s in (raw or "").split(",")):
        if t in valid and t not in out:
            out.append(t)
    return tuple(out)


def parse_params(mapping):
    """Normalize a dict-like of raw query values into the 6 canonical params.

    Every value is clamped to a known one, so a hand-edited URL can never reach
    fit()/groupby (which would KeyError -> HTTP 500 on the public route).
    `exclude` takes a bare "sub-0200" (whole subject) or a dotted "sub-0200.OD"
    (that eye only); it is sorted so it makes a stable lru_cache key. `band` is a
    comma list whose order is meaning (band[0] is the primary), so it is kept as
    given and only falls back to the default when nothing valid survives.
    """
    g = mapping.get
    tokens = (t.strip() for t in (g("exclude", "") or "").split(","))
    params = {"exclude": tuple(sorted({t for t in tokens if t in VALID_EXCLUDE}))}
    params["band"] = _ordered(g("band", ""), T1_COLS) or DEFAULTS["band"]
    for key, allowed in _ALLOWED.items():
        params[key] = g(key) if g(key) in allowed else DEFAULTS[key]
    return {k: params[k] for k in PARAM_ORDER}


def serialize_params(params):
    """Canonical query string, fixed key order, tuple params joined by commas."""
    return "&".join(f"{k}={','.join(v) if isinstance(v, tuple) else v}"
                    for k, v in ((k, params[k]) for k in PARAM_ORDER))


_uri = lambda p: "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()
MAC_URI, DISC_URI = _uri(DATA / "macula_OD.jpg"), _uri(DATA / "disc_OD.jpg")

AX = dict(color="black", linecolor="black", showline=True, mirror=False, showgrid=False,
          zeroline=False, ticks="outside", tickcolor="black", fixedrange=True)

# Every figure draws on a FIXED pixel canvas — never autosize/responsive. Every
# consumer (dashboard, /figure/<id> pages, paper.md) scales the drawn result with
# a CSS transform instead, so Plotly measures once and never redraws on resize.
FIG_SIZE = {"fig1": (555, 402), "fig2": (638, 285), "fig3": (638, 285)}
_canvas = lambda figid: dict(zip(("autosize", "width", "height"), (False,) + FIG_SIZE[figid]))

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

    excluded : tuple of exclusion tokens (see parse_params), hashable so results
    are cached. Returns a dict of results, or None if there are too few points.
      b0,b1  intercept/slope       R2,R  (marginal) R-squared / signed R
      p      slope p-value         x,y   points to plot in Figure 2
      eye    per-point eye (LME only) — None in average mode
      gx,gy,gsubj,geye  the EXCLUDED points, so Figure 2 can still draw them
                        hollow (click to put them back)
    """
    gone, out_eyes = split_excluded(excluded)
    keep = ~(MERGED["MRI_ID"] + EYE_SEP + MERGED["Eye"]).isin(out_eyes)
    rows, dropped = MERGED[keep], MERGED[~keep]

    if mode == "avg":                       # average both eyes -> one point per subject -> OLS
        d = (rows.groupby("MRI_ID", as_index=False)[[sector, band]].mean()
                 .dropna(subset=[sector, band]))
        if len(d) < 5:
            return None
        # ghosts: only subjects that lost BOTH eyes (a subject that lost one eye
        # is still plotted, just averaged over the eye that is left)
        g = (dropped[dropped["MRI_ID"].isin(gone)]
             .groupby("MRI_ID", as_index=False)[[sector, band]].mean()
             .dropna(subset=[sector, band]))
        f = linregress(d[sector], d[band])
        return dict(b0=f.intercept, b1=f.slope, R2=f.rvalue ** 2, R=f.rvalue, p=f.pvalue,
                    x=tuple(d[sector]), y=tuple(d[band]), eye=None, subj=tuple(d["MRI_ID"]),
                    gx=tuple(g[sector]), gy=tuple(g[band]), gsubj=tuple(g["MRI_ID"]), geye=None)

    # mode == "lme": per-eye data, random intercept per subject
    d = rows[["MRI_ID", "Eye", sector, band]].dropna()
    if len(d) < 6 or d["MRI_ID"].nunique() < 5:
        return None
    try:
        with warnings.catch_warnings():          # statsmodels convergence chatter
            warnings.simplefilter("ignore")
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
    # ghosts: every dropped eye, on the same marginal scale. A dropped eye whose
    # subject is still in the model borrows that subject's intercept; a fully
    # dropped subject has none, so it falls back to the population mean (u = 0).
    gd = dropped[["MRI_ID", "Eye", sector, band]].dropna()
    u_map = {k: float(v.iloc[0]) for k, v in res.random_effects.items()}
    gy = gd[band] - gd["MRI_ID"].map(u_map).fillna(0.0)
    return dict(b0=b0, b1=b1, R2=R2, R=R, p=float(res.pvalues[sector]),
                x=tuple(d[sector]), y=tuple(y_marg), eye=tuple(d["Eye"]), subj=tuple(d["MRI_ID"]),
                gx=tuple(gd[sector]), gy=tuple(gy), gsubj=tuple(gd["MRI_ID"]),
                geye=tuple(gd["Eye"]))


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


# ============================================================================
# FIGURE 1 — T1 profile along the optic nerve
# ============================================================================
def build_fig1(view):
    _, out_eyes = split_excluded(view["excluded"])
    prof = PROFILE[~(PROFILE["MRI_ID"] + EYE_SEP + PROFILE["Eye"]).isin(out_eyes)]
    mm = np.arange(N_SLICES) + 0.5
    fig = go.Figure()
    for eye, col in [("OD", C_OD), ("OS", C_OS)]:
        M = prof.loc[prof.Eye == eye, SLICE_COLS].to_numpy() * T1_SCALE
        if M.size == 0:
            continue
        faint = col.replace("rgb", "rgba").replace(")", ",0.2)")
        for row in M:
            fig.add_scatter(x=mm, y=row, mode="lines", line=dict(color=faint, width=0.8),
                            hoverinfo="skip", showlegend=False, connectgaps=False)
        xs, mean, sd = [], [], []
        for j in range(N_SLICES):
            v = M[:, j]; v = v[~np.isnan(v)]
            if len(v) < MIN_N:
                continue
            xs.append(mm[j]); mean.append(v.mean()); sd.append(v.std(ddof=1))
        od = eye == "OD"
        fig.add_scatter(x=xs, y=mean, mode="lines+markers", name="OD (Right)" if od else "OS (Left)",
            line=dict(color=col, width=2),
            marker=dict(size=10, color="white" if od else col,
                        line=dict(color=col if od else "white", width=1.5)),
            error_y=dict(type="data", array=sd, visible=True, color=col, thickness=1.2, width=4),
            hovertemplate=f"mm %{{x}}<br>T₁ %{{y:.0f}} ms<extra>{eye}</extra>")
    fig.update_layout(
        **_canvas("fig1"), paper_bgcolor="white", plot_bgcolor="white", dragmode=False,
        font=dict(color="black", family="DejaVu Sans, Arial, sans-serif", size=14),
        title=dict(text="T₁ as a function of position along the ON", x=0.5, xanchor="center"),
        margin=dict(l=70, r=10, t=45, b=45),
        xaxis=dict(**AX, title=dict(text="Position along ON (mm)", standoff=5), range=[0, 15]),
        yaxis=dict(**AX, title=dict(text="T₁ (ms)"), range=[500, 1800]),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0)"))
    return fig


# ============================================================================
# FIGURE 2 — correlation with 4 regression lines per panel
# ============================================================================
F2_LABEL = dict(showarrow=False, font=dict(size=10),
                bgcolor="rgba(255,255,255,0.7)", borderpad=1)

# Label placement geometry, derived from the fig2 canvas and the subplot layout
# in build_fig2: 638 wide - 70 left margin - 10 right = 558 px of plot area, and
# horizontal_spacing=0.12 leaves each of the two columns (1 - 0.12)/2 = 0.44 of
# it; 285 tall - 40 top - 45 bottom = 200 px.
F2_PANEL_PX = (0.44 * (FIG_SIZE["fig2"][0] - 80), FIG_SIZE["fig2"][1] - 85)
# Character width is a text metric we approximate rather than measure. Erring
# wide only buys clearance — raise it if a label ever clips a point.
F2_CHAR_PX = 6.0
F2_LINE_PX = 15.0      # label height at font size 10, plus borderpad
F2_MARKER_PX = 7.0     # marker radius + outline: the clearance a point needs
F2_LABEL_DX, F2_LABEL_DY = 10, 8       # nudge the label clear of its own line


def _padded_range(vals, pad=0.05):
    lo, hi = float(min(vals)), float(max(vals))
    if hi == lo:
        hi = lo + 1.0
    m = (hi - lo) * pad
    return [lo - m, hi + m]


def f2_label_box(text, x, y, xr, yr):
    """The (x0, y0, x1, y1) a label occupies, in axis fractions.

    x, y is the annotation anchor in data coords, drawn xanchor=left,
    yanchor=bottom — so the box grows right and up from it.
    """
    pw, ph = F2_PANEL_PX
    u = (x - xr[0]) / (xr[1] - xr[0])
    v = (y - yr[0]) / (yr[1] - yr[0])
    return (u, v, u + (len(text) * F2_CHAR_PX + 2) / pw, v + F2_LINE_PX / ph)


def _place_labels(labels, points, xr, yr):
    """Position each Fig 2 line label so it clears the data and its neighbours.

    labels : [(text, (x0, y0), (x1, y1))] in data coords, most important first
    points : [(x, y)] every visible marker in the panel
    xr, yr : the panel's axis ranges

    Each label walks along its own regression line and takes the first slot
    clear of every point, every already-placed label, and the panel edges. If
    nothing is clear it takes the least-crowded slot, so a label is never
    dropped. Returns one (x, y) anchor per label, in the input order.
    """
    pw, ph = F2_PANEL_PX
    xspan, yspan = xr[1] - xr[0], yr[1] - yr[0]
    mx, my = F2_MARKER_PX / pw, F2_MARKER_PX / ph
    dx, dy = F2_LABEL_DX / pw, F2_LABEL_DY / ph
    pts = [((px - xr[0]) / xspan, (py - yr[0]) / yspan) for px, py in points]

    placed, out = [], []
    for text, (x0, y0), (x1, y1) in labels:
        w = (len(text) * F2_CHAR_PX + 2) / pw
        h = F2_LINE_PX / ph
        best, best_cost = None, None
        # Candidates: walk the line at t in [0.02, 0.95], try the box growing
        # right or left of the anchor, above or below it, and — if the label's
        # own line is too crowded to find a clear slot nearby — step further
        # off the line in multiples of F2_LABEL_DY. Cheapest options first so
        # the early-exit on cost == 0 still fires fast in the common case.
        for step in (1, 2, 3):
            for t in np.linspace(0.02, 0.95, 36):
                cu = (x0 + t * (x1 - x0) - xr[0]) / xspan
                cv = (y0 + t * (y1 - y0) - yr[0]) / yspan
                for above in (True, False):
                    voff = step * dy
                    by = cv + voff if above else cv - voff - h
                    for leftward in (False, True):
                        bx = cu - w - dx if leftward else cu + dx
                        box = (bx, by, bx + w, by + h)
                        off_panel = box[0] < 0 or box[2] > 1 or box[1] < 0 or box[3] > 1
                        cost = sum(1 for pu, pv in pts
                                   if box[0] - mx <= pu <= box[2] + mx
                                   and box[1] - my <= pv <= box[3] + my)
                        cost += 10 * sum(1 for q in placed if not (
                            box[2] + 0.005 <= q[0] or q[2] + 0.005 <= box[0]
                            or box[3] + 0.005 <= q[1] or q[3] + 0.005 <= box[1]))
                        if off_panel:              # last resort only: never beats an on-panel slot
                            cost += 1_000_000
                        if best_cost is None or cost < best_cost:
                            best, best_cost = box, cost
                        if cost == 0:
                            break
                    if best_cost == 0:
                        break
                if best_cost == 0:
                    break
            if best_cost == 0:
                break
        if best is None:                      # no candidate was ever tried (unreachable in practice)
            best = (0.0, 0.0, w, h)
        elif best[0] < 0 or best[2] > 1 or best[1] < 0 or best[3] > 1:
            # every candidate fell off-panel: keep the closest-to-its-line one and
            # clamp it into the panel with a pure translation (box < panel, so it
            # always fits) instead of cornering it away from its own line.
            bx0, by0, bx1, by1 = best
            sx = -bx0 if bx0 < 0 else (1 - bx1 if bx1 > 1 else 0.0)
            sy = -by0 if by0 < 0 else (1 - by1 if by1 > 1 else 0.0)
            best = (bx0 + sx, by0 + sy, bx1 + sx, by1 + sy)
        placed.append(best)
        out.append((xr[0] + best[0] * xspan, yr[0] + best[1] * yspan))
    return out


def _cdata(subj, eye, ghost):
    """Per-point customdata rows: [label, exclusion token, ghost flag].

    The token is what a Figure 2 click toggles: the bare subject in average mode
    (one point = one subject) and 'SUBJ.EYE' in LME mode (one point = one eye).
    """
    subj = np.asarray(subj, dtype=object)
    tok = (subj if eye is None else
           np.array([f"{s}{EYE_SEP}{e}" for s, e in zip(subj, eye)], dtype=object))
    return np.stack([subj, tok, np.full(len(subj), int(ghost), dtype=object)], axis=-1)


def build_fig2(view):
    mode, stat, sel_band = view["mode"], view["stat"], view["band"]
    y_title = ("marginal ON T₁ (ms)" if mode == "lme" else "ON T₁ (ms)")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=[f"Macula – {sector_name(view['mac'])}",
                                        f"Optic Disc – {sector_name(view['disc'])}"])
    for col, (title, xlab, sector, key) in enumerate(
            [(PANELS[0][0], PANELS[0][2], view["mac"], "mac"),
             (PANELS[1][0], PANELS[1][2], view["disc"], "disc")], start=1):
        leg = "legend" if col == 1 else "legend2"
        panel = view["panel_fits"][key]
        fits = [panel["fits"][b] for b in T1_COLS]
        pf = [panel["pf"][b] for b in T1_COLS]
        for (band, lbl, c), r, q in zip(T1_BANDS, fits, pf):
            if not r:
                continue
            vis = True if band == sel_band else "legendonly"
            grp = f"{col}-{band}"
            x, y = np.array(r["x"]), np.array(r["y"])
            cd = _cdata(r["subj"], r["eye"], False)
            hov = lambda side="", tail="click → remove": (
                f"<b>%{{customdata[0]}}</b>{side} · {lbl}<br>"
                f"{sector_name(sector)} = %{{x:.2f}}"
                f"<br>T₁ = %{{y:.0f}} ms<br><i>({tail})</i><extra></extra>")
            if r["eye"] is None:                       # average mode: one marker per subject
                fig.add_scatter(x=x, y=y, mode="markers", legend=leg, legendgroup=grp,
                    showlegend=False, visible=vis, row=1, col=col, customdata=cd,
                    marker=dict(color=c, size=10, line=dict(color="white", width=1.2)),
                    hovertemplate=hov(), hoverlabel=dict(bgcolor="#222", font=dict(color="#fff")))
            else:                                      # LME mode: OD filled, OS open
                eye = np.array(r["eye"])
                for e in ("OD", "OS"):
                    sel_e = eye == e
                    marker = (dict(color="white", size=10, line=dict(color=c, width=1.5)) if e == "OD"
                              else dict(color=c, size=10, line=dict(width=0)))
                    fig.add_scatter(x=x[sel_e], y=y[sel_e], mode="markers", legend=leg,
                        legendgroup=grp, showlegend=False, visible=vis, row=1, col=col,
                        customdata=cd[sel_e], marker=marker,
                        hovertemplate=hov(f" · {e}"), hoverlabel=dict(bgcolor="#222", font=dict(color="#fff")))
            # excluded points: grey and hollow, outside the fit, click to restore
            if r["gx"]:
                gcd = _cdata(r["gsubj"], r["geye"], True)
                fig.add_scatter(x=np.array(r["gx"]), y=np.array(r["gy"]), mode="markers",
                    legend=leg, legendgroup=grp, showlegend=False, visible=vis, row=1, col=col,
                    customdata=gcd, opacity=0.55,
                    marker=dict(color="rgba(0,0,0,0)", size=10, line=dict(color="#888", width=1.5)),
                    hovertemplate=hov(" · excluded", "click → put back"),
                    hoverlabel=dict(bgcolor="#222", font=dict(color="#fff")))
            xs = np.array([x.min(), x.max()])
            ys = r["b0"] + r["b1"] * xs
            star = " *" if (not np.isnan(q) and q < 0.05) else ""
            name = f"{lbl.replace(' mm','')} ({stat_lbl(stat)}={fmt2(stat_val(r, stat))}){star}"
            fig.add_scatter(x=xs, y=ys, mode="lines", legend=leg,
                legendgroup=grp, visible=vis, row=1, col=col, line=dict(color=c, width=2),
                hoverinfo="skip", name=name)
            if vis is True:
                fig.add_annotation(x=xs[0], y=ys[0], row=1, col=col, text=name,
                    xanchor="left", yanchor="bottom", xshift=F2_LABEL_DX, yshift=F2_LABEL_DY,
                    **{**F2_LABEL, "font": dict(size=10, color=c)})
        # grey dotted reference: default sector at the selected band (only when a
        # non-default sector is selected) — lets you compare against the baseline.
        ref = panel.get("ref")
        if ref:
            rx = np.array([min(ref["x"]), max(ref["x"])])
            ry = ref["b0"] + ref["b1"] * rx
            rname = (f"{BAND_LABEL[sel_band].replace(' mm','')} "
                     f"{sector_name(panel['default_sector']).split(' (')[0]}")
            fig.add_scatter(x=rx, y=ry, mode="lines", legend=leg,
                row=1, col=col, line=dict(color="#999", width=1.5, dash="dot"),
                hoverinfo="skip", name=rname)
            # labelled at the RIGHT end so it cannot land on the band label above
            fig.add_annotation(x=rx[1], y=ry[1], row=1, col=col, text=rname,
                xanchor="right", yanchor="top", xshift=-F2_LABEL_DX, yshift=-F2_LABEL_DY,
                **{**F2_LABEL, "font": dict(size=10, color="#999")})
        fig.update_xaxes(title=dict(text=xlab, standoff=5), row=1, col=col, **AX)
        fig.update_yaxes(title=dict(text=y_title), row=1, col=col, **AX)

    fig.update_layout(**_canvas("fig2"), paper_bgcolor="white", plot_bgcolor="white",
        dragmode=False, showlegend=False,
        font=dict(color="black", family="DejaVu Sans, Arial, sans-serif", size=13),
        margin=dict(l=70, r=10, t=40, b=45))
    for a in fig.layout.annotations[:2]:      # the two subplot titles only
        a.font.size = 15
    return fig


# ============================================================================
# FIGURE 3 — OCT sector maps
# ============================================================================
def build_fig3(view):
    mode, stat, sel_band = view["mode"], view["stat"], view["band"]
    vmin = -1 if stat == "Rm" else 0
    band_lbl = BAND_LABEL[sel_band]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0)
    ann = []

    def draw(sectors, axis, image, fits, pf):
        xa, ya = ("x", "y") if axis == 1 else ("x2", "y2")
        fig.add_layout_image(dict(source=image, xref=xa, yref=ya, x=-LIM, y=LIM,
            sizex=2 * LIM, sizey=2 * LIM, xanchor="left", yanchor="top",
            sizing="stretch", layer="below"))
        for s in sectors:
            r = fits[s["m"]]
            val = stat_val(r, stat)
            sig = "*" if (r and not np.isnan(pf[s["m"]]) and pf[s["m"]] < 0.05) else ""
            x, y = wedge(s["th1"], s["th2"], s["ri"], s["ro"])
            fig.add_scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=jet(val, vmin=vmin),
                line=dict(color="white", width=2), hoveron="fills", showlegend=False,
                row=1, col=axis, customdata=[s["m"]] * len(x), meta=s["m"], hoverinfo="text",
                text=f"{sector_name(s['m'])} — {stat_lbl(stat)} = {fmt2(val)}{sig}"
                     f"<br><i>(click → use in Figure 2)</i>",
                hoverlabel=dict(bgcolor="#222", font=dict(color="#fff")))
            rm = 0 if s["ri"] == 0 else (s["ri"] + s["ro"]) / 2
            tm = (s["th1"] + s["th2"]) / 2
            ann.append(dict(x=rm * np.cos(tm) * 0.95, y=rm * np.sin(tm) * 0.95, xref=xa, yref=ya,
                text="" if np.isnan(val) else f"{fmt2(val)}{sig}", showarrow=False,
                font=dict(color="white", size=15), bgcolor="rgba(0,0,0,0.4)", borderpad=2.5))
        return xa, ya

    xa, ya = draw(MAC_SECTORS, 1, MAC_URI, view["mac_fits"], view["mac_pf"])
    xa2, ya2 = draw(DISC_SECTORS, 2, DISC_URI, view["disc_fits"], view["disc_pf"])
    for k, lab in enumerate(mac_q):
        tm = (mac_ang[k] + mac_ang[k + 1]) / 2
        ann.append(dict(x=1.1 * np.cos(tm), y=1.1 * np.sin(tm), xref=xa, yref=ya,
                        text=f"<b>{lab}</b>", showarrow=False, font=dict(color="white", size=15)))
    for k, lab in enumerate(disc_short):
        tm = (disc_th[k] + disc_th[k + 1]) / 2
        ann.append(dict(x=1.2 * np.cos(tm), y=1.2 * np.sin(tm), xref=xa2, yref=ya2,
                        text=f"<b>{lab}</b>", showarrow=False, font=dict(color="white", size=15)))
    ann += [
        dict(text=f"Macula – T₁ ({band_lbl})", xref="paper", yref="paper", x=0.25, y=1.09,
             xanchor="center", showarrow=False, font=dict(color="black", size=15)),
        dict(text=f"Optic Disc – T₁ ({band_lbl})", xref="paper", yref="paper", x=0.75, y=1.09,
             xanchor="center", showarrow=False, font=dict(color="black", size=15)),
        dict(text="* p<sub>FDR</sub> < 0.05", xref="paper", yref="paper", x=0.885, y=0.04,
             xanchor="center", showarrow=False, font=dict(color="white", size=12),
             bgcolor="rgba(0,0,0,0.4)", borderpad=2),
    ]
    fig.add_scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale="Jet", cmin=vmin, cmax=1, color=[vmin], size=0.1, showscale=True,
            colorbar=dict(title=dict(text=stat_lbl(stat), side="top"), x=0.98, len=0.85,
                          thickness=12, tickfont=dict(color="black"))))
    axb = dict(range=[-LIM, LIM], visible=False, fixedrange=True)
    fig.update_layout(**_canvas("fig3"), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black", family="DejaVu Sans, Arial, sans-serif"), dragmode=False,
        margin=dict(l=0, r=0, t=30, b=15), showlegend=False, annotations=ann,
        xaxis=dict(**axb, domain=[0, 0.5], scaleanchor="y", scaleratio=1), yaxis=dict(**axb, domain=[0, 1]),
        xaxis2=dict(**axb, domain=[0.5, 1], scaleanchor="y2", scaleratio=1), yaxis2=dict(**axb, domain=[0, 1]))
    return fig


def resolve_view(exclude=(), stat="R2m", band=("T1_mean_015",),
                 mac="All_1_3_gcc", disc="All_um_", mode="avg"):
    """Single source of truth: filter data, run every fit needed to draw
    Fig 1/2/3 + the averages table for this state. Cheap thanks to fit()'s
    lru_cache; returns plain dicts so it JSON-serializes and unit-tests easily.

    `band` may be one column or an ordered sequence of them. Figure 2 draws all
    of them; everything else follows the first (the primary)."""
    if isinstance(band, str):
        band = (band,)
    params = parse_params({"exclude": ",".join(exclude) if exclude else "",
                           "stat": stat, "band": ",".join(band), "mac": mac,
                           "disc": disc, "mode": mode})
    excluded, bands, mode = params["exclude"], params["band"], params["mode"]
    band = bands[0]                      # the primary: Fig 3, ref line, table
    gone, out_eyes = split_excluded(excluded)
    included = [s for s in SUBJECTS if s not in gone]

    # every metric family at every band, for the averages table. FDR is applied
    # within one family at one band.
    avg_fits, avg_pf = {}, {}
    for b in T1_COLS:
        m_fits, m_pf = fit_family(excluded, MAC_METRICS, b, mode)
        d_fits, d_pf = fit_family(excluded, DISC_METRICS, b, mode)
        avg_fits[b] = {"mac": m_fits, "disc": d_fits}
        avg_pf[b] = {"mac": m_pf, "disc": d_pf}

    # Fig 2 panels fit every band against the selected sector; FDR within panel
    panel_fits = {}
    for key, sector in (("mac", params["mac"]), ("disc", params["disc"])):
        fits = [fit(excluded, sector, b, mode) for b in T1_COLS]
        pf = bh_fdr([f["p"] if f else float("nan") for f in fits])
        # reference: the DEFAULT sector's fit at the selected band, drawn as a grey
        # dotted line when a non-default sector is selected (mirrors index.html).
        default_sector = DEF_MAC if key == "mac" else DEF_DISC
        panel_fits[key] = {
            "sector": sector, "default_sector": default_sector,
            "ref": None if sector == default_sector else avg_fits[band][key][default_sector],
            "fits": dict(zip(T1_COLS, fits)),
            "pf": dict(zip(T1_COLS, pf)),
        }

    return {
        "params": params, "excluded": excluded,
        "subjects": included, "n": len(included),
        "n_eyes": sum(t not in out_eyes for t in ALL_EYES),
        "stat": params["stat"], "band": band, "bands": bands, "mode": mode,
        "mac": params["mac"], "disc": params["disc"],
        # Fig 3 colours its wedges from the selected band's family fits
        "mac_fits": avg_fits[band]["mac"], "mac_pf": avg_pf[band]["mac"],
        "disc_fits": avg_fits[band]["disc"], "disc_pf": avg_pf[band]["disc"],
        "panel_fits": panel_fits, "avg_pf": avg_pf, "avg_fits": avg_fits,
    }
