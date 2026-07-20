"""
OCT – T1 correlation dashboard  (Plotly Dash app)

Recreates the three figures of the static HTML dashboard, plus two regression
modes selectable in the sidebar:

  * "Average + OLS"  -> average both eyes per subject, ordinary least squares
  * "LME"            -> per-eye data, linear mixed-effects model with a random
                        intercept per subject; Figure 2 plots the marginal T1
                        (observed T1 minus the subject random intercept).

The code is intentionally written in the same plain style as the figure
notebooks so it can be read top to bottom.

Run locally:   python Dash_client.py       (opens http://127.0.0.1:8050)
Deploy:        gunicorn Dash_client:server  (see render.yaml)
"""

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State, ctx, ALL

from opticnerve_core import (
    T1_SCALE, N_SLICES, MIN_N, C_OD, C_OS, T1_BANDS, T1_COLS, T1_COLS_ORDER,
    BAND_LABEL, DEF_MAC, DEF_DISC, PANELS, DISC_METRICS, MAC_METRICS, MAC_AVG,
    DISC_AVG, NAMES, sector_name, PI, LIM, rC, rI, rO, mac_q, mac_ang,
    MAC_SECTORS, disc_short, disc_th, DISC_SECTORS, MERGED, PROFILE, SLICE_COLS,
    SUBJECTS, MAC_URI, DISC_URI, AX, bh_fdr, fit, fit_family, stat_val, stat_lbl,
    fmt2, jet, wedge, resolve_view, DEFAULTS,
)

# ============================================================================
# FIGURE 1 — T1 profile along the optic nerve
# ============================================================================
def build_fig1(excluded):
    prof = PROFILE[~PROFILE["MRI_ID"].isin(excluded)]
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
        autosize=True, paper_bgcolor="white", plot_bgcolor="white", dragmode=False,
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
def build_fig2(excluded, mode, stat, sel_band, sel):
    y_title = ("marginal ON T₁ (ms)" if mode == "lme" else "ON T₁ (ms)")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=[f"Macula – {sector_name(sel['mac'])}",
                                        f"Optic Disc – {sector_name(sel['disc'])}"])
    for col, (title, xlab, sector) in enumerate(
            [(PANELS[0][0], PANELS[0][2], sel["mac"]),
             (PANELS[1][0], PANELS[1][2], sel["disc"])], start=1):
        leg = "legend" if col == 1 else "legend2"
        fits = [fit(excluded, sector, band, mode) for band in T1_COLS_ORDER]
        pf = bh_fdr([f["p"] if f else np.nan for f in fits])
        for (band, lbl, c), r, q in zip(T1_BANDS, fits, pf):
            if not r:
                continue
            vis = True if band == sel_band else "legendonly"
            grp = f"{col}-{band}"
            x, y, subj = np.array(r["x"]), np.array(r["y"]), np.array(r["subj"])
            hov = lambda side="": (f"<b>%{{customdata}}</b>{side} · {lbl}<br>"
                                   f"{sector_name(sector)} = %{{x:.2f}}"
                                   f"<br>T₁ = %{{y:.0f}} ms<extra></extra>")
            if r["eye"] is None:                       # average mode: one marker per subject
                fig.add_scatter(x=x, y=y, mode="markers", legend=leg, legendgroup=grp,
                    showlegend=False, visible=vis, row=1, col=col, customdata=subj,
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
                        customdata=subj[sel_e], marker=marker,
                        hovertemplate=hov(f" · {e}"), hoverlabel=dict(bgcolor="#222", font=dict(color="#fff")))
            xs = np.array([x.min(), x.max()])
            star = " *" if (not np.isnan(q) and q < 0.05) else ""
            fig.add_scatter(x=xs, y=r["b0"] + r["b1"] * xs, mode="lines", legend=leg,
                legendgroup=grp, visible=vis, row=1, col=col, line=dict(color=c, width=2),
                hoverinfo="skip", name=f"{lbl}  ({stat_lbl(stat)}={fmt2(stat_val(r, stat))}){star}")
        fig.update_xaxes(title=dict(text=xlab, standoff=5), row=1, col=col, **AX)
        fig.update_yaxes(title=dict(text=y_title), row=1, col=col, **AX)

    leg_style = dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1,
                     font=dict(size=12), xanchor="right", yanchor="top", y=0.99,
                     groupclick="togglegroup", tracegroupgap=1)
    fig.update_layout(autosize=True, paper_bgcolor="white", plot_bgcolor="white", dragmode=False,
        font=dict(color="black", family="DejaVu Sans, Arial, sans-serif", size=13),
        margin=dict(l=70, r=10, t=40, b=45),
        legend=dict(**leg_style, x=0.43), legend2=dict(**leg_style, x=0.99))
    for a in fig.layout.annotations:
        a.font.size = 15
    return fig


# ============================================================================
# FIGURE 3 — OCT sector maps
# ============================================================================
def build_fig3(excluded, mode, stat, sel_band):
    vmin = -1 if stat == "Rm" else 0
    band_lbl = BAND_LABEL[sel_band]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0)
    ann = []

    def draw(sectors, axis, image):
        fits, pf = fit_family(excluded, [s["m"] for s in sectors], sel_band, mode)
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

    xa, ya = draw(MAC_SECTORS, 1, MAC_URI)
    xa2, ya2 = draw(DISC_SECTORS, 2, DISC_URI)
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
    fig.update_layout(autosize=True, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black", family="DejaVu Sans, Arial, sans-serif"), dragmode=False,
        margin=dict(l=0, r=0, t=30, b=15), showlegend=False, annotations=ann,
        xaxis=dict(**axb, domain=[0, 0.5], scaleanchor="y", scaleratio=1), yaxis=dict(**axb, domain=[0, 1]),
        xaxis2=dict(**axb, domain=[0.5, 1], scaleanchor="y2", scaleratio=1), yaxis2=dict(**axb, domain=[0, 1]))
    return fig


# ============================================================================
# AVERAGES TABLE — clickable rows drive Figure 2
# ============================================================================
def build_avg_table(excluded, mode, stat, sel):
    # fit whole families (for correct FDR), keep the displayed rows
    per_band = {}
    for band, _, _ in T1_BANDS:
        _, mac_pf = fit_family(excluded, MAC_METRICS, band, mode)
        _, disc_pf = fit_family(excluded, DISC_METRICS, band, mode)
        per_band[band] = dict(mac=mac_pf, disc=disc_pf)

    def cell(metric, region, band):
        r = fit(excluded, metric, band, mode)
        val = stat_val(r, stat)
        pf = per_band[band][region][metric]
        sig = bool(r and not np.isnan(pf) and pf < 0.05)
        txt = ("—" if (val is None or np.isnan(val)) else f"{val:.2f}") + ("*" if sig else "")
        return html.Td(txt, className="aval sig" if sig else "aval")

    def rows(group_label, items, region, selected):
        out = []
        for i, (metric, name) in enumerate(items):
            tds = [cell(metric, region, band) for band, _, _ in T1_BANDS]
            region_td = ([html.Td(group_label, className="region", rowSpan=len(items))]
                         if i == 0 else [])
            out.append(html.Tr(
                region_td + [html.Td(name, className="anm")] + tds,
                id={"type": "avgrow", "metric": metric, "region": region},
                n_clicks=0,
                className="avgrow sel" if metric == selected else "avgrow"))
        return out

    header = html.Thead(html.Tr(
        [html.Th("T₁ sector", className="corner", colSpan=2)]
        + [html.Th(lbl) for _, lbl, _ in T1_BANDS]))
    body = html.Tbody(
        rows("Macula (GCC)", MAC_AVG, "mac", sel["mac"])
        + rows("Optic Disc (RNFL)", DISC_AVG, "disc", sel["disc"]))
    return [
        html.Div([f"OCT × T₁ band correlation · {stat_lbl(stat)} ",
                  html.Span("— click a row to use that sector in Figure 2", className="avghint")],
                 className="avgtitle"),
        html.Table([header, body], className="avgtbl"),
    ]


# ============================================================================
# DASH APP  (layout + callbacks)
# ============================================================================
app = Dash(__name__, title="OCT – T₁ Correlation")
server = app.server                      # gunicorn entry point (Render)

GRAPH_CFG = lambda name, w, h: dict(
    scrollZoom=False, displaylogo=False, displayModeBar="hover", responsive=True,
    modeBarButtonsToRemove=["select2d", "lasso2d", "zoom2d", "pan2d", "zoomIn2d",
                            "zoomOut2d", "autoScale2d", "resetScale2d"],
    toImageButtonOptions=dict(format="png", filename=name, width=w, height=h, scale=600 / 96))

app.layout = html.Div([
    html.H1(["OCT – T", html.Sub("1"), " Correlation Figures"]),
    dcc.Store(id="sel", data={"mac": DEF_MAC, "disc": DEF_DISC}),
    html.Div(id="app", children=[
        # ---- sidebar ----
        html.Aside(id="sidebar", children=[
            html.H3("Subjects"),
            html.Div("Uncheck a subject to remove it from every figure.", className="hint"),
            html.Div([html.Button("All", id="sel-all"), html.Button("None", id="sel-none")],
                     className="btnrow"),
            dcc.Checklist(id="subjects", options=[{"label": s, "value": s} for s in SUBJECTS],
                          value=SUBJECTS, className="subjlist"),
            html.Div(id="ncount", className="ncount"),
            html.Hr(),
            html.Div("Statistic", className="lbl"),
            dcc.RadioItems(id="stat", value="R2m",
                           options=[{"label": " R²", "value": "R2m"}, {"label": " R", "value": "Rm"}]),
            html.Hr(),
            html.Div("T₁ sector", className="lbl"),
            dcc.Dropdown(id="t1band", clearable=False, value="T1_mean_015",
                         options=[{"label": lbl, "value": k} for k, lbl, _ in T1_BANDS]),
            html.Hr(),
            html.Div("Regression model", className="lbl"),
            dcc.RadioItems(id="mode", value="avg",
                           options=[{"label": " Average + OLS", "value": "avg"},
                                    {"label": " LME (per-eye, marginal T₁)", "value": "lme"}]),
        ]),
        # ---- figures ----
        html.Div(id="figures", children=[
            html.Div(className="leftcol", children=[
                html.Section(className="figpanel", children=[
                    dcc.Graph(id="fig1", style={"height": "100%"},
                              config=GRAPH_CFG("fig01_T1_profile", 864, 360))]),
                html.Section(className="figpanel statpanel", children=[
                    html.Div(id="avgbox")]),
            ]),
            html.Section(className="figpanel fig2", children=[
                dcc.Graph(id="fig2", style={"height": "100%"},
                          config=GRAPH_CFG("fig02_regression", 936, 408))]),
            html.Section(className="figpanel fig3", children=[
                dcc.Graph(id="fig3", style={"height": "100%"},
                          config=GRAPH_CFG("fig03_OCT_maps", 816, 408))]),
        ]),
    ]),
])


# ---- All / None buttons set the subject checklist ----
@app.callback(Output("subjects", "value"),
              Input("sel-all", "n_clicks"), Input("sel-none", "n_clicks"),
              prevent_initial_call=True)
def _select_all_none(_a, _n):
    return SUBJECTS if ctx.triggered_id == "sel-all" else []


# ---- clicking a Fig 3 wedge or an averages row selects the Fig 2 sector ----
@app.callback(Output("sel", "data"),
              Input("fig3", "clickData"),
              Input({"type": "avgrow", "metric": ALL, "region": ALL}, "n_clicks"),
              State("sel", "data"), prevent_initial_call=True)
def _select_sector(click, _rows, sel):
    sel = dict(sel)
    trig = ctx.triggered_id
    metric, region = None, None
    if trig == "fig3" and click and click.get("points"):
        pt = click["points"][0]
        metric = pt.get("customdata", pt.get("meta"))     # per-point id, meta as fallback
        if isinstance(metric, (list, tuple)):             # some Plotly builds wrap customdata
            metric = metric[0] if metric else None
        if isinstance(metric, str):
            region = "mac" if metric.endswith("_gcc") else "disc"
        else:
            metric = None
    elif isinstance(trig, dict) and trig.get("type") == "avgrow":
        metric, region = trig["metric"], trig["region"]
    if metric:                                   # toggle: click the selected one to reset
        default = DEF_MAC if region == "mac" else DEF_DISC
        sel[region] = default if sel[region] == metric else metric
    return sel


# ---- main render ----
@app.callback(
    Output("fig1", "figure"), Output("fig2", "figure"), Output("fig3", "figure"),
    Output("avgbox", "children"), Output("ncount", "children"),
    Input("subjects", "value"), Input("stat", "value"), Input("t1band", "value"),
    Input("mode", "value"), Input("sel", "data"))
def _render(included, stat, sel_band, mode, sel):
    included = included or []
    excluded = tuple(sorted(set(SUBJECTS) - set(included)))
    count = f"{len(included)} / {len(SUBJECTS)} subjects included"
    return (build_fig1(excluded),
            build_fig2(excluded, mode, stat, sel_band, sel),
            build_fig3(excluded, mode, stat, sel_band),
            build_avg_table(excluded, mode, stat, sel),
            count)


# ---- dark theme (kept inline so the app stays a single file) ----
app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { margin:0; background:#111; color:#eee; font-family:Arial,Helvetica,sans-serif; }
  h1 { text-align:center; font-size:22px; margin:14px 0 8px; }
  #app { display:flex; align-items:stretch; gap:16px; padding:0 16px 16px; height:calc(100vh - 60px); }
  #sidebar { width:210px; flex:0 0 210px; background:#1e1e1e; border-radius:6px;
    padding:14px; overflow-y:auto; }
  #sidebar h3 { margin:0 0 8px; font-size:14px; }
  #sidebar .hint { font-size:11px; color:#999; margin-bottom:10px; line-height:1.4; }
  #sidebar .lbl { font-size:12px; color:#aaa; margin-bottom:6px; }
  #sidebar hr { border:none; border-top:1px solid #333; margin:14px 0 12px; }
  .btnrow { display:flex; gap:6px; margin-bottom:8px; }
  .btnrow button { flex:1; background:#2a2a2a; color:#ddd; border:1px solid #444;
    border-radius:4px; padding:4px; font-size:11px; cursor:pointer; }
  .btnrow button:hover { background:#3a3a3a; }
  /* all sidebar OPTIONS share one font size (incl. the dropdown value + menu) */
  .subjlist label,
  #stat label, #mode label,
  #t1band .Select-control, #t1band .Select-value-label,
  #t1band .Select-placeholder, #t1band .Select-menu-outer,
  #t1band .Select-option, #t1band .VirtualizedSelectOption { font-size:12px; }
  .subjlist label { display:flex; align-items:center; padding:1px 0; cursor:pointer; color:#9ecbff; }
  .subjlist input { margin-right:7px; }
  #stat label { display:inline-block; color:#ddd; margin-right:18px; cursor:pointer; }
  #mode label { display:block; color:#ddd; margin-bottom:5px; cursor:pointer; }
  #stat input, #mode input { margin-right:7px; }
  .ncount { font-size:11px; color:#7fd17f; margin-top:8px; }
  #sidebar .dash-dropdown { color:#111; }
  #figures { flex:1 1 auto; min-width:0; display:grid; gap:16px;
    grid-template-columns:7fr 8fr; grid-template-rows:1fr 1fr;
    grid-template-areas:"f1 f2" "f1 f3"; }
  .leftcol { grid-area:f1; display:flex; flex-direction:column; gap:16px; min-height:0; }
  .leftcol .figpanel:first-child { flex:1 1 auto; min-height:0; }
  .fig2 { grid-area:f2; } .fig3 { grid-area:f3; }
  .figpanel { background:#fff; border-radius:6px; padding:10px; min-width:0; min-height:0;
    box-shadow:0 1px 4px rgba(0,0,0,.4); }
  .statpanel { background:#1e1e1e; box-shadow:none; padding:12px; overflow:auto; flex:0 0 auto; }
  .avgtitle { font-size:14px; color:#eee; margin-bottom:10px; }
  .avghint { font-size:11px; color:#777; }
  .avgtbl { border-collapse:collapse; width:100%; table-layout:fixed; }
  .avgtbl th { font-size:12px; font-weight:normal; color:#999; text-align:center; padding:2px 8px;
    text-transform:uppercase; letter-spacing:.4px; }
  .avgtbl th.corner { text-align:left; }
  .avgtbl td { padding:3px 8px; font-size:12px; text-align:center; }
  .avgtbl td.anm { text-align:left; color:#ddd; }
  .avgtbl td.region { text-align:left; color:#bbb; font-size:12px; font-weight:bold;
    text-transform:uppercase; letter-spacing:.4px; vertical-align:middle; }
  .avgtbl tr.avgrow { cursor:pointer; }
  .avgtbl tr.avgrow:hover td { background:#2a2a2a; }
  .avgtbl tr.avgrow.sel td { background:#26323f; }
  .avgtbl tr.avgrow.sel td.region { background:transparent; }
  .avgtbl .aval { color:#fff; font-weight:600; font-variant-numeric:tabular-nums; }
  .avgtbl .aval.sig { color:#ff383c; }
  g.annotation rect { rx:6px; ry:6px; }
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""


if __name__ == "__main__":
    app.run(debug=True)
