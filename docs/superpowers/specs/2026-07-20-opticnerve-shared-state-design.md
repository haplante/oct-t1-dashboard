# Optic Nerve OCT–T1 — Shared-State Interactive Publication

**Design spec** · 2026-07-20 · branch `main` (edits in working tree; Hugo handles all git)

## 1. Goal

Wire the finished OCT–T1 dashboard and the three standalone article figures so they
share one state, fully described by a URL query string. Any change from any widget
(dashboard control, dashboard figure click, or a standalone figure click) updates the
URL and causes every other widget on the page to re-render to match. A Flask route
returns the data for exactly the state named by the query string, so notebook figures
can fetch and render it.

This spec supersedes the brief's 5-param model (see §3) and reconciles the brief with
the repo's actual, further-along state (see §2).

## 2. Starting point (repo reality)

Already built:
- `content/Dash_client.py` — a complete Dash port of `dashboard_standard_reg/index.html`:
  all three figures, sidebar, clickable averages table, Fig-3 wedge clicks, dark theme.
- Statistics already in Python: `avg` mode via `scipy.linregress`; `lme` via
  `statsmodels.mixedlm`; `bh_fdr` reimplemented. `fit()`/`fit_family()` with `lru_cache`.
- `render.yaml` deploys the Dash app to Render via gunicorn.
- `paper.md` embeds a dashboard as an iframe to a Netlify static build, and has glue
  targets `#fig1cell/#fig2cell/#fig3cell`.
- `content/figure_1|2|3.ipynb` — single-cell interactive **Plotly** notebooks with
  `#| label: figNcell` directives; currently hardcoded to default state and OLS-only.

Not built yet (the novel work):
- No `dcc.Location` URL sync, no `resolve_view`, no Flask `/opticnerve` route, no
  `state_sync.js`, no `opticnerve:state` CustomEvent bus.

Deviations from the brief's stated layout:
- `binder/` directory is absent — will be created (§7).

## 3. Decisions (resolved with Hugo)

1. **Architecture:** Full brief architecture — live Dash app + Flask JSON route +
   same-page `opticnerve:state` CustomEvent bus wiring the 3 inline figures.
2. **State = 6 shared params**, not 5. The `mode` toggle (`avg` = Average+OLS vs
   `lme` = per-eye mixed model, marginal T1) is a full 6th shared param. LME kept.
3. **Dashboard embedding:** iframe + `postMessage` bridge to the same-page bus.
4. **Default `exclude`:** none excluded (neutral default; keeps all subjects incl.
   Sub-0610). Exclusion stays per-subject (by `MRI_ID`); no per-eye exclusion.
   Note: `paper.md`'s numeric Results are from an **old analysis and are not ground
   truth** — Hugo will rewrite those paragraphs after the interactive build is done.
   Tests therefore pin the code to an independently recomputed OLS, not to paper values,
   and no task edits the article prose.
5. **Data:** `data/` is canonical; all backends read from it. `dashboard_standard_reg/`
   keeps its own reference copies (duplication accepted, logged as tech debt).
6. **Production URL:** deferred/non-blocking. Dev = `http://localhost:3000`; prod
   NeuroLibre base URL from Agah before deploy. Env-configurable, never hardcoded.
7. **Route payload:** the route returns **Plotly figure specs** built by the shared
   Python `build_figN` functions (not raw semantic data rebuilt in JS).
8. **Param parsing:** hand-rolled shared `parse_params`/`serialize_params` (not
   `dash_querystrings`).
9. **Notebooks:** import the shared `content/opticnerve_core.py` — parity by
   construction; single source of truth for stats + plotting.

## 4. The state model

| param     | URL key   | values                                             | default          |
|-----------|-----------|----------------------------------------------------|------------------|
| exclude   | `exclude` | CSV of `MRI_ID`s removed via checkbox              | `` (none)        |
| stat      | `stat`    | `R2m` (R²) or `Rm` (R)                              | `R2m`            |
| band      | `band`    | `T1_mean_015｜T1_mean_05｜T1_mean_510｜T1_mean_1015` | `T1_mean_015`    |
| mac       | `mac`     | selected macula sector metric                      | `All_1_3_gcc`    |
| disc      | `disc`    | selected disc sector metric                        | `All_um_`        |
| mode      | `mode`    | `avg` or `lme`                                      | `avg`            |

Invalid/missing values normalize to the default (never error).

## 5. Shared Python core — `content/opticnerve_core.py`

Extract the stats + geometry + plotting from `Dash_client.py` into an importable module
consumed by the dashboard, the notebooks, and the Flask route.

```python
DEFAULTS = {"exclude": (), "stat": "R2m", "band": "T1_mean_015",
            "mac": "All_1_3_gcc", "disc": "All_um_", "mode": "avg"}

def parse_params(mapping) -> dict          # query mapping -> normalized 6 params
def serialize_params(params) -> str        # normalized params -> canonical query string

def resolve_view(exclude, stat, band, mac, disc, mode) -> dict:
    """Filter data (drop excluded MRI_IDs), run all fits (avg or lme), return
    everything needed to draw Fig 1/2/3 + the averages table for this state.
    Pure and cacheable. The brief's mandated single source of truth."""
    # -> {"params": {...}, "n": int, "subjects": [...],
    #     "fig1": <semantic>, "fig2": <semantic>, "fig3": <semantic>,
    #     "avg_table": <semantic>}

def build_fig1(view) -> go.Figure          # consume resolve_view output
def build_fig2(view) -> go.Figure
def build_fig3(view) -> go.Figure
def build_avg_table(view) -> list          # Dash html components (dashboard only)
```

`resolve_view` is the semantic layer; `build_figN` is the render layer on top of it.
Both Dash callbacks and the Flask route call the same path, so plotting logic exists
once. Existing `@lru_cache` on `fit()` keeps repeated states cheap.

## 6. Flask JSON route (same server as Dash, `app.server`)

```
GET /opticnerve/<figid>?exclude=&stat=&band=&mac=&disc=&mode=
    figid ∈ {fig1, fig2, fig3, all}
```

- Parses the same 6 params with `parse_params` (identical to the dashboard).
- Single figid response:
  ```json
  { "figid": "fig2",
    "params": {"exclude": [], "stat": "R2m", "band": "T1_mean_015",
               "mac": "All_1_3_gcc", "disc": "All_um_", "mode": "avg"},
    "n": 17,
    "figure": {"data": [ ... ], "layout": { ... }} }
  ```
  `figid=all` → `{fig1, fig2, fig3, avg_table}` specs in one payload.
- `figure` is `build_figN(resolve_view(**params)).to_dict()`.
- Base URL via env `OPTICNERVE_API_BASE` (dev `http://localhost:3000`).
- CORS enabled (`flask-cors`) — figures may be served from a different origin in dev.
- Bad params → clamp to default (no 500). Unknown figid → 404.

## 7. Client module — `content/static/state_sync.js`

Framework-free. Loaded once; each figure calls `OpticNerve.mount({figId, divId})`.

- **Mount:** read 6 params from `location.search` (fallback DEFAULTS) →
  `fetch(${API_BASE}/opticnerve/${figId}?${qs})` → `Plotly.react(div, figure.data,
  figure.layout)`. Re-attach the figure's own click handlers after each react.
- **Local interaction** (Fig 3 wedge click sets `mac`/`disc`, toggling back to default
  on re-click, mirroring `onFig3Click`): update params → `history.replaceState` →
  re-fetch/re-render self → dispatch `opticnerve:state` (detail = all 6 params).
- **Receive `opticnerve:state`:** if detail ≠ last-applied (serialized-params compare),
  re-fetch/re-render self; do **not** re-dispatch (loop guard).
- **`popstate`:** treated as an external state change (browser back/forward works).
- `API_BASE` from `<meta name="opticnerve-api">` or `window.OPTICNERVE_API_BASE`.

## 8. Dashboard changes — `Dash_client.py`

Additive only; layout/figures unchanged.
- `dcc.Location(id="url")` — read query on load, write on change.
- Refactor callbacks through `resolve_view`/`build_figN` (import from `opticnerve_core`).
- **Clientside bridge** (Dash runs inside the iframe):
  - On any control/click change: `history.replaceState` on the frame +
    `parent.postMessage({type:"opticnerve:state", detail}, "*")`.
  - `message` listener (with an explicit origin check) writes incoming `detail` into a
    hidden `dcc.Store(id="url-state")`; a Python callback fans it out to the controls.
- Article-page side (in `state_sync.js` or a small `iframe_bridge.js`): relay
  `opticnerve:state` ⇄ `postMessage` with the iframe, so inline figures and the iframed
  dashboard share one logical bus. `dcc.Location` also keeps the dashboard deep-linkable
  standalone.

## 9. Deliverable A — notebooks mirror the dashboard

- Each notebook cell imports the shared core and renders the **default-state** figure by
  the same code path as the dashboard:
  ```python
  from opticnerve_core import resolve_view, build_fig1, DEFAULTS
  build_fig1(resolve_view(**DEFAULTS)).show()
  ```
  Parity is guaranteed by construction; duplicated per-notebook stats/plot code removed.
- Keep `#| label: figNcell` so `paper.md` glue targets resolve.
- Ensure `opticnerve_core.py` is importable in the execution env (path/package setup).

## 10. Deliverable C — wire the standalone figures

- The published cell output keeps a **valid static default figure** (from §9) as a
  fallback and for NeuroLibre's executed-notebook archival.
- Each cell additionally emits (via `IPython.display.HTML`) a snippet that loads
  `state_sync.js` once and calls `OpticNerve.mount({figId, divId})` for that cell's
  Plotly div, so in a live browser the figure re-fetches from the route and joins the bus.
- Because MyST executes notebooks server-side (route may be unreachable at build time),
  the live re-fetch is a **browser-time enhancement**; the static default must be correct
  on its own — which §9 guarantees.

## 11. Deliverable D — embedding in `paper.md`

- **Do not modify the article prose.** Only the embedding machinery changes: the injected
  `<meta>`/script, the dashboard iframe block, and confirming the figure glue directives.
  The Results paragraphs (and their numbers) are Hugo's to rewrite later.
- **3 figures inline, never iframed** — already `:::{figure} #figNcell` glue embeds,
  which render executed notebook Plotly output into the page DOM (the same-page condition
  the CustomEvent bus requires). Verify the build does not iframe them.
- **Dashboard: iframe + postMessage** — repoint the existing iframe from the Netlify
  static build to the live Dash app (env-driven URL); add the postMessage bridge with an
  origin check. Note origin sensitivity: robust bridging wants the Dash app on the same
  origin as the article, else an explicit origin allow-check.
- **API base injection:** emit `<meta name="opticnerve-api">` once near the top of
  `paper.md`, value from an env var at build (dev `localhost:3000` / prod NeuroLibre).

## 12. Dependencies

- Top-level `requirements.txt`: add `flask-cors`. Add `requests` only if a notebook ends
  up fetching server-side (browser-time model makes it optional). Not adding
  `dash_querystrings`.
- Create `binder/requirements.txt` mirroring the top-level deps (NeuroLibre exec env).

## 13. Local dev / validation order

1. Dashboard at `localhost:3000`, refactored through `opticnerve_core`, matches
   `index.html`; data from `data/`. (`app.run(port=3000)`.)
2. Add `resolve_view` + Flask route; verify:
   `curl "localhost:3000/opticnerve/fig2?exclude=&stat=R2m&band=T1_mean_015&mac=All_1_3_gcc&disc=All_um_&mode=avg"`.
3. Wire Fig 2 first, then Fig 3 (wedge clicks) to the route; confirm own clicks update URL.
4. Add the `opticnerve:state` bus; test page with two inline figures + iframed dashboard;
   confirm a change in any one updates the others (incl. across the iframe).
5. Swap `localhost:3000` for the NeuroLibre base (env var) — needs Agah's real URL.

## 14. Scope boundaries (YAGNI)

- Per-subject exclusion only (no per-eye).
- Do not modify `dashboard_standard_reg/index.html` (reference source of truth).
- No dashboard redesign — additive state wiring only.
- No git actions; edits in the working tree on `main`. Hugo stages/commits/pushes.

## 15. Open / deferred

- Production NeuroLibre base URL (from Agah) — blocks only step 13.5.
- Same-origin vs cross-origin serving of the Dash app affects postMessage robustness;
  resolve at deploy time with an origin allow-list.
