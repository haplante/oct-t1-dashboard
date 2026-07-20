# Optic Nerve OCT–T1 Shared-State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the OCT–T1 dashboard and the three standalone article figures to one shared state, fully described by a URL query string, with a Flask route serving the data for any state and a same-page event bus keeping every widget in sync.

**Architecture:** Extract a single source-of-truth Python module (`opticnerve_core.py`) holding stats, geometry, `resolve_view`, and `build_figN`. The Dash app, the Flask JSON route, and the three notebooks all consume it. The route returns Plotly figure specs; a framework-free `state_sync.js` does `fetch → Plotly.react`. Inline figures share an `opticnerve:state` CustomEvent bus; the iframed Dash dashboard bridges into it via `postMessage`.

**Tech Stack:** Python (Dash, Flask, Plotly, pandas, numpy, scipy, statsmodels), vanilla JS (Plotly.js), MyST-MD notebooks, pytest.

## Global Constraints

- Work on branch `main`, directly in the working tree (no feature branch/worktree) — per brief §8, confirmed by Hugo.
- State model is **6 params** with exact keys/defaults: `exclude`=`` (none), `stat`=`R2m`, `band`=`T1_mean_015`, `mac`=`All_1_3_gcc`, `disc`=`All_um_`, `mode`=`avg`.
- `stat` ∈ {`R2m`,`Rm`}; `band` ∈ {`T1_mean_015`,`T1_mean_05`,`T1_mean_510`,`T1_mean_1015`}; `mode` ∈ {`avg`,`lme`}.
- Canonical data lives in `data/` (`data_merged.csv`, `data_profile.csv`, `macula_OD.jpg`, `disc_OD.jpg`). Do **not** modify `dashboard_standard_reg/`.
- T1 stored in seconds → multiply by `T1_SCALE = 1000.0` for ms.
- Eye colours: `C_OD = "rgb(34,139,94)"`, `C_OS = "rgb(59,130,246)"`.
- API base URL must be env-configurable via `OPTICNERVE_API_BASE` (dev `http://localhost:3000`); never hardcode the prod host.
- Route returns Plotly figure specs from shared `build_figN` (not raw semantic data rebuilt in JS).
- Hand-roll param parsing; do **not** add `dash_querystrings`.
- Invalid/missing params normalize to defaults; the route never 500s on a bad query.
- Dashboard changes are additive — do not redesign its look/behaviour.
- **Do not modify `paper.md` prose.** Only add/adjust embedding machinery (the injected `<meta>`/script, the dashboard iframe block, and confirming the figure glue directives). The paper's numeric Results (R² values, etc.) come from an **old analysis and are NOT ground truth** — Hugo will rewrite those paragraphs himself after the interactive build is done. No task asserts against or edits those numbers.

---

### Task 1: Test scaffolding + shared param parsing

**Files:**
- Create: `content/opticnerve_core.py`
- Create: `tests/test_params.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DEFAULTS: dict` — `{"exclude": (), "stat": "R2m", "band": "T1_mean_015", "mac": "All_1_3_gcc", "disc": "All_um_", "mode": "avg"}`
  - `parse_params(mapping) -> dict` — accepts any dict-like (Flask `request.args`, plain dict); returns all 6 keys normalized. `exclude` → tuple of `MRI_ID` strings (blank/absent → `()`); unknown `stat`/`band`/`mode` → default.
  - `serialize_params(params) -> str` — canonical query string (keys in fixed order `exclude,stat,band,mac,disc,mode`; `exclude` joined by `,`).

- [ ] **Step 1: Add pytest to dev requirements**

Append to `requirements.txt`:

```
flask-cors>=4.0
pytest>=8.0
```

- [ ] **Step 2: Write `tests/conftest.py` so `content/` is importable**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "content"))
```

- [ ] **Step 3: Write the failing test**

`tests/test_params.py`:

```python
from opticnerve_core import DEFAULTS, parse_params, serialize_params


def test_empty_mapping_returns_defaults():
    assert parse_params({}) == DEFAULTS


def test_blank_exclude_is_empty_tuple():
    assert parse_params({"exclude": ""})["exclude"] == ()


def test_exclude_parsed_to_tuple():
    assert parse_params({"exclude": "Sub-0610,Sub-0611"})["exclude"] == ("Sub-0610", "Sub-0611")


def test_unknown_stat_falls_back_to_default():
    assert parse_params({"stat": "bogus"})["stat"] == "R2m"


def test_unknown_band_and_mode_fall_back():
    p = parse_params({"band": "nope", "mode": "nope"})
    assert p["band"] == "T1_mean_015"
    assert p["mode"] == "avg"


def test_valid_values_pass_through():
    p = parse_params({"stat": "Rm", "band": "T1_mean_05", "mode": "lme",
                      "mac": "N_1_3_gcc", "disc": "TS_um_"})
    assert (p["stat"], p["band"], p["mode"], p["mac"], p["disc"]) == \
           ("Rm", "T1_mean_05", "lme", "N_1_3_gcc", "TS_um_")


def test_serialize_is_canonical_and_ordered():
    qs = serialize_params(DEFAULTS)
    assert qs == "exclude=&stat=R2m&band=T1_mean_015&mac=All_1_3_gcc&disc=All_um_&mode=avg"


def test_roundtrip():
    p = {"exclude": ("Sub-0610",), "stat": "Rm", "band": "T1_mean_510",
         "mac": "S_3_6_gcc", "disc": "IT_um_", "mode": "lme"}
    assert parse_params(dict(zip(
        ["exclude", "stat", "band", "mac", "disc", "mode"],
        ["Sub-0610", "Rm", "T1_mean_510", "S_3_6_gcc", "IT_um_", "lme"]))) == p
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_params.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'opticnerve_core'`.

- [ ] **Step 5: Write minimal implementation**

Create `content/opticnerve_core.py`:

```python
"""Shared source of truth for the OCT–T1 figures: params, stats, geometry,
resolve_view, and the Plotly builders. Imported by the Dash app, the Flask
route, and the three notebooks."""

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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_params.py -v`
Expected: PASS (8 passed).

- [ ] **Step 7: Commit**

```bash
git add content/opticnerve_core.py tests/test_params.py tests/conftest.py requirements.txt
git commit -m "feat: shared 6-param parse/serialize + test scaffolding"
```

---

### Task 2: Extract stats, geometry, and `resolve_view` into the core

**Files:**
- Modify: `content/opticnerve_core.py`
- Modify: `content/Dash_client.py` (remove the moved definitions; import them)
- Create: `tests/test_resolve_view.py`

**Interfaces:**
- Consumes: `DEFAULTS`, `parse_params` (Task 1).
- Produces:
  - Constants: `T1_SCALE`, `C_OD`, `C_OS`, `T1_BANDS`, `T1_COLS`, `T1_COLS_ORDER`, `BAND_LABEL`, `DEF_MAC`, `DEF_DISC`, `PANELS`, `DISC_METRICS`, `MAC_METRICS`, `MAC_AVG`, `DISC_AVG`, `NAMES`, `sector_name`, `MAC_SECTORS`, `DISC_SECTORS`, `mac_q`, `mac_ang`, `disc_short`, `disc_th`, `LIM`, `PI`, `AX`, `MAC_URI`, `DISC_URI`.
  - Functions: `bh_fdr(p)`, `fit(excluded, sector, band, mode)`, `fit_family(excluded, metrics, band, mode)`, `stat_val(r, stat)`, `stat_lbl(stat)`, `fmt2(v)`, `jet(v, a, vmin)`, `wedge(th1, th2, ri, ro, n)`.
  - `resolve_view(exclude, stat, band, mac, disc, mode) -> dict` returning keys:
    `params` (normalized dict), `n` (int), `subjects` (list), `excluded` (tuple),
    `stat`, `band`, `mode`, `mac`, `disc`, and cached sub-results used by builders
    (`mac_fits`, `mac_pf`, `disc_fits`, `disc_pf` for the chosen band; `panel_fits`
    keyed by `("mac"|"disc", band)`). Exact shape below.

- [ ] **Step 1: Write the failing test (internal-consistency, NOT paper values)**

`tests/test_resolve_view.py` — pins `resolve_view` to an OLS recomputed independently
from the CSV in-test. This does **not** depend on the paper's numbers (which are from an
old analysis being revised); it only asserts the core is internally consistent.

```python
from pathlib import Path
import pandas as pd
from scipy.stats import linregress
from opticnerve_core import DEFAULTS, resolve_view


def _independent_ols_r2(sector, band):
    # avg-mode OLS on subject means, computed straight from the CSV, independent
    # of opticnerve_core — a reproducible anchor that survives analysis revisions.
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data" / "data_merged.csv")
    d = (df.groupby("MRI_ID", as_index=False)[[sector, band]].mean()
           .dropna(subset=[sector, band]))
    return linregress(d[sector], d[band]).rvalue ** 2


def test_default_view_counts_all_subjects():
    v = resolve_view(**DEFAULTS)
    assert v["n"] == len(v["subjects"]) > 0
    assert v["params"]["mode"] == "avg"


def test_default_macula_matches_independent_ols():
    v = resolve_view(**DEFAULTS)
    r = v["mac_fits"]["All_1_3_gcc"]
    assert abs(r["R2"] - _independent_ols_r2("All_1_3_gcc", "T1_mean_015")) < 1e-9


def test_default_disc_matches_independent_ols():
    v = resolve_view(**DEFAULTS)
    r = v["disc_fits"]["All_um_"]
    assert abs(r["R2"] - _independent_ols_r2("All_um_", "T1_mean_015")) < 1e-9


def test_default_fits_are_valid():
    # analysis-agnostic sanity: fits exist and R² is a proper fraction
    v = resolve_view(**DEFAULTS)
    for r in (v["mac_fits"]["All_1_3_gcc"], v["disc_fits"]["All_um_"]):
        assert r is not None and 0.0 <= r["R2"] <= 1.0


def test_excluding_a_subject_changes_n():
    v0 = resolve_view(**DEFAULTS)
    some = v0["subjects"][0]
    v1 = resolve_view(**{**DEFAULTS, "exclude": (some,)})
    assert v1["n"] == v0["n"] - 1


def test_lme_mode_produces_a_fit():
    v = resolve_view(**{**DEFAULTS, "mode": "lme"})
    assert v["mac_fits"]["All_1_3_gcc"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_resolve_view.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_view'`.

- [ ] **Step 3: Move the stats/geometry/data blocks into the core**

In `content/opticnerve_core.py`, add (moving **verbatim** from `content/Dash_client.py`, which currently defines them at the cited lines):
- Imports needed: `import warnings, base64`; `from functools import lru_cache`; `from pathlib import Path`; `import numpy as np, pandas as pd`; `from scipy.stats import linregress`; `import statsmodels.formula.api as smf`; `from plotly.colors import sample_colorscale`. Add `warnings.simplefilter("ignore")`.
- The `_HERE`/`DATA` resolver and data load block (`Dash_client.py:40-107`): `T1_SCALE`, `N_SLICES`, `MIN_N`, `C_OD`, `C_OS`, `T1_BANDS`, `T1_COLS`, `BAND_LABEL`, `DEF_MAC`, `DEF_DISC`, `PANELS`, `DISC_METRICS`, `MAC_METRICS`, `MAC_AVG`, `DISC_AVG`, `NAMES`, `sector_name`, the map geometry (`PI`, `LIM`, `rC/rI/rO`, `mac_q`, `mac_ang`, `MAC_SECTORS`, `disc_short`, `disc_order`, `disc_th`, `DISC_SECTORS`), `MERGED`, `PROFILE`, `SLICE_COLS`, `SUBJECTS`, `MAC_URI`, `DISC_URI`, `AX`.
- The stats block (`Dash_client.py:115-201`): `bh_fdr`, `fit`, `fit_family`, `stat_val`, `stat_lbl`, `fmt2`, `jet`, `wedge`.
- Add `T1_COLS_ORDER = [b[0] for b in T1_BANDS]` (currently `Dash_client.py:302`).

- [ ] **Step 4: Add `resolve_view` to the core**

Append to `content/opticnerve_core.py`:

```python
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
```

- [ ] **Step 5: Point `Dash_client.py` at the core (remove duplication)**

In `content/Dash_client.py`, delete the blocks moved in Step 3 and replace the top-of-file config/stats with:

```python
from opticnerve_core import (
    T1_SCALE, N_SLICES, MIN_N, C_OD, C_OS, T1_BANDS, T1_COLS, T1_COLS_ORDER,
    BAND_LABEL, DEF_MAC, DEF_DISC, PANELS, DISC_METRICS, MAC_METRICS, MAC_AVG,
    DISC_AVG, NAMES, sector_name, PI, LIM, rC, rI, rO, mac_q, mac_ang,
    MAC_SECTORS, disc_short, disc_th, DISC_SECTORS, MERGED, PROFILE, SLICE_COLS,
    SUBJECTS, MAC_URI, DISC_URI, AX, bh_fdr, fit, fit_family, stat_val, stat_lbl,
    fmt2, jet, wedge, resolve_view, DEFAULTS,
)
```

Keep the Dash imports (`from dash import ...`), `plotly.graph_objects as go`, `make_subplots`. The `build_fig1/2/3`, `build_avg_table`, layout, and callbacks stay for now (refactored in Task 3).

- [ ] **Step 6: Run tests + import smoke check**

Run: `python -m pytest tests/ -v`
Expected: PASS (all params + resolve_view tests).
Run: `python -c "import sys; sys.path.insert(0,'content'); import Dash_client"`
Expected: no output, exit 0 (module imports cleanly after the refactor).

- [ ] **Step 7: Commit**

```bash
git add content/opticnerve_core.py content/Dash_client.py tests/test_resolve_view.py
git commit -m "refactor: extract stats + resolve_view into opticnerve_core"
```

---

### Task 3: Move the Plotly builders into the core; drive the dashboard through resolve_view

**Files:**
- Modify: `content/opticnerve_core.py` (add `build_fig1/2/3`)
- Modify: `content/Dash_client.py` (import builders; callback calls `resolve_view`)
- Create: `tests/test_builders.py`

**Interfaces:**
- Consumes: `resolve_view` and all constants (Task 2).
- Produces:
  - `build_fig1(view) -> go.Figure`
  - `build_fig2(view) -> go.Figure`
  - `build_fig3(view) -> go.Figure`
  - `build_avg_table(view) -> list` stays in `Dash_client.py` (returns Dash `html.*`
    components, which don't belong in the framework-free core).

- [ ] **Step 1: Write the failing test**

`tests/test_builders.py`:

```python
import plotly.graph_objects as go
from opticnerve_core import DEFAULTS, resolve_view, build_fig1, build_fig2, build_fig3


def test_builders_return_figures():
    v = resolve_view(**DEFAULTS)
    for fn in (build_fig1, build_fig2, build_fig3):
        assert isinstance(fn(v), go.Figure)


def test_fig2_legend_names_carry_the_stat():
    v = resolve_view(**DEFAULTS)
    names = [t.name for t in build_fig2(v).data if t.name]
    assert any("R²" in n or "R²" in n for n in names)


def test_fig3_has_two_map_images():
    v = resolve_view(**DEFAULTS)
    assert len(build_fig3(v).layout.images) == 2


def test_fig1_serializes_to_dict():
    v = resolve_view(**DEFAULTS)
    d = build_fig1(v).to_dict()
    assert "data" in d and "layout" in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_builders.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_fig1'`.

- [ ] **Step 3: Move + adapt the builders into the core**

Move `build_fig1`/`build_fig2`/`build_fig3` from `Dash_client.py:207-367` into `opticnerve_core.py`, changing their signatures to take the `view` dict and read from it instead of recomputing:
- `build_fig1(view)`: filter `PROFILE` by `view["excluded"]`; body identical to `Dash_client.py:207-240`.
- `build_fig2(view)`: use `view["mode"]`, `view["stat"]`, `view["band"]` and `view["panel_fits"]["mac"|"disc"]` (already-computed fits + FDR) instead of calling `fit`/`bh_fdr` inline; the trace-drawing body is otherwise identical to `Dash_client.py:246-298`. Panel sectors come from `view["mac"]`/`view["disc"]`.
- `build_fig3(view)`: use `view["mac_fits"]/mac_pf` and `view["disc_fits"]/disc_pf` for the selected `view["band"]`; drawing body identical to `Dash_client.py:308-367`.

Add `import plotly.graph_objects as go` and `from plotly.subplots import make_subplots` to the core.

- [ ] **Step 4: Rewire the Dash callback + avg table through the view**

In `content/Dash_client.py`:
- Import the builders: add `build_fig1, build_fig2, build_fig3` to the `from opticnerve_core import (...)` list.
- Change `build_avg_table(excluded, mode, stat, sel)` to `build_avg_table(view)` reading `view["avg_pf"]`, `view["stat"]`, `view["mode"]`, `view["mac"]`, `view["disc"]`; the row/cell HTML body is unchanged (`Dash_client.py:373-413`), except `cell()` reads `view`'s precomputed fits/pf rather than calling `fit`.
- Replace the `_render` callback body (`Dash_client.py:515-523`) with:

```python
@app.callback(
    Output("fig1", "figure"), Output("fig2", "figure"), Output("fig3", "figure"),
    Output("avgbox", "children"), Output("ncount", "children"),
    Input("subjects", "value"), Input("stat", "value"), Input("t1band", "value"),
    Input("mode", "value"), Input("sel", "data"))
def _render(included, stat, sel_band, mode, sel):
    included = included or []
    excluded = tuple(sorted(set(SUBJECTS) - set(included)))
    view = resolve_view(exclude=excluded, stat=stat, band=sel_band,
                        mac=sel["mac"], disc=sel["disc"], mode=mode)
    count = f"{view['n']} / {len(SUBJECTS)} subjects included"
    return (build_fig1(view), build_fig2(view), build_fig3(view),
            build_avg_table(view), count)
```

- [ ] **Step 5: Run tests + launch smoke check**

Run: `python -m pytest tests/ -v`
Expected: PASS (params + resolve_view + builders).
Run (background, then curl): `python content/Dash_client.py &` then
`curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/`
Expected: `200`. Stop the server after.

- [ ] **Step 6: Manual parity check**

Open `http://127.0.0.1:8050/`, and separately serve the reference dashboard
(`cd dashboard_standard_reg && python -m http.server 8000`, open `http://localhost:8000`).
Confirm default-state Fig 1/2/3, the averages table, R² values, colours, and sector
selection behave identically. Note any drift; fix in the builder before continuing.

- [ ] **Step 7: Commit**

```bash
git add content/opticnerve_core.py content/Dash_client.py tests/test_builders.py
git commit -m "refactor: builders in core; dashboard renders via resolve_view"
```

---

### Task 4: Flask JSON route on the Dash server

**Files:**
- Modify: `content/Dash_client.py` (add the route + CORS + port)
- Create: `tests/test_route.py`

**Interfaces:**
- Consumes: `resolve_view`, `build_fig1/2/3`, `parse_params` (Tasks 2–3).
- Produces: HTTP `GET /opticnerve/<figid>` on `app.server`; JSON
  `{figid, params, n, figure}` for a single figure, or `{figid:"all", params, n, fig1, fig2, fig3}` for `all`.

- [ ] **Step 1: Write the failing test (Flask test client)**

`tests/test_route.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "content"))

import Dash_client  # noqa: E402


client = Dash_client.server.test_client()


def test_fig2_route_returns_plotly_spec():
    r = client.get("/opticnerve/fig2?exclude=&stat=R2m&band=T1_mean_015"
                   "&mac=All_1_3_gcc&disc=All_um_&mode=avg")
    assert r.status_code == 200
    body = r.get_json()
    assert body["figid"] == "fig2"
    assert "data" in body["figure"] and "layout" in body["figure"]
    assert body["params"]["mode"] == "avg"


def test_all_route_bundles_three_figures():
    r = client.get("/opticnerve/all")
    body = r.get_json()
    assert set(("fig1", "fig2", "fig3")).issubset(body)


def test_bad_params_clamp_to_defaults_no_500():
    r = client.get("/opticnerve/fig1?stat=bogus&band=bogus&mode=bogus")
    assert r.status_code == 200
    assert r.get_json()["params"]["stat"] == "R2m"


def test_unknown_figid_is_404():
    assert client.get("/opticnerve/figZ").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_route.py -v`
Expected: FAIL — 404 for `/opticnerve/fig2` (route not registered).

- [ ] **Step 3: Register the route**

In `content/Dash_client.py`, after `server = app.server` (`Dash_client.py:420`), add:

```python
from flask import request, jsonify, abort
from flask_cors import CORS

CORS(server, resources={r"/opticnerve/*": {"origins": "*"}})

_BUILDERS = {"fig1": build_fig1, "fig2": build_fig2, "fig3": build_fig3}


@server.route("/opticnerve/<figid>")
def opticnerve(figid):
    if figid not in _BUILDERS and figid != "all":
        abort(404)
    p = parse_params(request.args)
    view = resolve_view(**p)
    payload = {"figid": figid, "params": {**p, "exclude": list(p["exclude"])},
               "n": view["n"]}
    if figid == "all":
        payload.update({k: _BUILDERS[k](view).to_dict() for k in _BUILDERS})
    else:
        payload["figure"] = _BUILDERS[figid](view).to_dict()
    return jsonify(payload)
```

Add `parse_params` to the `from opticnerve_core import (...)` list.

- [ ] **Step 4: Set the dev port to 3000**

Change the bottom of `content/Dash_client.py` (`Dash_client.py:586-587`) to:

```python
if __name__ == "__main__":
    app.run(debug=True, port=3000)
```

- [ ] **Step 5: Run tests + curl the running server**

Run: `python -m pytest tests/test_route.py -v`
Expected: PASS (4 passed).
Run: `python content/Dash_client.py &` then
`curl -s "http://localhost:3000/opticnerve/fig2?exclude=&stat=R2m&band=T1_mean_015&mac=All_1_3_gcc&disc=All_um_&mode=avg" | python -c "import sys,json; d=json.load(sys.stdin); print(d['figid'], d['n'], len(d['figure']['data']))"`
Expected: `fig2 <n> <ntraces>` (non-zero). Stop the server.

- [ ] **Step 6: Commit**

```bash
git add content/Dash_client.py tests/test_route.py
git commit -m "feat: /opticnerve/<figid> JSON route returning Plotly specs"
```

---

### Task 5: Dashboard reads/writes the URL query string

**Files:**
- Modify: `content/Dash_client.py` (add `dcc.Location`, sync callbacks)

**Interfaces:**
- Consumes: `serialize_params`, `parse_params`, existing control ids (`subjects`,`stat`,`t1band`,`mode`,`sel`).
- Produces: dashboard is deep-linkable — controls initialize from `?…` on load and rewrite `?…` on every change (via `dcc.Location`).

- [ ] **Step 1: Add `dcc.Location` and a Store to the layout**

In `app.layout` (top, near `dcc.Store(id="sel"...)`, `Dash_client.py:430`) add:

```python
dcc.Location(id="url", refresh=False),
```

Add `serialize_params, parse_params` to the core import list if not already present.

- [ ] **Step 2: Initialize controls from the URL on load**

Add callback:

```python
@app.callback(
    Output("subjects", "value"), Output("stat", "value"), Output("t1band", "value"),
    Output("mode", "value"), Output("sel", "data"),
    Input("url", "search"), prevent_initial_call=False)
def _from_url(search):
    p = parse_params(_qs_to_dict(search))
    included = [s for s in SUBJECTS if s not in p["exclude"]]
    return included, p["stat"], p["band"], p["mode"], {"mac": p["mac"], "disc": p["disc"]}
```

Add the helper near the top of the callback section:

```python
from urllib.parse import parse_qsl

def _qs_to_dict(search):
    return dict(parse_qsl((search or "").lstrip("?")))
```

Note: `_from_url` replaces the existing `_select_all_none` wiring for `subjects.value`
only at load; keep `_select_all_none` but change it to also flow through the URL in Step 3
(Dash allows only one callback to own an output — merge All/None into `_from_url` by adding
`Input("sel-all","n_clicks")`, `Input("sel-none","n_clicks")` and branching on
`ctx.triggered_id`). Full merged callback:

```python
@app.callback(
    Output("subjects", "value"), Output("stat", "value"), Output("t1band", "value"),
    Output("mode", "value"), Output("sel", "data"),
    Input("url", "search"), Input("sel-all", "n_clicks"), Input("sel-none", "n_clicks"),
    State("stat", "value"), State("t1band", "value"), State("mode", "value"),
    State("sel", "data"), prevent_initial_call=False)
def _from_url(search, _a, _n, stat, band, mode, sel):
    trig = ctx.triggered_id
    if trig == "sel-all":
        return SUBJECTS, stat, band, mode, sel
    if trig == "sel-none":
        return [], stat, band, mode, sel
    p = parse_params(_qs_to_dict(search))
    included = [s for s in SUBJECTS if s not in p["exclude"]]
    return included, p["stat"], p["band"], p["mode"], {"mac": p["mac"], "disc": p["disc"]}
```

Delete the old `_select_all_none` callback (`Dash_client.py:476-480`).

- [ ] **Step 3: Write the URL on every state change**

```python
@app.callback(Output("url", "search"),
              Input("subjects", "value"), Input("stat", "value"),
              Input("t1band", "value"), Input("mode", "value"), Input("sel", "data"))
def _to_url(included, stat, band, mode, sel):
    excluded = tuple(sorted(set(SUBJECTS) - set(included or [])))
    return "?" + serialize_params({"exclude": excluded, "stat": stat, "band": band,
                                   "mac": sel["mac"], "disc": sel["disc"], "mode": mode})
```

- [ ] **Step 4: Manual verification**

Run `python content/Dash_client.py`. Open `http://localhost:3000/`. Change stat→R, band→0–5 mm, uncheck a subject, click a Fig-3 wedge. Confirm the address bar updates each time. Copy the URL, open in a new tab: the dashboard restores that exact state.

- [ ] **Step 5: Commit**

```bash
git add content/Dash_client.py
git commit -m "feat: dashboard reads/writes 6-param URL query string"
```

---

### Task 6: `state_sync.js` — inline figures fetch the route and share the bus

**Files:**
- Create: `content/static/state_sync.js`
- Create: `content/static/_test_page.html` (local manual harness; not shipped)

**Interfaces:**
- Consumes: the Flask route (Task 4), Plotly.js (global `Plotly`).
- Produces: global `OpticNerve.mount({figId, divId})`; the window CustomEvent
  `"opticnerve:state"` (detail = the 6 params); reads `API_BASE` from
  `<meta name="opticnerve-api">` / `window.OPTICNERVE_API_BASE` (default `http://localhost:3000`).

- [ ] **Step 1: Write the module**

`content/static/state_sync.js`:

```javascript
/* Shared-state client for the OCT–T1 inline figures.
   Each figure calls OpticNerve.mount({figId, divId}); all figures + the
   iframed dashboard stay in sync via the URL + the "opticnerve:state" bus. */
(function () {
  "use strict";
  const ORDER = ["exclude", "stat", "band", "mac", "disc", "mode"];
  const DEF = {exclude: "", stat: "R2m", band: "T1_mean_015",
               mac: "All_1_3_gcc", disc: "All_um_", mode: "avg"};

  function apiBase() {
    const m = document.querySelector('meta[name="opticnerve-api"]');
    return (window.OPTICNERVE_API_BASE || (m && m.content) || "http://localhost:3000")
      .replace(/\/$/, "");
  }
  function readParams() {
    const q = new URLSearchParams(location.search), p = {};
    ORDER.forEach(k => { p[k] = q.has(k) ? q.get(k) : DEF[k]; });
    return p;
  }
  function serialize(p) { return ORDER.map(k => `${k}=${p[k]}`).join("&"); }
  function equal(a, b) { return serialize(a) === serialize(b); }
  function writeURL(p) {
    history.replaceState(null, "", location.pathname + "?" + serialize(p) + location.hash);
  }

  const registry = [];          // {figId, divId, apply, last}
  let applying = false;         // loop guard

  function fetchAndRender(entry, p) {
    return fetch(`${apiBase()}/opticnerve/${entry.figId}?${serialize(p)}`)
      .then(r => r.json())
      .then(j => {
        Plotly.react(entry.divId, j.figure.data, j.figure.layout);
        entry.last = p;
        wireClicks(entry);
      });
  }

  // Fig 3 wedge clicks set mac/disc (toggle back to default on re-click).
  function wireClicks(entry) {
    if (entry.figId !== "fig3") return;
    const gd = document.getElementById(entry.divId);
    if (gd._onWired) return;
    gd._onWired = true;
    gd.on("plotly_click", ev => {
      const pt = ev.points && ev.points[0];
      const m = pt && (pt.data.meta || (pt.customdata && pt.customdata[0]));
      if (!m) return;
      const p = readParams();
      if (m.endsWith("_gcc")) p.mac = (p.mac === m) ? DEF.mac : m;
      else if (m.endsWith("_um_")) p.disc = (p.disc === m) ? DEF.disc : m;
      broadcast(p);
    });
  }

  // Originate a state change: write URL, re-render everyone, dispatch once.
  function broadcast(p) {
    writeURL(p);
    applying = true;
    Promise.all(registry.map(e => fetchAndRender(e, p))).finally(() => {
      applying = false;
      window.dispatchEvent(new CustomEvent("opticnerve:state", {detail: p}));
    });
  }

  // Receive external state (another figure, the dashboard bridge, or popstate).
  function receive(p) {
    if (applying) return;
    registry.forEach(e => { if (!e.last || !equal(e.last, p)) fetchAndRender(e, p); });
  }
  window.addEventListener("opticnerve:state", e => receive(e.detail));
  window.addEventListener("popstate", () => receive(readParams()));

  window.OpticNerve = {
    mount: function (opts) {
      const entry = {figId: opts.figId, divId: opts.divId, last: null};
      registry.push(entry);
      fetchAndRender(entry, readParams());
    },
    _readParams: readParams, _broadcast: broadcast   // exposed for the bridge/tests
  };
})();
```

- [ ] **Step 2: Build the manual harness page**

`content/static/_test_page.html`:

```html
<!doctype html><meta charset="utf-8">
<meta name="opticnerve-api" content="http://localhost:3000">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<div id="fig2" style="width:700px;height:340px"></div>
<div id="fig3" style="width:820px;height:340px"></div>
<script src="./state_sync.js"></script>
<script>
  OpticNerve.mount({figId:"fig2", divId:"fig2"});
  OpticNerve.mount({figId:"fig3", divId:"fig3"});
</script>
```

- [ ] **Step 3: Manual verification (route running)**

Run `python content/Dash_client.py` (port 3000). Serve the static dir:
`cd content/static && python -m http.server 8080`, open `http://localhost:8080/_test_page.html`.
Confirm: both figures load; clicking a Fig-3 macula wedge updates Fig 3's colours AND
re-renders Fig 2's macula panel to that sector; the address bar query string updates;
reloading the page with that query string restores the same view.

- [ ] **Step 4: Commit**

```bash
git add content/static/state_sync.js content/static/_test_page.html
git commit -m "feat: state_sync.js — inline figures fetch route + share event bus"
```

---

### Task 7: Bridge the iframed dashboard into the bus via postMessage

**Files:**
- Modify: `content/Dash_client.py` (clientside callback: emit/absorb postMessage)
- Modify: `content/static/state_sync.js` (relay bus ⇄ iframe)

**Interfaces:**
- Consumes: `opticnerve:state` bus (Task 6); dashboard `url.search` (Task 5).
- Produces: state changes made in the dashboard reach inline figures and vice versa,
  across the iframe boundary, with an origin check.

- [ ] **Step 1: Dashboard emits its URL changes to the parent**

In `content/Dash_client.py`, add a clientside callback (after the Python callbacks):

```python
app.clientside_callback(
    """
    function(search) {
        if (window.parent && window.parent !== window) {
            window.parent.postMessage(
                {type: "opticnerve:state", search: search || ""}, "*");
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("url", "search"),   # no_update — read-only use of the value
    Input("url", "search"),
)
```

- [ ] **Step 2: Dashboard absorbs parent messages into its URL**

Add a `dcc.Store(id="bridge")` to the layout and a clientside callback that listens once
for `message` and writes into the Location search. Add to layout near `dcc.Location`:

```python
dcc.Store(id="bridge"),
```

Add (clientside, registered at import time via a small inline script in `app.index_string`
`<footer>` is simplest — insert before `{%config%}`):

```html
<script>
window.addEventListener("message", function (e) {
  var d = e.data || {};
  if (d.type !== "opticnerve:state") return;
  // reflect into this app's URL so the normal _from_url callback runs
  if (typeof d.search === "string" && d.search !== window.location.search) {
    var url = window.location.pathname + d.search;
    window.history.replaceState(null, "", url);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }
});
</script>
```

Note: Dash's `dcc.Location` updates on `popstate`, so dispatching it makes `_from_url` re-run.

- [ ] **Step 3: Article side relays bus ⇄ iframe**

Append to `content/static/state_sync.js` inside the IIFE (before the closing `})();`):

```javascript
  // --- iframe bridge: relay between the same-page bus and the Dash iframe ---
  const ALLOW = (window.OPTICNERVE_DASH_ORIGIN || "").replace(/\/$/, "");
  function dashFrame() { return document.querySelector('iframe[data-opticnerve-dash]'); }

  // bus change -> tell the iframe
  window.addEventListener("opticnerve:state", e => {
    const f = dashFrame();
    if (f && f.contentWindow) {
      f.contentWindow.postMessage(
        {type: "opticnerve:state", search: "?" + serialize(e.detail)}, ALLOW || "*");
    }
  });
  // iframe change -> drive the bus
  window.addEventListener("message", e => {
    if (ALLOW && e.origin !== ALLOW) return;         // origin allow-check
    const d = e.data || {};
    if (d.type !== "opticnerve:state" || typeof d.search !== "string") return;
    const q = new URLSearchParams(d.search.replace(/^\?/, "")), p = {};
    ORDER.forEach(k => { p[k] = q.has(k) ? q.get(k) : DEF[k]; });
    const cur = readParams();
    if (!equal(cur, p)) broadcast(p);
  });
```

- [ ] **Step 4: Manual verification**

Extend `_test_page.html`: add `<iframe data-opticnerve-dash src="http://localhost:3000/" style="width:100%;height:600px"></iframe>` and set `window.OPTICNERVE_DASH_ORIGIN="http://localhost:3000"` before loading `state_sync.js`. With the Dash server running, open the page: changing the band in the iframe re-renders the inline Fig 2/3; clicking an inline Fig-3 wedge updates the iframe dashboard's Fig 2. Confirm no infinite loop (watch the console/network — one fetch per change).

- [ ] **Step 5: Commit**

```bash
git add content/Dash_client.py content/static/state_sync.js content/static/_test_page.html
git commit -m "feat: postMessage bridge between iframed dashboard and the bus"
```

---

### Task 8: Deliverable A — notebooks render default state via the shared core

**Files:**
- Modify: `content/figure_1.ipynb`, `content/figure_2.ipynb`, `content/figure_3.ipynb`

**Interfaces:**
- Consumes: `opticnerve_core` (`resolve_view`, `build_figN`, `DEFAULTS`).
- Produces: each notebook's `#| label: figNcell` cell renders the dashboard's
  default-state figure by the same code path.

- [ ] **Step 1: Rewrite `figure_1.ipynb` cell**

Replace the single code cell's source with:

```python
#| label: fig1cell
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "content")
                if os.path.isdir(os.path.join(os.getcwd(), "content")) else os.getcwd())
from opticnerve_core import resolve_view, build_fig1, DEFAULTS

fig = build_fig1(resolve_view(**DEFAULTS))
fig.show()
```

- [ ] **Step 2: Rewrite `figure_2.ipynb` cell**

```python
#| label: fig2cell
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "content")
                if os.path.isdir(os.path.join(os.getcwd(), "content")) else os.getcwd())
from opticnerve_core import resolve_view, build_fig2, DEFAULTS

fig = build_fig2(resolve_view(**DEFAULTS))
fig.show()
```

- [ ] **Step 3: Rewrite `figure_3.ipynb` cell**

```python
#| label: fig3cell
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "content")
                if os.path.isdir(os.path.join(os.getcwd(), "content")) else os.getcwd())
from opticnerve_core import resolve_view, build_fig3, DEFAULTS

fig = build_fig3(resolve_view(**DEFAULTS))
fig.show()
```

- [ ] **Step 4: Execute the notebooks (parity check)**

Run (from repo root, with `data/` present):
`python -m jupyter nbconvert --to notebook --execute --inplace content/figure_1.ipynb content/figure_2.ipynb content/figure_3.ipynb`
Expected: no errors; each notebook now has an executed Plotly output. Note: the core
reads `data/` relative to `opticnerve_core.py`'s location resolver, so execution works
regardless of the notebook's CWD.

- [ ] **Step 5: Visual parity check**

Open each executed notebook; confirm Fig 1/2/3 match the dashboard's default state
(same traces, colours, R² in the legend/wedges).

- [ ] **Step 6: Commit**

```bash
git add content/figure_1.ipynb content/figure_2.ipynb content/figure_3.ipynb
git commit -m "feat: notebooks render default state via shared opticnerve_core"
```

---

### Task 9: Deliverable C — mount `state_sync.js` in each notebook figure

**Files:**
- Modify: `content/figure_1.ipynb`, `content/figure_2.ipynb`, `content/figure_3.ipynb`

**Interfaces:**
- Consumes: `state_sync.js` (Task 6), the executed Plotly div id.
- Produces: each cell also emits an HTML snippet that loads `state_sync.js` once and
  calls `OpticNerve.mount` on that figure's div, so in a live browser the figure joins
  the bus. Static output from Task 8 remains the fallback.

- [ ] **Step 1: Append a mount snippet to `figure_1.ipynb`**

Add to the end of the cell (after `fig.show()`):

```python
from IPython.display import HTML, display
display(HTML('''
<div id="opticnerve-fig1"></div>
<script>
(function () {
  function boot() {
    var host = document.getElementById("opticnerve-fig1");
    var gd = host && host.previousElementSibling &&
             host.previousElementSibling.querySelector(".plotly-graph-div");
    if (!gd || !window.OpticNerve) { return setTimeout(boot, 150); }
    gd.id = gd.id || "opticnerve-fig1-gd";
    window.OpticNerve.mount({figId: "fig1", divId: gd.id});
  }
  if (!window.OpticNerve) {
    var s = document.createElement("script");
    s.src = (window.OPTICNERVE_STATIC || "/static") + "/state_sync.js";
    s.onload = boot; document.head.appendChild(s);
  } else { boot(); }
})();
</script>'''))
```

- [ ] **Step 2: Repeat for `figure_2.ipynb`**

Same snippet with `fig2`/`opticnerve-fig2` substituted for `fig1`/`opticnerve-fig1`.

- [ ] **Step 3: Repeat for `figure_3.ipynb`**

Same snippet with `fig3`/`opticnerve-fig3` substituted.

- [ ] **Step 4: Re-execute the notebooks**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace content/figure_1.ipynb content/figure_2.ipynb content/figure_3.ipynb`
Expected: no errors; each cell now emits the Plotly figure plus the mount `<script>`.

- [ ] **Step 5: Commit**

```bash
git add content/figure_1.ipynb content/figure_2.ipynb content/figure_3.ipynb
git commit -m "feat: mount state_sync.js in each notebook figure output"
```

---

### Task 10: Deliverable D — paper.md embedding, deps, binder env

**Files:**
- Modify: `paper.md`
- Modify: `requirements.txt` (already has flask-cors from Task 1)
- Create: `binder/requirements.txt`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: the article page injects `API_BASE`, embeds the dashboard as an iframe with
  the bridge, and renders the 3 figures inline (never iframed). NeuroLibre exec env has
  the deps. **Only embedding blocks change — no Results/prose text is touched.**

- [ ] **Step 1: Inject API base + dashboard origin near the top of `paper.md`**

Below the frontmatter, before `## Dashboard`, add:

```html
<meta name="opticnerve-api" content="http://localhost:3000">
<script>
  window.OPTICNERVE_API_BASE = "http://localhost:3000";
  window.OPTICNERVE_DASH_ORIGIN = "http://localhost:3000";
  window.OPTICNERVE_STATIC = "/build/static";  /* MyST static path; adjust at deploy */
</script>
```

(Env-driven substitution at deploy replaces `localhost:3000` with the NeuroLibre base.)

- [ ] **Step 2: Repoint the dashboard iframe + tag it for the bridge**

Replace the `## Dashboard` iframe block (`paper.md:9-14`) so the iframe carries
`data-opticnerve-dash` and points at the live app (env-driven), and load the bridge:

```html
## Dashboard

<div style="width:100%; aspect-ratio:1400/840; overflow:hidden;">
<div style="position:relative; width:1400px; height:840px; transform:scale(0.514286); transform-origin:top left;">
<iframe data-opticnerve-dash src="http://localhost:3000/" style="width:100%; height:100%; border:0;"></iframe>
</div>
</div>
<script src="/build/static/state_sync.js"></script>
```

- [ ] **Step 3: Confirm the 3 figures stay inline**

The existing `:::{figure} #fig1cell|#fig2cell|#fig3cell` directives (`paper.md:41,51,65`)
render executed notebook output inline — leave them as glue embeds. Do **not** wrap them
in `{iframe}`.

- [ ] **Step 4: Create `binder/requirements.txt`**

```
dash>=4.0
plotly>=6.0
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14
flask-cors>=4.0
gunicorn>=21.0
```

- [ ] **Step 5: Build check**

Run: `myst build --html` (or `myst start`) from the repo root with the Dash server
running on port 3000. Confirm: the article builds; the 3 figures render inline (inspect
DOM — they are `.plotly-graph-div`, not inside an `<iframe>`); the dashboard shows in its
iframe. Manually confirm a change in the dashboard updates the inline figures and vice
versa.

- [ ] **Step 6: Commit**

```bash
git add paper.md binder/requirements.txt requirements.txt
git commit -m "feat: embed dashboard (iframe+bridge) and inline figures in paper.md"
```

---

## Self-Review

**Spec coverage:**
- §4 state model → Task 1 (parse/serialize), verified throughout.
- §5 `opticnerve_core` (resolve_view + builders) → Tasks 2–3.
- §6 Flask route (Plotly specs, CORS, clamp, 404, env base) → Task 4 (+ Task 10 base injection).
- §7 `state_sync.js` (mount, fetch, react, local click, bus, loop guard, popstate) → Task 6.
- §8 dashboard changes (dcc.Location, clientside bridge) → Tasks 5, 7.
- §9 Deliverable A (notebooks import core) → Task 8.
- §10 Deliverable C (mount snippet, static fallback) → Task 9.
- §11 Deliverable D (inline figures, iframe+postMessage, meta injection) → Task 10.
- §12 deps (flask-cors, binder) → Tasks 1, 10.
- §13 dev order → mirrored by task order + each task's manual-verify step.
- §14 scope (no per-eye exclusion; don't touch index.html; additive) → honoured; no task modifies `dashboard_standard_reg/`.

**Deferred (spec §15, not in scope of these tasks):** production NeuroLibre URL and same-origin deploy topology — handled by env substitution in Task 10 Step 1; flagged for Agah before deploy.

**Placeholder scan:** no TBD/TODO; every code step shows full code; extraction steps cite exact source line ranges in `Dash_client.py` for verbatim moves.

**Type consistency:** `resolve_view(...)` keyword signature and return keys (`mac_fits`, `disc_fits`, `mac_pf`, `disc_pf`, `panel_fits`, `avg_pf`, `n`, `subjects`, `excluded`, `params`) are defined in Task 2 and consumed identically in Tasks 3–4. `build_figN(view)` signature consistent across Tasks 3, 4, 8, 9. `OpticNerve.mount({figId, divId})` and the `opticnerve:state` detail shape (6 params) consistent across Tasks 6, 7, 9. Route JSON keys (`figid`, `params`, `n`, `figure`) consistent across Task 4 and consumed in Task 6.
