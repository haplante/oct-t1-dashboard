# OCT – T₁ Correlation Figures (dashboard)

A static, client-side reproduction of the **three figures** from
`script/correlation_figures.ipynb`, stacked in a single view. Everything is
computed live in the browser (no server), so it deploys to Netlify as-is.

## What it shows

1. **Figure 1 — T₁ profile** along the optic nerve (OD / OS group means ± SD,
   plus faint per-eye lines).
2. **Figure 2 — Correlation** with 4 regression lines per panel
   (0–15, 0–5, 5–10, 10–15 mm) for the macula (`All_1_3_gcc`) and optic disc
   (`All_um_`) sectors. Marginal T₁ on *y*, OCT thickness on *x*.
3. **Figure 3 — OCT sector maps**: the chosen statistic of
   `T1_mean_015 ~ OCT_sector + (1|subject)` painted onto the macula (GCC) and
   disc (RNFL) sectors, over the scan underlays. `*` = `p_FDR < 0.05`.

Only the **skeleton** segmentation is used.

## Controls

- **Subjects** — tick / untick to include or exclude a subject from *every*
  figure. The random-intercept mixed-effects models are refit live.
  `sub-0610` is excluded by default (matching the notebook).
- **Statistic** — switch the value shown in Figures 2 & 3 between
  **R²ₘ** (marginal R², 0–1) and **Rₘ** (signed marginal *r*, −1…1).

## Statistics

Each line/sector is a random-intercept REML model `y ~ 1 + x + (1|subject)`,
fit exactly as in the notebook's `fit_ri_reml`:

- profiled over ρ = var(subject)/var(resid) with a closed-form 2×2 GLS;
- **marginal R²** = SSR_fixed / SST; **Rₘ** = sign(slope)·√R²ₘ;
- slope *p*-value from a Student-*t* test with **df = N − 2**;
- **Benjamini–Hochberg FDR** within the disc (9) and macula (12) metric sets.

These reproduce the notebook to 3 decimals (e.g. RNFL Overall R²ₘ = 0.675).

## Files

| File | Purpose |
| ---- | ------- |
| `index.html` | the whole dashboard (markup + Plotly + stats engine) |
| `data_merged.csv` | per-eye OCT + T₁ band means (skeleton, all subjects) |
| `data_profile.csv` | per-slice T₁ profile (skeleton, all subjects) |
| `macula_OD.jpg`, `disc_OD.jpg` | scan underlays for Figure 3 |

## Preview locally

The page fetches the CSVs, so it must be served over HTTP (not `file://`):

```
cd netlify_correlation_dashboard
python -m http.server
# open http://localhost:8000
```

## Deploy to Netlify

Drag-and-drop this folder onto <https://app.netlify.com/drop>, or point a
Netlify site at it with the publish directory set to this folder. No build
step is required.
