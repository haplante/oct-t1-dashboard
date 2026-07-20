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
