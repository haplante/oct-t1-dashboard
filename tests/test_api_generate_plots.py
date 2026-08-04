import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import server


def _post(client, payload):
    return client.post(
        "/api/generate_plots",
        data=json.dumps(payload),
        content_type="application/json",
    )


def test_generate_plots_default_fig1_returns_figure():
    client = server.test_client()
    resp = _post(client, {"figid": "fig1"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["figid"] == "fig1"
    assert "figure" in body
    assert body["params"]["stat"] == "R2m"


def test_generate_plots_all_returns_three_figures():
    client = server.test_client()
    resp = _post(client, {"figid": "all"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(["fig1", "fig2", "fig3"]).issubset(body.keys())


def test_generate_plots_respects_params():
    client = server.test_client()
    resp = _post(client, {"figid": "fig2", "stat": "Rm", "mode": "lme"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["params"]["stat"] == "Rm"
    assert body["params"]["mode"] == "lme"


def test_generate_plots_rejects_unknown_figid():
    client = server.test_client()
    resp = _post(client, {"figid": "nope"})
    assert resp.status_code == 404


def test_generate_plots_returns_500_json_on_unexpected_error():
    client = server.test_client()
    with patch("app.resolve_view", side_effect=RuntimeError("boom")):
        resp = _post(client, {"figid": "fig1"})
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["error"] == "boom"
