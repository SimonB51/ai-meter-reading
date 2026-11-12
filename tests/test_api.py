import os
from pathlib import Path

os.environ["STORAGE"] = "local"
FIX = Path(__file__).parent / "fixtures"


def test_upload_review_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    import importlib, meter_reader.api as api
    importlib.reload(api)
    c = TestClient(api.app)

    r = c.post("/api/readings", files={"file": ("m.jpg", (FIX / "1ec531.jpg").read_bytes(), "image/jpeg")},
               data={"meter_id": "t1"})
    assert r.status_code == 200
    rec = r.json()
    assert rec["value"] == "926"
    assert c.get(f"/api/readings/{rec['id']}/image").status_code == 200

    r = c.post(f"/api/readings/{rec['id']}/review", json={"value": "926"})
    assert r.json()["status"] == "confirmed"
    assert c.post(f"/api/readings/{rec['id']}/review", json={"value": "92"}).status_code == 422

    s = c.get("/api/stats").json()
    assert s["total"] == 1 and s["confirmed"] == 1
    assert "926" in c.get("/api/export/corrections.csv").text


def test_rejects_non_image(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    import meter_reader.api as api
    c = TestClient(api.app)
    r = c.post("/api/readings", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400
