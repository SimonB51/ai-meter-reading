"""FastAPI service: upload a photo -> reading + confidence; review queue; stats; export.

run locally:  uvicorn meter_reader.api:app --app-dir src --port 8080
"""
from __future__ import annotations

import csv
import io
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import anomaly
from .pipeline import MeterReader
from .storage import make_storage, new_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("meter")

app = FastAPI(title="AI Meter Reading", version="1.0")
reader = MeterReader(review_threshold=float(os.getenv("REVIEW_THRESHOLD", "0.5")))
store = make_storage()
STATIC = Path(__file__).parent / "static"
MAX_BYTES = 10 * 1024 * 1024


class Review(BaseModel):
    value: str  # the 3 digits a human read on the photo


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/ops")
def ops():
    """Operations view: review queue + dashboard."""
    return FileResponse(STATIC / "ops.html")


SAMPLES = Path(os.getenv("SAMPLES_DIR", Path(__file__).resolve().parents[2] / "samples"))


@app.on_event("startup")
def warm_up():
    """Run one full reading at startup: the first inference (ONNX sessions, torch kernels) is
    slow, so do it before the container reports healthy instead of on a visitor's first photo."""
    import cv2
    first = next(iter(sorted(SAMPLES.glob("*.jpg"))), None)
    if first is not None:
        try:
            reader.read(cv2.imread(str(first)))
            log.info("warm-up reading done")
        except Exception as e:  # never block startup on the warm-up
            log.warning("warm-up failed: %s", e)


@app.get("/api/samples")
def list_samples():
    return sorted(p.name for p in SAMPLES.glob("*.jpg"))


@app.get("/samples/{name}")
def get_sample(name: str):
    p = (SAMPLES / name).resolve()
    if p.parent != SAMPLES.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/readings")
async def create_reading(file: UploadFile = File(...), meter_id: str | None = Form(None)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large (10 MB max)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    r = reader.read(img)                     # read at full resolution: digits can be small
    # store a lighter copy (phones send 12 MP photos)
    s = 1280 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rid = new_id()
    key = f"{rid}.jpg"
    store.put_image(key, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())

    rec = {"id": rid, "meter_id": meter_id or "web-demo", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "image_key": key, **{k: v for k, v in r.to_dict().items() if k != "band_jpg"}}
    rec["status"] = r.status
    if r.value and meter_id:   # consumption check only makes sense for a known meter
        history = [x for x in store.list() if x["meter_id"] == meter_id]
        rec["consumption"], rec["anomaly"] = anomaly.check(r.value, history)
        if rec["anomaly"] and rec["status"] == "auto":
            rec["status"] = "review"   # suspicious consumption -> a human double-checks
    store.save(rec)
    rec = {**rec, "band_jpg": r.band_jpg}     # returned to the page, not stored
    log.info("reading id=%s meter=%s value=%s conf=%.3f status=%s method=%s ms=%s",
             rid, meter_id, r.value, r.confidence, rec["status"], r.method, r.timings_ms.get("total"))
    return rec


@app.get("/api/readings")
def list_readings(status: str | None = None, limit: int = 200):
    recs = store.list()
    if status:
        recs = [r for r in recs if r["status"] == status]
    return recs[:limit]


@app.get("/api/readings/{rid}/image")
def get_image(rid: str):
    rec = store.get(rid)
    if not rec:
        raise HTTPException(404)
    return Response(store.get_image(rec["image_key"]), media_type="image/jpeg")


@app.post("/api/readings/{rid}/review")
def review(rid: str, body: Review):
    if not (body.value.isdigit() and len(body.value) == 3):
        raise HTTPException(422, "value must be exactly 3 digits")
    rec = store.get(rid)
    if not rec:
        raise HTTPException(404)
    rec["corrected_value"] = body.value
    rec["status"] = "confirmed" if body.value == rec.get("value") else "corrected"
    rec["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    store.save(rec)
    log.info("review id=%s model=%s human=%s", rid, rec.get("value"), body.value)
    return rec


@app.get("/api/stats")
def stats():
    recs = store.list()
    n = len(recs)
    reviewed = [r for r in recs if r["status"] in ("confirmed", "corrected")]
    by = lambda s: sum(r["status"] == s for r in recs)  # noqa: E731
    return {
        "total": n,
        "auto": by("auto"),
        "pending_review": by("review"),
        "failed": by("failed"),
        "confirmed": by("confirmed"),
        "corrected": by("corrected"),
        "automation_rate": round(by("auto") / n, 3) if n else None,
        "model_accuracy_on_reviewed": round(by("confirmed") / len(reviewed), 3) if reviewed else None,
        "anomalies": sum(bool(r.get("anomaly")) for r in recs),
        "avg_latency_ms": round(float(np.mean([r["timings_ms"]["total"] for r in recs
                                               if r.get("timings_ms") and "total" in r["timings_ms"]])), 1) if n else None,
    }


@app.get("/api/export/corrections.csv")
def export_corrections():
    """Human-verified labels -> next training set (active learning loop)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["image_key", "model_value", "human_value", "confidence", "status"])
    for r in store.list():
        if r.get("corrected_value"):
            w.writerow([r["image_key"], r.get("value"), r["corrected_value"], r.get("confidence"), r["status"]])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=corrections.csv"})
