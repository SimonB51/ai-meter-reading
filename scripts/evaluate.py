"""Evaluate the pipeline on labelled photos.

usage: python scripts/evaluate.py data/eval data/labels.csv
Outputs: exact-match accuracy, automation rate / accuracy at several review thresholds,
and a per-image CSV in reports/eval_results.csv.
"""
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader import MeterReader  # noqa: E402

img_dir, labels_csv = Path(sys.argv[1]), Path(sys.argv[2])
labels = {r["image_id"]: r["reading"] for r in csv.DictReader(open(labels_csv))}
reader = MeterReader()

rows = []
for p in sorted(img_dir.glob("*.jpg")):
    if p.stem not in labels:
        continue
    r = reader.read(cv2.imread(str(p)))
    rows.append({"image_id": p.stem, "truth": labels[p.stem], "pred": r.value or "",
                 "correct": int(r.value == labels[p.stem]), "confidence": r.confidence,
                 "method": r.method, "angle": r.angle_deg, "raw_ocr": r.raw_ocr,
                 "ms": r.timings_ms.get("total"), "error": r.error or ""})
    print(f"{p.stem}  truth={labels[p.stem]}  pred={r.value}  conf={r.confidence:.2f}  {r.method}  raw={r.raw_ocr}")

out = Path(__file__).resolve().parents[1] / "reports"
out.mkdir(exist_ok=True)
with open(out / "eval_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

n = len(rows)
acc = sum(r["correct"] for r in rows) / n
print(f"\nN={n}  exact-match accuracy (last 3 digits) = {acc:.1%}")
print("threshold  automated  accuracy_on_automated")
for th in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    auto = [r for r in rows if r["confidence"] >= th and r["pred"]]
    a = sum(r["correct"] for r in auto) / len(auto) if auto else float("nan")
    print(f"  {th:.1f}       {len(auto)/n:6.1%}     {a:6.1%}")
