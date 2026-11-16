"""Pick the review threshold from labelled photos.

    python scripts/calibrate.py PHOTOS_DIR data/labels.csv [--exclude data/label_issues.csv]

Reads every labelled photo with the production pipeline and prints, for a range of thresholds,
the share of photos accepted automatically and the precision on those. Choose the lowest
threshold that keeps the precision you need (we use >= 95 %).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.pipeline import MeterReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos")
    ap.add_argument("labels")
    ap.add_argument("--exclude", default=None, help="CSV of image_ids whose label is wrong")
    ap.add_argument("--out", default="reports/calibration.csv")
    a = ap.parse_args()

    labels = {r["image_id"]: r["reading"].zfill(3) for r in csv.DictReader(open(a.labels))}
    skip = set()
    if a.exclude:
        skip = {r[next(iter(r))] for r in csv.DictReader(open(a.exclude))}
    mr = MeterReader()
    rows = []
    for f in sorted(Path(a.photos).glob("*.jpg")):
        if f.stem not in labels or f.stem in skip:
            continue
        r = mr.read(cv2.imread(str(f)))
        rows.append(dict(image_id=f.stem, truth=labels[f.stem], pred=r.value or "",
                         conf=r.confidence, correct=int(r.value == labels[f.stem])))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    print(f"{n} labelled photos, {sum(r['correct'] for r in rows)} read correctly")
    for th in (0.60, 0.70, 0.75, 0.78, 0.80, 0.82):
        auto = [r for r in rows if r["conf"] >= th]
        ok = sum(r["correct"] for r in auto)
        print(f"threshold {th:.2f}: auto {len(auto):3d}/{n} ({len(auto)/n:4.0%})  "
              f"precision {ok}/{len(auto)} ({ok/max(1, len(auto)):.1%})")


if __name__ == "__main__":
    main()
