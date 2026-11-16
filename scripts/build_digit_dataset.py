"""Turn the team's zone crops into straightened black-band images for the digit reader.
usage: python scripts/build_digit_dataset.py <crops_dir> <out_dir>
Uses the same OBB + rotation code as production, so train and inference see the same thing.
"""
import csv, math, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader.pipeline import MeterReader, rotate_keep_all  # noqa: E402

crops_dir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
labels = {r["image_id"]: r["reading"] for r in csv.DictReader(open(root / "data/labels.csv"))}
test = {p.stem for p in (root / "data/eval").glob("*.jpg")}
R = MeterReader(mode="yolo")
rows = []
for p in sorted(crops_dir.rglob("*_crop.jpg")):
    i = p.name.replace("_crop.jpg", "")
    if i not in labels or i in test:
        continue
    crop = cv2.imread(str(p))
    ob, orr = R.detect_oblocks(crop)
    if ob is None or orr is None:
        continue
    (bx, by), (rx, ry) = ob.center, orr.center
    rot, m = rotate_keep_all(crop, math.degrees(math.atan2(ry - by, rx - bx)))
    q = np.hstack([ob.pts, np.ones((4, 1))]) @ m.T
    x1, y1 = q.min(axis=0).astype(int); x2, y2 = q.max(axis=0).astype(int)
    band = rot[max(0, y1):y2, max(0, x1):x2]
    if band.size == 0:
        continue
    cv2.imwrite(str(out / f"{i}.jpg"), band)
    rows.append({"image_id": i, "reading": labels[i]})
with open(out / "labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image_id", "reading"]); w.writeheader(); w.writerows(rows)
print(len(rows), "bands written")
