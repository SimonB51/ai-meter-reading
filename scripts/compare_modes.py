"""Compare pipeline variants on the labelled test photos.
usage: python scripts/compare_modes.py data/eval data/labels.csv data/label_issues.csv
"""
import csv, statistics, sys, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meter_reader import MeterReader  # noqa: E402

img_dir, labels_csv, issues_csv = map(Path, sys.argv[1:4])
labels = {r["image_id"]: r["reading"] for r in csv.DictReader(open(labels_csv))}
bad = {r["image_id"] for r in csv.DictReader(open(issues_csv))}
photos = [p for p in sorted(img_dir.glob("*.jpg")) if p.stem in labels and p.stem not in bad]
imgs = {p.stem: cv2.imread(str(p)) for p in photos}

out = Path(__file__).resolve().parents[1] / "reports"; out.mkdir(exist_ok=True)
summary = []
for mode in ["yolo", "obb", "hybrid"]:
    R = MeterReader(mode=mode)
    rows = []
    for i, img in imgs.items():
        t = time.perf_counter(); r = R.read(img); ms = (time.perf_counter() - t) * 1000
        rows.append({"image_id": i, "truth": labels[i], "pred": r.value or "", "conf": r.confidence,
                     "method": r.method, "correct": int(r.value == labels[i]), "ms": ms})
    with open(out / f"eval_{mode}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(rows); acc = sum(r["correct"] for r in rows) / n
    auto = [r for r in rows if r["conf"] >= 0.5 and r["pred"]]
    summary.append((mode, n, acc, len(auto) / n, sum(r["correct"] for r in auto) / max(1, len(auto)),
                    statistics.median(r["ms"] for r in rows)))
print(f"{'mode':8} {'n':>3} {'exact':>7} {'auto@0.5':>9} {'acc_auto':>9} {'ms':>6}")
for m, n, a, au, aa, ms in summary:
    print(f"{m:8} {n:3d} {a:7.1%} {au:9.1%} {aa:9.1%} {ms:6.0f}")
