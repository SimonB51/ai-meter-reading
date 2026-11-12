"""Water-meter reading pipeline.

photo -> [YOLO #1] counter window -> [YOLO #2] black/red digit blocks
      -> rotation (black -> red axis made horizontal) -> crop black block
      -> [PP-OCR recogniser via RapidOCR, 4 test-time variants] -> last 3 digits + confidence

The models are the ones trained in the original Colab project:
  * zone_detector.pt        YOLOv8n, 1 class  ("chiffres": the counter window)
  * orientation_detector.pt YOLOv8n, 2 classes (0 = "noir" integer digits, 1 = "rouge" decimals)
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# letters the recogniser sometimes outputs for odometer digits
CONFUSABLES = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "C": "0", "U": "0", "G": "6",
                             "b": "6", "I": "1", "l": "1", "i": "1", "|": "1", "S": "5", "s": "5",
                             "B": "8", "Z": "2", "z": "2", "T": "7", "g": "9", "q": "9"})
MODELS = ROOT / "models"


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Reading:
    value: str | None              # last 3 digits, e.g. "344"; None if unreadable
    confidence: float              # 0..1, product-style overall score
    status: str                    # "auto" | "review" | "failed"
    raw_ocr: str = ""
    stage_conf: dict = field(default_factory=dict)
    angle_deg: float | None = None
    method: str = ""               # "black_box" | "legacy_fixed_crop"
    timings_ms: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _best_box(result, cls_id: int | None = None) -> Box | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    for b in boxes:
        if cls_id is not None and int(b.cls[0]) != cls_id:
            continue
        c = float(b.conf[0])
        if best is None or c > best.conf:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            best = Box(x1, y1, x2, y2, c)
    return best


def rotate_keep_all(img: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate around the centre WITHOUT cutting corners (canvas is enlarged).
    Returns the rotated image and the 2x3 affine matrix used."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, m


def transform_box(box: Box, m: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.array([[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]])
    pts = np.hstack([pts, np.ones((4, 1))]) @ m.T
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def cut_red_columns(img: np.ndarray) -> np.ndarray:
    """Legacy step from the original notebook: cut the image at the first
    column (from the left) that contains clearly red pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
    cols = np.where(red.sum(axis=0) > 255 * 3)[0]
    cols = cols[cols > img.shape[1] * 0.3]   # ignore noise on the far left
    return img[:, : cols[0]] if len(cols) else img


class MeterReader:
    def __init__(self, models_dir: Path = MODELS, review_threshold: float = 0.5,
                 device: str = "cpu"):
        from ultralytics import YOLO
        from rapidocr_onnxruntime import RapidOCR

        self.zone_model = YOLO(str(models_dir / "yolo" / "zone_detector.pt"))
        self.orient_model = YOLO(str(models_dir / "yolo" / "orientation_detector.pt"))
        self.ocr = RapidOCR()  # PP-OCR ONNX models ship inside the pip wheel
        self.review_threshold = review_threshold
        # NB: the weights' metadata says {0: 'rouge', 1: 'noir'} but the training labels
        # were written with noir=0 / rouge=1 (see the Colab notebook). Verified visually:
        # class 0 boxes the black integer digits. So we use ids, not names.
        self.BLACK, self.RED = 0, 1

    # ---------- stages ----------
    def detect_zone(self, img: np.ndarray) -> Box | None:
        r = self.zone_model.predict(img, conf=0.25, verbose=False, device="cpu")[0]
        return _best_box(r)

    def detect_blocks(self, crop: np.ndarray) -> tuple[Box | None, Box | None]:
        r = self.orient_model.predict(crop, conf=0.25, verbose=False, device="cpu")[0]
        return _best_box(r, self.BLACK), _best_box(r, self.RED)

    @staticmethod
    def _variants(img: np.ndarray):
        """Test-time augmentation: the same crop seen 4 slightly different ways."""
        h, w = img.shape[:2]
        yield img
        yield img[int(.08 * h):int(.92 * h), :]
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        yield cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        yield img[:, int(.03 * w):int(.97 * w)]

    def read_digits(self, img: np.ndarray) -> tuple[str, float, float]:
        """Returns (digits, ocr_score, agreement).
        agreement = share of the 4 variants that give the same last-3 digits."""
        outs = []
        for v in self._variants(img):
            text, score = self.ocr.text_recognizer([v])[0][0]
            d = re.sub(r"\D", "", text.translate(CONFUSABLES))
            if len(d) >= 3:
                outs.append((d, float(score)))
        if not outs:
            return "", 0.0, 0.0
        best, score = max(outs, key=lambda o: o[1])
        agreement = sum(o[0][-3:] == best[-3:] for o in outs) / 4
        return best, score, agreement

    # ---------- full pipeline ----------
    def read(self, img: np.ndarray) -> Reading:
        t0 = time.perf_counter(); tm = {}
        zone = self.detect_zone(img); tm["zone"] = (time.perf_counter() - t0) * 1000
        if zone is None:
            return Reading(None, 0.0, "failed", error="counter window not found", timings_ms=tm)
        h, w = img.shape[:2]
        crop = img[max(0, int(zone.y1)):min(h, int(zone.y2)), max(0, int(zone.x1)):min(w, int(zone.x2))]

        t = time.perf_counter()
        black, red = self.detect_blocks(crop); tm["orientation"] = (time.perf_counter() - t) * 1000
        stage = {"zone": round(zone.conf, 3),
                 "black": round(black.conf, 3) if black else 0.0,
                 "red": round(red.conf, 3) if red else 0.0}

        angle = None
        if black and red:
            (bx, by), (rx, ry) = black.center, red.center
            angle = math.degrees(math.atan2(ry - by, rx - bx))
            rot, m = rotate_keep_all(crop, angle)
            x1, y1, x2, y2 = transform_box(black, m)
            digits_img = rot[max(0, y1):y2, max(0, x1):x2]   # tight crop works best for OCR
            method = "black_box"
        else:
            # fallback = original notebook logic: fixed crop on a 256x256 resize, then cut red
            rs = cv2.resize(crop, (256, 256))
            digits_img = cut_red_columns(rs[83:173, 18:238])
            method = "legacy_fixed_crop"

        t = time.perf_counter()
        digits, ocr_conf, agree = self.read_digits(digits_img); tm["ocr"] = (time.perf_counter() - t) * 1000
        stage["ocr"] = round(ocr_conf, 3); stage["ocr_agreement"] = agree
        tm["total"] = (time.perf_counter() - t0) * 1000

        if len(digits) < 3:
            return Reading(None, 0.0, "review", raw_ocr=digits, stage_conf=stage,
                           angle_deg=angle, method=method, timings_ms=tm,
                           error="fewer than 3 digits read")
        value = digits[-3:]
        conf = zone.conf * (black.conf if black else 0.5) * ocr_conf * agree
        status = "auto" if conf >= self.review_threshold else "review"
        return Reading(value, round(conf, 4), status, raw_ocr=digits, stage_conf=stage,
                       angle_deg=None if angle is None else round(angle, 1),
                       method=method, timings_ms={k: round(v, 1) for k, v in tm.items()})

    def debug_images(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """Intermediate images for the UI / README illustrations."""
        out = {"input": img}
        zone = self.detect_zone(img)
        if zone is None:
            return out
        crop = img[int(zone.y1):int(zone.y2), int(zone.x1):int(zone.x2)]
        out["zone"] = crop
        black, red = self.detect_blocks(crop)
        if black and red:
            (bx, by), (rx, ry) = black.center, red.center
            rot, m = rotate_keep_all(crop, math.degrees(math.atan2(ry - by, rx - bx)))
            out["rotated"] = rot
            x1, y1, x2, y2 = transform_box(black, m)
            out["digits"] = rot[max(0, y1):y2, max(0, x1):x2]
        return out
