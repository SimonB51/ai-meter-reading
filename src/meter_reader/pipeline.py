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
    band_jpg: str | None = None    # base64 JPEG of the straightened digits the readers saw (for the UI)

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


@dataclass
class OBox:
    """Oriented box: 4 corners (x, y) in image pixels + confidence."""
    pts: np.ndarray
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        c = self.pts.mean(axis=0)
        return float(c[0]), float(c[1])


def _best_obox(result, cls_id: int) -> OBox | None:
    obb = result.obb
    if obb is None or len(obb) == 0:
        return None
    best = None
    for pts, c, k in zip(obb.xyxyxyxy.cpu().numpy(), obb.conf.cpu().numpy(), obb.cls.cpu().numpy()):
        if int(k) == cls_id and (best is None or c > best.conf):
            best = OBox(pts.astype(float), float(c))
    return best


class MeterReader:
    def __init__(self, models_dir: Path = MODELS, review_threshold: float = 0.5,
                 device: str = "cpu", mode: str | None = None,
                 use_digit_reader: bool = True):
        """mode: 'obb'    -> oriented-box detector (tight crop, any angle)
                 'yolo'   -> original axis-aligned detector + 2nd pass
                 'hybrid' -> read the crops of both and keep the most confident (default if OBB weights exist)"""
        from ultralytics import YOLO
        from rapidocr_onnxruntime import RapidOCR

        self.zone_model = YOLO(str(models_dir / "yolo" / "zone_detector.pt"))
        self.orient_model = YOLO(str(models_dir / "yolo" / "orientation_detector.pt"))
        obb_path = models_dir / "yolo" / "orientation_obb.pt"
        self.obb_model = YOLO(str(obb_path)) if obb_path.exists() else None
        self.mode = mode or ("hybrid" if self.obb_model else "yolo")
        dr_path = models_dir / "digit_reader.pt"
        self.digit_reader = None
        if dr_path.exists() and use_digit_reader:
            from .digit_reader import DigitReader
            self.digit_reader = DigitReader(dr_path)
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

    def detect_oblocks(self, crop: np.ndarray) -> tuple[OBox | None, OBox | None]:
        r = self.obb_model.predict(crop, conf=0.25, verbose=False, device="cpu", imgsz=416)[0]
        return _best_obox(r, 0), _best_obox(r, 1)   # data.yaml: 0 = noir, 1 = rouge

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
        candidates: list[tuple[str, np.ndarray]] = []
        if self.mode in ("obb", "hybrid") and self.obb_model is not None:
            # take the zone with a margin from the full photo: on tilted meters the
            # black band's corners often stick out of YOLO #1's axis-aligned box
            mx, my = int(0.12 * (zone.x2 - zone.x1)), int(0.12 * (zone.y2 - zone.y1))
            ocrop = img[max(0, int(zone.y1) - my):min(h, int(zone.y2) + my),
                        max(0, int(zone.x1) - mx):min(w, int(zone.x2) + mx)]
            ob, orr = self.detect_oblocks(ocrop)
            if ob is not None:
                stage["black_obb"] = round(ob.conf, 3)
                if orr is not None:   # orientation from the black -> red vector
                    (bx, by), (rx, ry) = ob.center, orr.center
                    angles = [math.degrees(math.atan2(ry - by, rx - bx))]
                else:                 # no red drum on this meter: the long side gives the angle up to 180 deg,
                    p = ob.pts        # so we try both and let the readers decide which way up is legible
                    e = max([p[1] - p[0], p[2] - p[1]], key=lambda v: np.hypot(*v))
                    a0 = math.degrees(math.atan2(e[1], e[0]))
                    angles = [a0, a0 + 180]
                    stage["orientation_unconfirmed"] = True
                for k, a in enumerate(angles):
                    rot, m = rotate_keep_all(ocrop, a)
                    q = np.hstack([ob.pts, np.ones((4, 1))]) @ m.T
                    x1, y1 = q.min(axis=0).astype(int); x2, y2 = q.max(axis=0).astype(int)
                    candidates.append(("obb" if k == 0 else "obb_flipped", rot[max(0, y1):y2, max(0, x1):x2]))
                angle = angles[0]
        if self.mode in ("yolo", "hybrid") and black and red:
            (bx, by), (rx, ry) = black.center, red.center
            angle = math.degrees(math.atan2(ry - by, rx - bx))
            rot, m = rotate_keep_all(crop, angle)
            # candidate A: the black box detected before rotation, carried through the rotation
            x1, y1, x2, y2 = transform_box(black, m)
            candidates.append(("black_box", rot[max(0, y1):y2, max(0, x1):x2]))
            # candidate B: re-detect on the straightened image -> tighter box when the meter was tilted
            black2, _ = self.detect_blocks(rot)
            if black2:
                candidates.append(("black_box_2nd_pass",
                                   rot[int(black2.y1):int(black2.y2), int(black2.x1):int(black2.x2)]))
        if not candidates:
            # fallback = original notebook logic: fixed crop on a 256x256 resize, then cut red
            rs = cv2.resize(crop, (256, 256))
            candidates.append(("legacy_fixed_crop", cut_red_columns(rs[83:173, 18:238])))

        t = time.perf_counter()
        # read every candidate crop, keep the one the OCR is most sure about
        reads = [(name, *self.read_digits(im)) for name, im in candidates if im.size]
        method, digits, ocr_conf, agree = max(reads, key=lambda r: r[2] * r[3]) if reads else ("none", "", 0.0, 0.0)
        tm["ocr"] = (time.perf_counter() - t) * 1000
        stage["ocr"] = round(ocr_conf, 3); stage["ocr_agreement"] = agree

        # second opinion: our CNN trained on meter drums, run on the oriented-box band
        cnn = None
        cands = dict(candidates)
        band = cands.get(method.split("+")[0]) if method.startswith("obb") else cands.get("obb")
        if self.digit_reader is not None and band is not None and band.size:
            cnn = self.digit_reader.read(band)
            stage["cnn"] = cnn[0]; stage["cnn_conf"] = round(cnn[1], 3)
        tm["total"] = (time.perf_counter() - t0) * 1000

        det_conf = stage.get("black_obb", 0.5) if method.startswith("obb") else (black.conf if black else 0.5)
        ocr_value = digits[-3:] if len(digits) >= 3 else None
        ocr_score = zone.conf * det_conf * ocr_conf * agree if ocr_value else 0.0
        value, conf = ocr_value, ocr_score
        if cnn is not None:
            cnn_score = zone.conf * stage.get("black_obb", 0.5) * cnn[1]
            if cnn[0] == ocr_value:                    # two independent readers agree
                value, conf = ocr_value, 1 - (1 - ocr_score) * (1 - cnn_score)
                method += "+cnn_agree"
            elif cnn_score > ocr_score:               # disagree: trust the more confident, but flag doubt
                value, conf = cnn[0], cnn_score * 0.6
                method = "cnn"
            else:
                conf = ocr_score * 0.6
        if stage.get("orientation_unconfirmed"):
            conf *= 0.8               # no red drum to confirm which way up the meter is
        if value is None:
            return Reading(None, 0.0, "review", raw_ocr=digits, stage_conf=stage,
                           angle_deg=angle, method=method, timings_ms=tm,
                           error="fewer than 3 digits read")
        status = "auto" if conf >= self.review_threshold else "review"
        shown = band if method == "cnn" else cands.get(method.split("+")[0])
        band_jpg = None
        if shown is not None and shown.size:
            import base64
            s_ = 96 / max(1, shown.shape[0])
            small = cv2.resize(shown, None, fx=s_, fy=s_) if s_ < 1 else shown
            band_jpg = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
        return Reading(value, round(conf, 4), status, raw_ocr=digits, stage_conf=stage,
                       angle_deg=None if angle is None else round(angle, 1),
                       method=method, timings_ms={k: round(v, 1) for k, v in tm.items()}, band_jpg=band_jpg)

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
