from pathlib import Path

import cv2
import pytest

from meter_reader import MeterReader

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def reader():
    return MeterReader()


@pytest.mark.parametrize("name,expected", [("1ec531.jpg", "926"), ("316c0e.jpg", "883")])
def test_reads_known_meters(reader, name, expected):
    r = reader.read(cv2.imread(str(FIX / name)))
    assert r.value == expected
    assert r.status == "auto"
    assert 0 < r.confidence <= 1


def test_rotation_invariance(reader):
    img = cv2.rotate(cv2.imread(str(FIX / "1ec531.jpg")), cv2.ROTATE_90_CLOCKWISE)
    assert reader.read(img).value == "926"


def test_garbage_image_fails_gracefully(reader):
    import numpy as np
    r = reader.read(np.full((480, 640, 3), 127, np.uint8))
    assert r.value is None and r.status in ("failed", "review")
