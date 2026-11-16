"""Small CNN that reads the last 3 black digits of a straightened band (trained by scripts/train_digits.py).

Why: the generic OCR has never seen water-meter drums; this model is trained on ~540 of our
own straightened bands (labels from the project's CSV), with the test photos held out.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

H, W = 48, 192


def prep(band: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (2, 8)).apply(g)
    return g.astype(np.float32) / 255.0


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 128))   # -> 128 x 3 x 12
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleList([nn.Linear(128 * 3 * 12, 10) for _ in range(3)])

    def forward(self, x):
        z = self.drop(self.f(x).flatten(1))
        return torch.stack([h(z) for h in self.heads], 1)    # B x 3 x 10


class DigitReader:
    def __init__(self, weights: Path):
        ck = torch.load(weights, map_location="cpu")
        self.net = DigitNet()
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()

    @torch.no_grad()
    def read(self, band: np.ndarray) -> tuple[str, float]:
        """Returns (3 digits, confidence = product of the 3 digit probabilities)."""
        x = torch.tensor(prep(band))[None, None]
        p = self.net(x).softmax(-1)[0]           # 3 x 10
        conf, idx = p.max(-1)
        return "".join(str(int(i)) for i in idx), float(conf.prod())
