"""Train a small CNN that reads the last 3 black digits from a straightened band.
usage: python scripts/train_digits.py <bands_dir> <out.pt>
"""
import csv, random, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn

H, W = 48, 192
torch.manual_seed(0); random.seed(0); np.random.seed(0)
torch.set_num_threads(2)


def prep(band: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (2, 8)).apply(g)
    return g.astype(np.float32) / 255.0


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 128))   # -> 128 x 3 x 12
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleList([nn.Linear(128 * 3 * 12, 10) for _ in range(3)])

    def forward(self, x):
        z = self.drop(self.f(x).flatten(1))
        return torch.stack([h(z) for h in self.heads], 1)    # B x 3 x 10


def augment(band):
    h, w = band.shape[:2]
    # random crop jitter: lose up to 8% on the left / 3% elsewhere (right side holds the digits we need)
    l, r = int(random.uniform(0, .08) * w), int(random.uniform(0, .03) * w)
    t, b = int(random.uniform(0, .06) * h), int(random.uniform(0, .06) * h)
    band = band[t:h - b or h, l:w - r or w]
    m = cv2.getRotationMatrix2D((band.shape[1] / 2, band.shape[0] / 2), random.uniform(-4, 4), random.uniform(.95, 1.05))
    band = cv2.warpAffine(band, m, (band.shape[1], band.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    g = prep(band)
    g = np.clip(g * random.uniform(.7, 1.3) + random.uniform(-.15, .15), 0, 1)
    if random.random() < .3:
        g = cv2.GaussianBlur(g, (3, 3), random.uniform(.3, 1.2))
    if random.random() < .3:
        g = 1 - g if random.random() < .1 else g
    return g


def main():
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader(open(d / "labels.csv")))
    random.shuffle(rows)
    nval = len(rows) // 6
    val, tr = rows[:nval], rows[nval:]
    load = lambda rs: [(cv2.imread(str(d / f"{r['image_id']}.jpg")), [int(c) for c in r["reading"]]) for r in rs]
    tr, val = load(tr), load(val)
    Xv = torch.tensor(np.stack([prep(b) for b, _ in val]))[:, None]; Yv = torch.tensor([y for _, y in val])
    net = DigitNet(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    epochs = 60; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * ((len(tr) + 31) // 32))
    best = (-1, None)
    for ep in range(epochs):
        net.train(); random.shuffle(tr)
        for k in range(0, len(tr), 32):
            bt = tr[k:k + 32]
            x = torch.tensor(np.stack([augment(b) for b, _ in bt]))[:, None]; y = torch.tensor([y for _, y in bt])
            loss = nn.functional.cross_entropy(net(x).reshape(-1, 10), y.reshape(-1), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            p = net(Xv).argmax(-1)
        digit = (p == Yv).float().mean().item(); full = (p == Yv).all(1).float().mean().item()
        if full >= best[0]:
            best = (full, {k: v.clone() for k, v in net.state_dict().items()})
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"epoch {ep+1:2d} loss {loss.item():.3f} val digit-acc {digit:.3f} val 3-digit exact {full:.3f}", flush=True)
    torch.save({"state_dict": best[1], "H": H, "W": W}, out)
    print("best val exact", best[0], "saved", out)


if __name__ == "__main__":
    main()
