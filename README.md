# AI Meter Reading

**A water-meter photo goes in. The reading, a confidence score and a human check when the model is unsure come out.**

**Live demo: https://meter-reader-764094047053.europe-west1.run.app** (Google Cloud Run). AWS deployment (ECS Express Mode + S3 + DynamoDB) is scripted in `infra/`.

> Started as a one-week team project in school (original notebook: [QuiterieGuignard/AI-for-meters-reading-](https://github.com/QuiterieGuignard/AI-for-meters-reading-)). This repo is my rebuild of it as a deployed product: a clean pipeline, a held-out evaluation, an API with human review, and AWS infrastructure.

Agents still read water meters by hand and type in the numbers, which makes errors and costs time. This service reads the meter from a phone photo. It accepts the readings it is sure about and sends the others to a review queue. Each human correction is saved as a new training label.

![pipeline](docs/img/pipeline.png)

| Read a meter | Review queue | Dashboard |
|---|---|---|
| ![](docs/img/ui_read.png) | ![](docs/img/ui_review.png) | ![](docs/img/ui_dash.png) |

---

## How it works

| Stage | Model / method | Why |
|---|---|---|
| 1. Find the counter window | **YOLOv8n #1**, fine-tuned from COCO weights on a small set of photos hand-labelled in Roboflow, then retrained on its own confident predictions (**pseudo-labelling**, conf ≥ 0.3). Dataset: 795 meter photos. | Photos are taken from any distance and angle. Detection isolates the region of interest. |
| 2. Orientation | **YOLOv8n-OBB** (rotated boxes, trained on my annotations) + the original **YOLOv8n #2** detect the *black* (m³) and *red* (litres) digit blocks | The black-to-red vector gives the meter's orientation, so photos taken sideways or upside down are fixed with one rotation (`atan2`). |
| 3. Rotate + crop | Rotate on an enlarged canvas so no corner is lost. Crop the rotated black box. | Only the integer digits go to the OCR. The red litres are dropped. |
| 4. Read | **PP-OCR recogniser** (ONNX, via RapidOCR) on **4 test-time variants** of each crop (original, trimmed top/bottom, contrast-boosted, trimmed sides) | The share of variants that agree is a cheap, honest signal of how sure the reading is. |
| 5. Decide | `confidence = OCR score` of the chosen crop (× 0.8 when no red drum confirms which way up the meter is). Readings ≥ 0.78 are auto-accepted, 0.60–0.78 are shown as "probably right" and checked, the rest go to review. A consumption check per meter (robust z-score, modulo-1000 rollover) also sends suspicious jumps to review. | Wrong readings are costly, so the system stays precise and a human handles the uncertain ones. |

## Results

Tested on 70 labelled photos the OCR never saw. 6 of those labels contradict the photo ([`data/label_issues.csv`](data/label_issues.csv)) and are excluded below.

![trade-off](docs/img/threshold_tradeoff.png)

Progression on the same 64 test photos (labels verified):

| Version | Exact match | Auto-accepted at threshold 0.5 | Accuracy on auto-accepted | Median latency (CPU) |
|---|---|---|---|---|
| Colab version rebuilt as-is (EasyOCR, first 30 photos) | ≈13 % | – | – | – |
| Axis-aligned YOLO crop, PP-OCR | 67 % | 39 % | 96 % | 170 ms |
| + 2nd detection pass on the straightened image | 83 % | 50 % | 94 % | 240 ms |
| Oriented-box detector (YOLOv8n-OBB) only | 81 % | 62 % | 95 % | 190 ms |
| **Hybrid: OBB + 2nd pass, keep the crop the OCR is most sure of (production)** | **84 %** | **66 %** | **95 %** | 290 ms |
| Experiment, not shipped: + small digit CNN as a second reader | 88 % | 67 % | 98 % | 320 ms |

The OBB detector (`models/yolo/orientation_obb.pt`) is a YOLOv8n-OBB fine-tuned on 164 crops I annotated with rotated polygons (black band / red band), with flips disabled (a mirror would put red on the left) and ±180° rotation augmentation. Validation mAP50 = 0.995, mAP50-95 = 0.87. None of the 70 test photos were in its training set (`scripts/train_obb.py`, `scripts/compare_modes.py`).

**Why there is no CNN in production.** A CNN that reads the raw photo needs far more data than we have (the ResNet18 tried in the team project reached ~1 %). Once the pipeline has isolated and straightened the digit band, the problem is much smaller, so I tried a small CNN (`models/digit_reader.pt`, 0.7 M parameters, 651 training bands) as a second reader. On the 64 test photos it fixed 3 readings, broke 1 and caught 1 wrong auto-accepted reading: +2 photos, which is within the noise of a test set this size, and on its own it is weaker than the OCR (75 % vs 84 %). It is kept in the repo (`use_digit_reader=True`, `scripts/train_digits.py`) and switched off. It becomes worth it once the human corrections collected by the review queue give it more training data.


**Calibration on 189 labelled photos (production rule).** The tables above use the first confidence formula, `zone × box × OCR × agreement`, on the 64 test photos. Re-checking on all 189 labelled raw photos showed two problems: the product dragged clean photos under the threshold (four stages at 0.83 give 0.47), and on the 125 photos outside the test set its auto-accepted readings were only 83 % right, so the 95 % above was partly luck of a small test set. Using the OCR score alone separates right from wrong better (`scripts/calibrate.py`, `reports/calibration.csv`):

| Rule | Auto-accepted | Precision on auto-accepted |
|---|---|---|
| Old: product of 4 scores ≥ 0.5 | 62 % | 87 % |
| **New: OCR score ≥ 0.78 (production)** | **52 %** | **95 %** (test photos 97 %, other photos 93 %) |
| OCR score ≥ 0.80 | 38 % | 97 % |

The threshold was chosen on the same 189 photos, so treat 95 % as an estimate; the per-split numbers above show it holds on both halves.

Main failure modes: digits caught mid-turn (e.g. 7↔8), heavy blur, and very small or dirty counters. Caveat: the eval photos come from the same pool the YOLO models were trained on, so the detection stages may look better here than on new meters. The OCR stage was never trained on this data.

**What changed vs. the first (Colab) version**
- The orientation weights store their class names swapped (`{0:'rouge', 1:'noir'}`) while the labels were written with `noir=0`. Using the names silently turned images upside down. The fix uses the class ids, checked by eye.
- A fixed crop `img[83:173, 18:238]` plus red-pixel cutting was replaced by a crop on the detected black box after rotation.
- EasyOCR (≈13 % here) was replaced by the PP-OCR recogniser with test-time augmentation (≈60 %+). The ResNet18 / 3-head CNN tried before reached ~1 %.
- The axis-aligned orientation boxes were loose on tilted meters (they swallowed background and red digits). I annotated 164 crops with rotated polygons and trained an oriented-box (OBB) detector: tight crops at any angle, +16 points of automation.
- Validation was done on the training set. It now uses a separate labelled evaluation (`scripts/evaluate.py`).

## Architecture (AWS)

![architecture](docs/img/architecture.png)

- **ECS Express Mode** (Fargate + Application Load Balancer + autoscaling 1–2 tasks, 1 vCPU / 2 GB). It runs the Docker image stored in **ECR**.
- **S3** stores the photos (private bucket, public access blocked). **DynamoDB** stores readings and human corrections (on-demand).
- **CloudWatch Logs** get one structured log line per reading and per review.
- **IAM**: the task role can only `PutObject`/`GetObject` on `photos/*` of its own bucket and `PutItem`/`GetItem`/`Scan` on its own table. No keys live in the code or the image.
- The same code runs locally with `STORAGE=local` (SQLite + disk), which keeps dev and tests free.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/readings` | multipart `file` (+ `meter_id`) → reading, confidence, status, consumption, anomaly |
| `GET` | `/api/readings?status=review` | review queue |
| `POST` | `/api/readings/{id}/review` | `{"value": "186"}` → `confirmed` / `corrected` |
| `GET` | `/api/stats` | automation rate, accuracy on reviewed, alerts, latency |
| `GET` | `/api/export/corrections.csv` | human-verified labels for the next training run |
| `GET` | `/health` | load-balancer health check |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pytest -q                                              # 12 tests
uvicorn meter_reader.api:app --app-dir src --port 8080 # open http://localhost:8080
```
or `docker build -t meter-reading . && docker run -p 8080:8080 meter-reading`

Evaluate on your own labelled photos (`image_id,reading` CSV):
```bash
python scripts/evaluate.py path/to/photos data/labels.csv
```

## Deploy to Google Cloud Run (live demo)

From Cloud Shell, in the repo:
```bash
gcloud builds submit --config cloudbuild.yaml --region=europe-west1
gcloud run deploy meter-reader --image europe-west1-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/cloud-run-source-deploy/meter-reader \
  --region europe-west1 --memory 2Gi --cpu 1 --cpu-boost --min-instances 1 --allow-unauthenticated
```
`cloudbuild.yaml` pulls the Python base image from Google's Docker Hub mirror (ECR Public rate-limits Google's builders).

## Deploy to AWS

**CI/CD (recommended).** Every push to `main` runs the tests and builds the image (`ci.yml`), then builds, pushes to ECR and deploys to ECS Express Mode (`deploy.yml`). GitHub authenticates to AWS with **OIDC**, so no AWS keys are stored anywhere. One-time setup: an IAM OIDC provider for `token.actions.githubusercontent.com`, a role that only this repo's `main` branch can assume, and the repo variable `AWS_ROLE_ARN`. Manual runs: *Actions → deploy → Run workflow* with `deploy`, `status` or `destroy`.

**From a laptop** (AWS CLI v2 + Docker):
```bash
./infra/deploy.sh    # S3 + DynamoDB + IAM + ECR + build/push + ECS Express service (updates if it exists)
./infra/status.sh    # status + public HTTPS URL
./infra/destroy.sh   # delete everything (stops all costs)
```
Cost when always on: about $50/month (Fargate 1 vCPU/2 GB ≈ $36 + load balancer ≈ $16). S3, DynamoDB and logs cost cents at demo scale.

## Repo map
```
src/meter_reader/pipeline.py   detection → rotation → crop → OCR → confidence
src/meter_reader/api.py        FastAPI service + review workflow
src/meter_reader/storage.py    local (SQLite) / AWS (S3 + DynamoDB) back-ends
src/meter_reader/anomaly.py    consumption check per meter
src/meter_reader/static/       single-page UI (no framework)
models/yolo/                   the two YOLOv8n weights (6 MB each)
scripts/evaluate.py            accuracy + automation-rate report
infra/                         AWS deploy / update / destroy scripts
tests/                         pipeline, API and anomaly tests
```

## Next steps
1. **Close the loop.** Use the exported human corrections to grow the training set, retrain the digit CNN on it (mid-turn digits are the main remaining error), re-run `compare_modes.py`, and switch it on only if accuracy at the same automation rate goes up clearly.
2. Batch mode: an S3 upload event triggers an SQS queue and a worker, for the thousands of photos a utility receives each day.
3. Authentication (Cognito) and per-agent audit trail. Meter-ID / GPS from the photo's EXIF.
4. Drift monitoring: a CloudWatch metric on the automation rate, with an alarm when it drops.
