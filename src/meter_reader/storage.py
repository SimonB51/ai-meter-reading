"""Persistence for readings + photos.

Two interchangeable back-ends, picked with the STORAGE env var:
  * "local" (default): SQLite file + images on disk  -> runs anywhere, used for dev/tests
  * "aws":  images in S3, readings in DynamoDB       -> used in production on AWS
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from decimal import Decimal
from pathlib import Path

FIELDS = ["id", "meter_id", "created_at", "image_key", "value", "confidence", "status",
          "raw_ocr", "method", "angle_deg", "stage_conf", "timings_ms", "error",
          "corrected_value", "reviewed_at", "consumption", "anomaly"]


def new_id() -> str:
    return time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]


class LocalStorage:
    def __init__(self, root: str | Path = "data/runtime"):
        self.root = Path(root); (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "readings.db", check_same_thread=False)
        self.lock = threading.Lock()
        cols = ", ".join(f"{f} TEXT" for f in FIELDS[1:])
        self.db.execute(f"CREATE TABLE IF NOT EXISTS readings (id TEXT PRIMARY KEY, {cols})")
        self.db.commit()

    def put_image(self, key: str, data: bytes) -> None:
        (self.root / "images" / key).write_bytes(data)

    def get_image(self, key: str) -> bytes:
        return (self.root / "images" / key).read_bytes()

    def save(self, rec: dict) -> None:
        vals = [json.dumps(rec.get(f)) for f in FIELDS]
        with self.lock:
            self.db.execute(f"INSERT OR REPLACE INTO readings VALUES ({','.join('?'*len(FIELDS))})", vals)
            self.db.commit()

    def get(self, rid: str) -> dict | None:
        row = self.db.execute("SELECT * FROM readings WHERE id=?", [json.dumps(rid)]).fetchone()
        return dict(zip(FIELDS, map(json.loads, row))) if row else None

    def list(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM readings").fetchall()
        recs = [dict(zip(FIELDS, map(json.loads, r))) for r in rows]
        return sorted(recs, key=lambda r: r["created_at"], reverse=True)


class AwsStorage:
    """S3 bucket for photos, DynamoDB table (partition key 'id') for readings.
    Credentials come from the App Runner instance role - never from code."""

    def __init__(self, bucket: str, table: str, region: str | None = None):
        import boto3
        self.s3 = boto3.client("s3", region_name=region)
        self.table = boto3.resource("dynamodb", region_name=region).Table(table)
        self.bucket = bucket

    def put_image(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=f"photos/{key}", Body=data, ContentType="image/jpeg")

    def get_image(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=f"photos/{key}")["Body"].read()

    @staticmethod
    def _enc(rec: dict) -> dict:
        return json.loads(json.dumps(rec), parse_float=Decimal)

    @staticmethod
    def _dec(item: dict) -> dict:
        return json.loads(json.dumps(item, default=lambda d: float(d)))

    def save(self, rec: dict) -> None:
        self.table.put_item(Item={k: v for k, v in self._enc(rec).items() if v is not None})

    def get(self, rid: str) -> dict | None:
        it = self.table.get_item(Key={"id": rid}).get("Item")
        return {f: self._dec(it).get(f) for f in FIELDS} if it else None

    def list(self) -> list[dict]:
        items, kw = [], {}
        while True:  # small demo table -> a scan is fine
            resp = self.table.scan(**kw)
            items += resp["Items"]
            if "LastEvaluatedKey" not in resp:
                break
            kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        recs = [{f: self._dec(i).get(f) for f in FIELDS} for i in items]
        return sorted(recs, key=lambda r: r["created_at"], reverse=True)


def make_storage():
    if os.getenv("STORAGE", "local") == "aws":
        return AwsStorage(os.environ["S3_BUCKET"], os.environ["DDB_TABLE"], os.getenv("AWS_REGION"))
    return LocalStorage(os.getenv("LOCAL_DATA_DIR", "data/runtime"))
