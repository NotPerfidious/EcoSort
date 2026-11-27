import asyncio
import json
import logging
import os
from typing import Any, Dict

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from metrics import ACTIVE_LEARNING_UPLOADS_TOTAL

logger = logging.getLogger("ecosort.s3_ingestor")


class ActiveLearningIngestor:
    """Non-blocking uploader for low-confidence frames to AWS S3."""

    def __init__(self) -> None:
        self.bucket = os.getenv("S3_BUCKET_NAME", "").strip()
        self.region = os.getenv("AWS_REGION", "ap-south-1").strip()
        self.prefix = os.getenv(
            "ACTIVE_LEARNING_PREFIX", "active-learning/incoming"
        ).strip("/")
        has_keys = bool(
            os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        self._client = (
            boto3.client("s3", region_name=self.region)
            if (self.bucket and has_keys)
            else None
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    def _upload_sync(
        self, request_id: str, raw_bytes: bytes, metadata: Dict[str, Any]
    ) -> None:
        if self._client is None:
            ACTIVE_LEARNING_UPLOADS_TOTAL.labels(status="skipped_unconfigured").inc()
            logger.info(
                "[active-learning] S3 not configured; skipped upload for %s", request_id
            )
            return

        img_key = f"{self.prefix}/{request_id}.jpg"
        meta_key = f"{self.prefix}/{request_id}.json"
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=img_key,
                Body=raw_bytes,
                ContentType="image/jpeg",
            )
            self._client.put_object(
                Bucket=self.bucket,
                Key=meta_key,
                Body=json.dumps(metadata, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            ACTIVE_LEARNING_UPLOADS_TOTAL.labels(status="uploaded").inc()
        except (BotoCoreError, ClientError) as exc:
            ACTIVE_LEARNING_UPLOADS_TOTAL.labels(status="error").inc()
            logger.warning(
                "[active-learning] S3 upload failed for %s: %s", request_id, exc
            )

    async def ingest(
        self, request_id: str, raw_bytes: bytes, metadata: Dict[str, Any]
    ) -> None:
        await asyncio.to_thread(self._upload_sync, request_id, raw_bytes, metadata)
