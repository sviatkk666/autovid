"""Upload finished assets to Cloudflare R2 (S3-compatible) so the CRM can fetch
them by a plain https URL.

autovid holds the R2 credentials (its OWN .env); the CRM never gets them — it
only receives the resulting public URL. The bucket is served publicly via a
custom domain / r2.dev (R2_PUBLIC_URL); object keys are slug-scoped and
unguessable enough for pre-publish assets. (Harden to private + presigned GET
later if needed — the CRM side already only stores the URL.)

Env (autovid .env):
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_URL
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import env

_CONTENT_TYPE = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".srt": "text/plain",
    ".vtt": "text/vtt",
}


def _client():
    """Lazily build an S3 client pointed at R2 (boto3 is an optional dep)."""
    try:
        import boto3  # noqa: PLC0415 — lazy so the rest of autovid needs no boto3
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("boto3 not installed — `pip install boto3` to push to R2") from e
    account = env("R2_ACCOUNT_ID")
    if not account:
        raise RuntimeError("R2_ACCOUNT_ID not set — configure R2 in autovid .env")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def upload_file(local: Path, key: str, content_type: str | None = None) -> str:
    """PUT a local file to R2 under `key`; return its public https URL."""
    local = Path(local)
    if not local.exists():
        raise FileNotFoundError(f"asset not found: {local}")
    bucket = env("R2_BUCKET")
    public = env("R2_PUBLIC_URL").rstrip("/")
    if not bucket or not public:
        raise RuntimeError("R2_BUCKET / R2_PUBLIC_URL not set — configure R2 in autovid .env")
    ctype = content_type or _CONTENT_TYPE.get(local.suffix.lower(), "application/octet-stream")
    client = _client()
    with open(local, "rb") as f:
        client.put_object(Bucket=bucket, Key=key, Body=f, ContentType=ctype)
    url = f"{public}/{key}"
    print(f"[r2] {local.name} -> {url}", file=sys.stderr)
    return url
