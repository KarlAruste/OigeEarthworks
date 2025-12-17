import os
import streamlit as st
import boto3
from botocore.client import Config

BUCKET = os.environ.get("R2_BUCKET", "").strip()

def get_s3():
    endpoint = os.environ.get("R2_ENDPOINT")
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    region = os.environ.get("R2_REGION", "auto")

    if not endpoint or not key_id or not secret or not BUCKET:
        st.error("R2 seaded puudu. Kontrolli R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.")
        st.stop()

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )

def safe_name(name: str) -> str:
    keep = " ._-()[]{}@+"
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in keep).strip()
    return cleaned.replace("..", ".") or "unnamed"

def project_prefix(project_name: str) -> str:
    p = safe_name(project_name)
    return p.rstrip("/") + "/"

def ensure_project_marker(s3, prefix: str):
    s3.put_object(Bucket=BUCKET, Key=prefix, Body=b"")

def list_files(s3, prefix: str):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    items = resp.get("Contents", [])
    files = []
    for it in items:
        key = it["Key"]
        if key.endswith("/"):
            continue
        files.append({"key": key, "name": key[len(prefix):], "size": it["Size"]})
    return sorted(files, key=lambda x: x["name"].lower())

def upload_file(s3, prefix: str, file):
    fname = safe_name(file.name)
    key = prefix + fname
    base, dot, ext = fname.partition(".")
    ext = (dot + ext) if dot else ""
    i = 2
    while True:
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            key = prefix + f"{base}_{i}{ext}"
            i += 1
        except Exception:
            break

    # IMPORTANT: boto3 wants bytes/bytearray/file-like, not memoryview
    s3.put_object(Bucket=BUCKET, Key=key, Body=file.getvalue())
    return key


def download_bytes(s3, key: str) -> bytes:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return obj["Body"].read()

def delete_key(s3, key: str):
    s3.delete_object(Bucket=BUCKET, Key=key)
