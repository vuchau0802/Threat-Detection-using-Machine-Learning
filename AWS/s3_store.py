import hashlib
import logging
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/LogisticRegression.pkl")
DATASET_PATH = os.environ.get("DATASET_PATH", "data/cleaned_dataset.csv")
S3_BUCKET = os.environ.get("AWS_S3_BUCKET") or os.environ.get("S3_BUCKET", "")
S3_KEY = os.environ.get("AWS_S3_KEY", "threat-detection/artifacts.zip")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
ZIP_PATH = os.environ.get("ARTIFACT_ZIP_PATH", "threat-detection-artifacts.zip")


def _get_s3_client():
    """Return a boto3 S3 client using environment credentials."""
    if boto3 is None:
        raise RuntimeError(
            "boto3 is not installed. Run: pip install -r requirements.txt"
        )

    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _artifact_paths() -> list[Path]:
    return [Path(MODEL_PATH), Path(DATASET_PATH)]


def _zip_artifacts(zip_path: str = ZIP_PATH) -> None:
    """Zip the model and cleaned dataset using their project-relative paths."""
    missing = [str(path) for path in _artifact_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {', '.join(missing)}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in _artifact_paths():
            zf.write(path, path.as_posix())

    size_mb = Path(zip_path).stat().st_size / (1024**2)
    logger.info("Zipped artifacts -> %s (%.1f MB)", zip_path, size_mb)


def _unzip_artifacts(zip_path: str = ZIP_PATH) -> None:
    """Extract artifacts into the current project/container working directory."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = Path(member.filename)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(".")

    logger.info("Extracted artifacts from %s", zip_path)


def _md5_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_to_s3(
    bucket: str = S3_BUCKET,
    s3_key: str = S3_KEY,
    zip_path: str = ZIP_PATH,
) -> bool:
    """Zip local project artifacts and upload them to S3."""
    if not bucket:
        logger.error("Set AWS_S3_BUCKET or S3_BUCKET before uploading.")
        return False

    try:
        _zip_artifacts(zip_path)
        size = Path(zip_path).stat().st_size

        print(f"Uploading to s3://{bucket}/{s3_key} ...")
        s3 = _get_s3_client()
        s3.upload_file(
            zip_path,
            bucket,
            s3_key,
            ExtraArgs={
                "Metadata": {
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "md5": _md5_file(zip_path),
                    "model_path": MODEL_PATH,
                    "dataset_path": DATASET_PATH,
                }
            },
            Callback=_ProgressCallback(size),
        )
        print(f"\nUploaded -> s3://{bucket}/{s3_key}")
        return True

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return False
    except RuntimeError as exc:
        logger.error("%s", exc)
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        return False
    except ClientError as exc:
        logger.error("S3 upload failed: %s", exc)
        return False
    finally:
        Path(zip_path).unlink(missing_ok=True)


def download_from_s3(
    bucket: str = S3_BUCKET,
    s3_key: str = S3_KEY,
    zip_path: str = ZIP_PATH,
) -> bool:
    """Download the artifact bundle from S3 and extract it locally."""
    if not bucket:
        logger.warning("AWS_S3_BUCKET/S3_BUCKET not set; skipping S3 download.")
        return False

    print(f"Downloading from s3://{bucket}/{s3_key} ...")
    try:
        s3 = _get_s3_client()
        meta = s3.head_object(Bucket=bucket, Key=s3_key)
        size = meta["ContentLength"]

        s3.download_file(
            bucket,
            s3_key,
            zip_path,
            Callback=_ProgressCallback(size),
        )
        print()

        _unzip_artifacts(zip_path)
        print("Artifacts ready.")
        return True

    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            logger.error("Artifact bundle not found. Run: python s3_store.py upload")
        else:
            logger.error("S3 download failed: %s", exc)
        return False
    except RuntimeError as exc:
        logger.error("%s", exc)
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        return False
    finally:
        Path(zip_path).unlink(missing_ok=True)


def sync_from_s3(
    bucket: str = S3_BUCKET,
    s3_key: str = S3_KEY,
) -> bool:
    """Download artifacts only when model or cleaned dataset is missing."""
    missing = [path for path in _artifact_paths() if not path.exists()]
    if not missing:
        logger.info("Model and dataset found locally; skipping S3 download.")
        return True

    logger.info(
        "Missing local artifact(s): %s. Downloading from S3...",
        ", ".join(str(path) for path in missing),
    )
    return download_from_s3(bucket=bucket, s3_key=s3_key)


def get_s3_status(
    bucket: str = S3_BUCKET,
    s3_key: str = S3_KEY,
) -> dict:
    """Return metadata about the S3 artifact bundle."""
    if not bucket:
        return {"error": "AWS_S3_BUCKET or S3_BUCKET not set"}

    try:
        s3 = _get_s3_client()
        meta = s3.head_object(Bucket=bucket, Key=s3_key)
        return {
            "exists": True,
            "size_mb": round(meta["ContentLength"] / (1024**2), 2),
            "last_modified": meta["LastModified"].isoformat(),
            "metadata": meta.get("Metadata", {}),
        }
    except RuntimeError as exc:
        return {"error": str(exc)}
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return {"exists": False}
        return {"error": str(exc)}


class _ProgressCallback:
    def __init__(self, total: int):
        self._total = total
        self._seen = 0

    def __call__(self, chunk: int):
        self._seen += chunk
        pct = self._seen / self._total * 100 if self._total else 0
        filled = int(pct // 5)
        bar = "#" * filled + "-" * (20 - filled)
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def _print_status() -> int:
    info = get_s3_status()
    if info.get("exists"):
        print("S3 artifact bundle found")
        print(f"Size:          {info['size_mb']} MB")
        print(f"Last modified: {info['last_modified']}")
        if info.get("metadata"):
            print(f"Uploaded at:   {info['metadata'].get('uploaded_at', 'unknown')}")
            print(f"MD5:           {info['metadata'].get('md5', 'unknown')}")
        return 0

    if info.get("exists") is False:
        print("No artifact bundle found in S3. Run: python s3_store.py upload")
        return 1

    print(f"Error: {info.get('error')}")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "upload":
        sys.exit(0 if upload_to_s3() else 1)
    if cmd == "download":
        sys.exit(0 if download_from_s3() else 1)
    if cmd == "sync":
        sys.exit(0 if sync_from_s3() else 1)
    if cmd == "status":
        sys.exit(_print_status())

    print("Usage:")
    print("  python s3_store.py upload    # upload model and cleaned dataset to S3")
    print("  python s3_store.py download  # download model and cleaned dataset from S3")
    print("  python s3_store.py sync      # download only if local artifacts are missing")
    print("  python s3_store.py status    # check S3 artifact bundle status")
