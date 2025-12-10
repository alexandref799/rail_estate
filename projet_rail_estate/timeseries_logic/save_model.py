"""
Helper to save a local model file to a GCP Cloud Storage bucket.
Requires `google-cloud-storage` and valid GCP credentials (e.g. via
GOOGLE_APPLICATION_CREDENTIALS or workload identity).
"""

from pathlib import Path
from typing import Optional

from google.cloud import storage


def save_model_local(model, path: str | Path, overwrite: bool = True) -> Path:
    """
    Save a Keras model locally.

    - If path ends with .keras or .h5: uses model.save(...)
    - Otherwise: exports a SavedModel directory via model.export(...)

    Args:
        model: The compiled Keras model to save.
        path: Destination path (e.g. "models/my_model.keras", "models/my_model.h5",
              or "models/saved_model_dir" for SavedModel).
        overwrite: Whether to overwrite existing content (applies to model.save).

    Returns:
        The pathlib.Path of the saved artifact.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".keras", ".h5"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path, overwrite=overwrite)
    else:
        # Treat as a directory and export SavedModel
        path.mkdir(parents=True, exist_ok=True)
        model.export(path)

    return path


def upload_to_gcp(local_path: str | Path, bucket_name: str, destination_blob: str) -> str:
    """
    Upload a local model artifact to a GCS bucket.

    Args:
        local_path: Path to the local file or directory to upload (e.g. .h5 or SavedModel dir).
        bucket_name: Name of the target GCS bucket.
        destination_blob: Path/key inside the bucket (e.g. "models/my_model.h5" or "models/saved_model/").

    Returns:
        The gs:// URI of the uploaded artifact.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local path not found: {local_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if local_path.is_dir():
        # Upload directory recursively (for SavedModel)
        for p in local_path.rglob("*"):
            if p.is_file():
                rel = p.relative_to(local_path)
                blob_path = f"{destination_blob.rstrip('/')}/{rel.as_posix()}"
                bucket.blob(blob_path).upload_from_filename(p)
    else:
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(local_path)

    return f"gs://{bucket_name}/{destination_blob.rstrip('/')}"
