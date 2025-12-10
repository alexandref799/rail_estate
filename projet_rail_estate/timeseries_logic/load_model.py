
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from google.cloud import storage
from tensorflow.keras.models import load_model as _keras_load_model, Model


def load_model(path: str | Path, compile: bool = True) -> Model:
    """
    Load a Keras model from disk.

    Args:
        path: Path to the saved model (.h5/.keras or SavedModel directory).
        compile: Whether to compile the model after loading.

    Returns:
        The loaded Keras model.
    """
    return _keras_load_model(path, compile=compile)


from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from google.cloud import storage
from keras.models import load_model as keras_load_model


from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from google.cloud import storage
from keras.models import load_model as keras_load_model


def load_model_from_gcp(
    bucket_name: str,
    blob_name: str,  # ex: "models/my_model.keras"
    compile: bool = True,
) -> "keras.Model":
    """
    Download a .keras model from GCS and load it with Keras.

    Args:
        bucket_name: Name of the GCS bucket.
        blob_name: Exact path of the file inside the bucket
                   (e.g. "models/my_model.keras", NOT "gs://bucket/models/my_model.keras").
        compile: Whether to compile the model after loading.

    Returns:
        The loaded Keras model.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Important: this must match the *exact* object name in the bucket
    if not blob.exists(client=client):
        raise FileNotFoundError(f"Blob not found in GCS: gs://{bucket_name}/{blob_name}")

    # Download into a temporary local file, then load with Keras
    with NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        blob.download_to_filename(str(tmp_path))

    try:
        model = keras_load_model(tmp_path, compile=compile)
    finally:
        # Nettoyage du fichier temporaire
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return model

