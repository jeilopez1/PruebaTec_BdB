import base64
import io
import os
from typing import Any, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.models import load_model


MODEL_PATH = os.getenv("MODEL_PATH", "modelo_mnist_final.keras")
INVERT_IF_BRIGHT = os.getenv("INVERT_IF_BRIGHT", "true").lower() in {"1", "true", "yes", "y"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 image string")


class PredictResponse(BaseModel):
    prediction: int


class ErrorResponse(BaseModel):
    detail: str


@app.on_event("startup")
def _load_model() -> None:
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. Set MODEL_PATH or place the exported model in the app folder."
        )
    model = load_model(MODEL_PATH)


def _strip_data_url_prefix(value: str) -> str:
    value = value.strip()
    if "," in value and value.lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def _decode_base64_image(image_b64: str) -> Image.Image:
    if not image_b64 or not image_b64.strip():
        raise HTTPException(status_code=400, detail="image_base64 is empty")

    payload = _strip_data_url_prefix(image_b64)

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 inválido")

    if not raw:
        raise HTTPException(status_code=400, detail="imagen vacía")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="formato no válido")
    except Exception:
        raise HTTPException(status_code=400, detail="no se pudo leer la imagen")

    return img


def _preprocess_image(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return both 4D and flat versions so the same API can work with CNNs and dense models."""
    gray = img.convert("L")
    gray = ImageOps.fit(gray, (28, 28), method=Image.Resampling.LANCZOS)

    arr = np.array(gray).astype("float32")

    # MNIST is usually white digit on black background after normalization.
    # If the uploaded image is black digit on white background, invert it.
    if INVERT_IF_BRIGHT and arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr_4d = arr.reshape(1, 28, 28, 1)
    arr_flat = arr.reshape(1, 784)
    return arr_4d, arr_flat


def _predict_with_shape_fallback(arr_4d: np.ndarray, arr_flat: np.ndarray) -> np.ndarray:
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="model is not loaded")

    input_shape = getattr(model, "input_shape", None)

    # Prefer the most likely format first.
    candidate_inputs: List[np.ndarray] = []
    if isinstance(input_shape, tuple):
        if len(input_shape) == 2 and input_shape[1] in (784, None):
            candidate_inputs = [arr_flat, arr_4d]
        elif len(input_shape) == 4:
            candidate_inputs = [arr_4d, arr_flat]
        else:
            candidate_inputs = [arr_4d, arr_flat]
    else:
        candidate_inputs = [arr_4d, arr_flat]

    last_error: Exception | None = None
    for x in candidate_inputs:
        try:
            preds = model.predict(x, verbose=0)
            return preds
        except Exception as exc:
            last_error = exc

    raise HTTPException(status_code=500, detail=f"prediction failed: {last_error}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(payload: PredictRequest) -> PredictResponse:
    img = _decode_base64_image(payload.image_base64)
    arr_4d, arr_flat = _preprocess_image(img)
    preds = _predict_with_shape_fallback(arr_4d, arr_flat)

    if preds is None or len(preds) == 0:
        raise HTTPException(status_code=500, detail="empty prediction output")

    probs = np.asarray(preds[0]).astype(float)
    pred = int(np.argmax(probs))

    return PredictResponse(
        prediction=pred,
    )


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "MNIST Digit Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
