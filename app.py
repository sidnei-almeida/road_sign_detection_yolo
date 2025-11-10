import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO
from ultralytics import __version__ as yolo_version


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "modelos")
DEFAULT_MODEL_URLS = [
    "https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/modelos/best.pt",
    "https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/resultados/runs/detect/train/weights/best.pt",
]
ALLOWED_CLASSES = ["Traffic Light", "Stop", "Speedlimit", "Crosswalk"]
CLASS_COLORS = {
    "Traffic Light": (0, 194, 255),
    "Stop": (255, 107, 107),
    "Speedlimit": (138, 43, 226),
    "Crosswalk": (82, 196, 26),
}
CANON_MAP = {
    "traffic light": "Traffic Light",
    "traffic-light": "Traffic Light",
    "semaphore": "Traffic Light",
    "signal light": "Traffic Light",
    "stop": "Stop",
    "stop sign": "Stop",
    "speedlimit": "Speedlimit",
    "speed limit": "Speedlimit",
    "speed-limit": "Speedlimit",
    "crosswalk": "Crosswalk",
    "pedestrian crossing": "Crosswalk",
    "zebra crossing": "Crosswalk",
}


def to_canonical(name: str) -> str:
    key = (name or "").strip().lower()
    return CANON_MAP.get(key, name.title())


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left coordinate in pixels")
    y1: int = Field(..., description="Top coordinate in pixels")
    x2: int = Field(..., description="Right coordinate in pixels")
    y2: int = Field(..., description="Bottom coordinate in pixels")


class Detection(BaseModel):
    class_name: str = Field(..., description="Canonical class name")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    bounding_box: BoundingBox = Field(..., description="Bounding box in pixel coordinates")


class PredictionResponse(BaseModel):
    detections: List[Detection] = Field(default_factory=list)
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    annotated_image_base64: Optional[str] = Field(
        None,
        description="PNG image with detections drawn, base64 encoded. Present only when requested.",
    )


class ModelInfo(BaseModel):
    status: str
    weights_path: Optional[str]
    class_names: List[str]
    device: str
    torch_version: str
    ultralytics_version: str


MODEL: Optional[YOLO] = None
MODEL_PATH: Optional[str] = None
CLASS_NAMES: List[str] = []
DEVICE: str = "cpu"


def _load_dataset_classes() -> List[str]:
    dataset_yaml = os.path.join(BASE_DIR, "dados", "road_signs_dataset.yaml")
    if os.path.exists(dataset_yaml):
        try:
            with open(dataset_yaml, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
            names = data_cfg.get("names")
            if isinstance(names, dict):
                sorted_names = [
                    names[key]
                    for key in sorted(
                        names.keys(),
                        key=lambda v: int(v) if str(v).isdigit() else str(v),
                    )
                ]
                return [to_canonical(n) for n in sorted_names]
            if isinstance(names, list):
                return [to_canonical(n) for n in names]
        except Exception:
            return ALLOWED_CLASSES
    return ALLOWED_CLASSES


def _resolve_urls() -> List[str]:
    urls: List[str] = []
    env_url = os.getenv("MODEL_URL")
    if env_url:
        urls.append(env_url)
    urls.extend(DEFAULT_MODEL_URLS)
    return urls


def _download_weights(target_path: str) -> Optional[str]:
    for url in _resolve_urls():
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(response.content)
            if os.path.getsize(target_path) < 1_000_000:
                continue
            return target_path
        except Exception:
            continue
    return None


def load_model() -> Tuple[Optional[YOLO], Optional[str]]:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    candidates = [
        os.path.join(WEIGHTS_DIR, "best.pt"),
        os.path.join(BASE_DIR, "resultados", "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(WEIGHTS_DIR, "last.pt"),
        os.path.join(BASE_DIR, "resultados", "runs", "detect", "train", "weights", "last.pt"),
    ]

    weights_path: Optional[str] = None
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            weights_path = candidate
            break

    if weights_path is None:
        downloaded = _download_weights(os.path.join(WEIGHTS_DIR, "best.pt"))
        if downloaded:
            weights_path = downloaded

    if weights_path is None:
        return None, None

    try:
        model = YOLO(weights_path)
        return model, weights_path
    except Exception:
        return None, None


def _annotate_image(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    names: Dict[int, str],
) -> np.ndarray:
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for idx in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
        raw_name = names.get(int(classes[idx]), str(int(classes[idx])))
        canon_name = to_canonical(raw_name)
        color = CLASS_COLORS.get(canon_name, (0, 194, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{canon_name} {confidences[idx]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode annotated image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, MODEL_PATH, CLASS_NAMES, DEVICE
    DEVICE = _get_device()
    CLASS_NAMES = _load_dataset_classes()
    MODEL, MODEL_PATH = load_model()
    if MODEL is None:
        print("⚠️ YOLO weights not found. Set MODEL_URL or place weights under 'modelos/'.")
    else:
        print(f"✅ YOLO model loaded from {MODEL_PATH}")
    yield
    MODEL = None


app = FastAPI(
    title="Road Sign Detection API",
    description="REST API for Brazilian road sign detection using a YOLO model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, Any])
def root() -> Dict[str, Any]:
    return {
        "message": "Road Sign Detection API",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "classes": "/classes",
            "predict": "/predict",
        },
    }


@app.get("/health", response_model=Dict[str, str])
def health() -> Dict[str, str]:
    status = "ready" if MODEL is not None else "model_not_loaded"
    return {"status": status}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    status = "loaded" if MODEL is not None else "unavailable"
    device_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE
    return ModelInfo(
        status=status,
        weights_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        device=device_name,
        torch_version=torch.__version__,
        ultralytics_version=yolo_version,
    )


@app.get("/classes", response_model=Dict[str, List[str]])
def classes() -> Dict[str, List[str]]:
    return {"classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file (PNG or JPG)"),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    image_size: int = 768,
    include_image: bool = False,
) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Verify weights availability.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}") from exc

    image_np = np.array(image)
    start_time = time.perf_counter()
    try:
        results = MODEL.predict(
            image_np,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device=DEVICE,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    detections: List[Detection] = []
    annotated_image_base64: Optional[str] = None

    if results:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            names = result.names
            for idx in range(len(xyxy)):
                raw_name = names.get(int(classes[idx]), str(int(classes[idx])))
                canon_name = to_canonical(raw_name)
                bbox = BoundingBox(
                    x1=int(xyxy[idx][0]),
                    y1=int(xyxy[idx][1]),
                    x2=int(xyxy[idx][2]),
                    y2=int(xyxy[idx][3]),
                )
                detections.append(
                    Detection(
                        class_name=canon_name,
                        confidence=float(confidences[idx]),
                        bounding_box=bbox,
                    )
                )

            if include_image:
                annotated = _annotate_image(image_np, xyxy, classes, confidences, names)
                annotated_image_base64 = _encode_png_base64(annotated)

    return PredictionResponse(
        detections=detections,
        inference_time_ms=elapsed_ms,
        image_width=image_np.shape[1],
        image_height=image_np.shape[0],
        annotated_image_base64=annotated_image_base64,
    )


@app.post("/warmup", response_model=Dict[str, str])
def warmup() -> Dict[str, str]:
    """
    Endpoint to warm up the model (useful for Hugging Face Spaces cold starts).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Verify weights availability.")
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    MODEL.predict(dummy, conf=0.5, iou=0.5, imgsz=64, device=DEVICE, verbose=False)
    return {"status": "warmed"}