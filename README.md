<p align="center">
  <img src="./images/header.png" alt="RoadSight — road sign intelligence powered by YOLO" width="920" />
</p>

<p align="center">
  <strong>Ultralytics YOLO · FastAPI · OpenCV · PyTorch</strong><br />
  <em>REST API for detecting Brazilian road signs — traffic lights, stops, speed limits, and crosswalks.</em>
</p>

<p align="center">
  <a href="https://github.com/sidnei-almeida/road_sign_detection_yolo"><strong>github.com/sidnei-almeida/road_sign_detection_yolo</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?logo=yolo&logoColor=black" alt="YOLO" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License" />
</p>

---

## Overview

**Road Sign Detection (YOLO)** packages a custom **Ultralytics YOLO** detector behind a **FastAPI** service. Upload a street-scene image and receive **bounding boxes**, **class labels**, **confidence scores**, optional **annotated PNG (base64)**, and **latency** in milliseconds.

**Supported classes (canonical names):**

| Class | Typical use |
|--------|-------------|
| **Traffic Light** | Signal heads at intersections |
| **Stop** | Stop signs |
| **Speedlimit** | Regulatory speed plates |
| **Crosswalk** | Pedestrian crossings |

Class names are normalized in code (`to_canonical`) so minor label variants from training still map to these four categories.

---

## End-to-end workflow

```mermaid
flowchart LR
  A[Images + labels] --> B[Notebooks / YOLO train]
  B --> C[best.pt weights]
  C --> D[FastAPI app.py]
  D --> E[POST /predict]
```

1. **Data** — Dataset config in `dados/road_signs_dataset.yaml` (class list, train/val paths). Sample images under `dados/image_examples/` for smoke tests.
2. **Training** — YOLO training artifacts land under `resultados/runs/detect/train/weights/` (e.g. `best.pt`). Notebooks in `notebooks/` cover EDA, preprocessing, and training (portfolio workflow).
3. **Serving** — `app.py` loads weights from `modelos/best.pt`, training output paths, or **`MODEL_URL`** / GitHub raw fallbacks at startup.
4. **Inference** — `MODEL.predict()` with tunable `conf_threshold`, `iou_threshold`, and `image_size` (default **768**). Device: **CUDA**, **MPS**, or **CPU**.

---

## Demo UI (frontend)

Any client can call the API. The layout below illustrates a **RoadSight**-style experience: sample gallery, live detection overlay, and confidence feedback.

<p align="center">
  <img src="./images/software.png" alt="RoadSight dashboard — samples, detection overlay, YOLOv8 insights" width="900" />
</p>

<p align="center">
  <sub>Example UI wired to <code>POST /predict</code> (optional <code>include_image=true</code> for annotated frames).</sub>
</p>

---

## Quick start (local)

**Requirements:** Python **3.10+**, Git **LFS** if you clone large `.pt` weights tracked in `.gitattributes`.

```bash
git clone https://github.com/sidnei-almeida/road_sign_detection_yolo.git
cd road_sign_detection_yolo

git lfs install && git lfs pull   # if weights are LFS-tracked

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Place weights at modelos/best.pt OR set MODEL_URL to a direct .pt URL
bash run_app.sh
```

API base URL: **`http://0.0.0.0:7860`** (override with env **`PORT`** in Docker).

Open **`http://127.0.0.1:7860/docs`** for interactive OpenAPI.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service message and endpoint map |
| GET | `/health` | `ready` vs `model_not_loaded` |
| GET | `/model/info` | Weights path, classes, device, Torch / Ultralytics versions |
| GET | `/classes` | List of class names |
| POST | `/predict` | Multipart image + query params (see below) |
| POST | `/warmup` | Dummy inference to reduce cold-start latency (Spaces) |

### `POST /predict` parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `file` | — | Image upload (PNG / JPG) |
| `conf_threshold` | `0.25` | Minimum detection confidence |
| `iou_threshold` | `0.5` | NMS IoU threshold |
| `image_size` | `768` | YOLO inference size |
| `include_image` | `false` | If `true`, returns `annotated_image_base64` (PNG) |

### Example

```bash
curl -X POST "http://localhost:7860/predict?include_image=true" \
  -F "file=@dados/image_examples/road0.jpg"
```

```json
{
  "detections": [
    {
      "class_name": "Traffic Light",
      "confidence": 0.92,
      "bounding_box": { "x1": 123, "y1": 45, "x2": 210, "y2": 300 }
    }
  ],
  "inference_time_ms": 56.42,
  "image_width": 1280,
  "image_height": 720,
  "annotated_image_base64": "..."
}
```

---

## Model weights

Resolution order at startup:

1. `modelos/best.pt`
2. `resultados/runs/detect/train/weights/best.pt`
3. `modelos/last.pt` / `resultados/.../last.pt`
4. Download from **`MODEL_URL`** (if set), then built-in GitHub raw URLs in `app.py`

Files under **`modelos/*.pt`** are intended for **Git LFS** (see `.gitattributes`).

---

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_URL` | HTTPS URL to download `.pt` weights at startup | — |
| `PORT` | Uvicorn listen port (Docker exposes **7860**) | `7860` |

---

## Docker & Hugging Face Spaces

- **`Dockerfile`**: Python 3.10 slim, OpenGL libs for OpenCV, installs `requirements.txt`, runs **`uvicorn app:app`** on port **7860**.
- Push the repo to a **Docker** Space; set **`MODEL_URL`** or commit LFS weights.
- Call **`POST /warmup`** after deploy to warm the model.

---

## Project structure

```
road_sign_detection_yolo/
├── app.py                 # FastAPI + YOLO inference
├── Dockerfile
├── requirements.txt
├── run_app.sh             # Local uvicorn launcher
├── dados/
│   ├── road_signs_dataset.yaml
│   ├── road_signs_annotations.csv
│   └── image_examples/
├── modelos/               # best.pt (LFS)
├── resultados/            # YOLO training runs
├── notebooks/             # EDA, training, evaluation
├── images/
│   ├── header.png
│   └── software.png
└── LICENSE
```

---

## Testing

```bash
curl http://localhost:7860/health

curl -X POST "http://localhost:7860/predict" \
  -F "file=@dados/image_examples/road0.jpg"
```

---

## Disclaimer

Computer-vision outputs are **probabilistic**. Do not rely on this API alone for safety-critical driving or regulatory compliance. Validate on your own data and hardware.

---

## License

This project is released under the **[MIT License](LICENSE)**.

---

## Author

**Sidnei Alves de Almeida** — [@sidnei-almeida](https://github.com/sidnei-almeida)
