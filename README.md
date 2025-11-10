---
title: Road Sign Detection API
emoji: "🚦"
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Road Sign Detection API

REST API for Brazilian road sign detection powered by a custom YOLO model. This repository is prepared for deployment on Hugging Face Spaces using the Docker runtime. It exposes endpoints for health checks, model metadata, class listing, inference, and warm-up routines.

- **Author**: [sidnei-almeida](https://github.com/sidnei-almeida)  
- **Contact**: <sidnei.almeida1806@gmail.com>

> The current model recognises four classes: **Traffic Light**, **Stop**, **Speedlimit**, and **Crosswalk**.

---

## Features
- FastAPI application with automatic model loading and warm-up endpoint.
- YOLO inference pipeline returning bounding boxes, confidence scores, and optional annotated images.
- Dockerfile tailored for Hugging Face Spaces (`sdk: docker`) deployment.
- Optional automatic download of weights through the `MODEL_URL` environment variable.

---

## Quick Start (Local)

Prerequisites: Python 3.10+ and Git LFS for large weight files.

```bash
git clone https://github.com/sidnei-almeida/road_sign_detection_yolo.git
cd road_sign_detection_yolo

python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# place your weights at modelos/best.pt or set MODEL_URL
bash run_app.sh
```

The API listens on `http://0.0.0.0:7860` by default. You can override the port with the `PORT` environment variable when running the Docker container.

---

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/` | Returns available endpoints. |
| `GET`  | `/health` | Reports whether the model is loaded. |
| `GET`  | `/model/info` | Metadata about the model, weights and environment. |
| `GET`  | `/classes` | Lists supported class names. |
| `POST` | `/predict` | Perform inference. Accepts `multipart/form-data` with an image file and optional parameters (`conf_threshold`, `iou_threshold`, `image_size`, `include_image`). |
| `POST` | `/warmup` | Executes a dummy inference to warm up the model (useful for Spaces cold starts). |

### Example Request

```bash
curl -X POST "http://localhost:7860/predict?include_image=true" \
  -F "file=@dados/image_examples/road0.jpg"
```

Sample response:

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
  "annotated_image_base64": "iVBORw0K..."
}
```

---

## Deploying on Hugging Face Spaces

1. Create a new Space and choose **Docker** as the runtime (`sdk: docker`).  
2. Push this repository (including `Dockerfile`) to the Space.  
3. Ensure model weights are available in the repository (e.g. via Git LFS) or configure the `MODEL_URL` secret in the Space settings.  
4. Once the Space builds, the API will be reachable at `https://<your-space>.hf.space`.

---

## Project Structure

```
road_sign_detection_yolo/
├─ app.py                      # FastAPI service
├─ Dockerfile                  # Docker runtime for Hugging Face Spaces
├─ requirements.txt            # Python dependencies
├─ run_app.sh                  # Helper script to start uvicorn locally
├─ dados/                      # Dataset metadata and sample images
│  ├─ road_signs_dataset.yaml
│  ├─ road_signs_annotations.csv
│  └─ image_examples/
├─ modelos/                    # Model weights (place best.pt here)
├─ resultados/                 # Training artefacts from YOLO runs
└─ notebooks/                  # EDA, preprocessing, training notebooks
```

---

## Model Weights

- Default path: `modelos/best.pt`
- Alternate path: `resultados/runs/detect/train/weights/best.pt`
- Remote download (optional): set the `MODEL_URL` environment variable to a direct link (e.g. Hugging Face Hub or GitHub Raw). The API will attempt to download the weights at startup if no local file is found.

---

## Environment Variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MODEL_URL` | Optional HTTPS URL to download the YOLO weights during startup. | _None_ |
| `PORT` | Uvicorn port (the Dockerfile exposes 7860, which Hugging Face expects). | `7860` |

---

## Testing

After starting the service, run:

```bash
curl http://localhost:7860/health
```

To run an inference test with a sample image:

```bash
curl -X POST "http://localhost:7860/predict" \
  -F "file=@dados/image_examples/road0.jpg"
```

---

## License

This project is distributed under the [MIT License](LICENSE).
