# 🚦 Road Sign Detection • YOLO

Road sign detection with YOLO, accompanied by an elegant Streamlit application (dark theme) for inference, training metrics visualization, and data exploration.

- **Author**: [sidnei-almeida](https://github.com/sidnei-almeida)
- **Contact**: <sidnei.almeida1806@gmail.com>

---

## ✨ Highlights
- Premium Streamlit app with dark theme and cyan/purple palette
- **Detection** page with image upload, **camera (streamlit-webrtc)** and **example selection (streamlit-image-select)**
- **Training** page with graphs (results.csv) and artifacts (confusion matrix, batches, validation)
- **Data** page displaying `dados/road_signs_dataset.yaml` and annotation CSV sample

> Note: the current model detects only: **Traffic Light**, **Stop**, **Speedlimit**, **Crosswalk**.

---

## 🚀 How to run

Prerequisites: Python 3.10+ and dependencies from `requirements.txt`.

```bash
# clone and enter the project
git clone https://github.com/sidnei-almeida/road_sign_detection_yolo.git
cd road_sign_detection_yolo

# (optional) create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows PowerShell

# install dependencies
pip install -r requirements.txt

# run the app
bash run_app.sh
# or
streamlit run app.py
```

Place the model weights in `modelos/best.pt` (or use `resultados/runs/detect/train/weights/best.pt`).

---

## 🧱 Structure

```
road_sign_detection_yolo/
├─ app.py                         # Streamlit app
├─ .streamlit/config.toml         # Custom dark theme
├─ dados/
│  ├─ road_signs_dataset.yaml     # YOLO dataset config
│  ├─ road_signs_annotations.csv  # Annotations (sample/EDA)
│  └─ image_examples/             # Images for Examples tab (PNG/JPG)
├─ modelos/
│  ├─ best.pt                     # Model weights (place here)
│  └─ last.pt
├─ resultados/runs/detect/train/  # YOLO training artifacts
│  ├─ results.csv                 # Metrics per epoch
│  ├─ results.png                 # Summary
│  ├─ confusion_matrix.png        # Confusion matrix
│  ├─ confusion_matrix_normalized.png
│  ├─ train_batch*.jpg            # Training batches
│  ├─ val_batch*_pred.jpg         # Validation predictions
│  └─ weights/best.pt             # Weights
└─ notebooks/                     # EDA, Preprocessing, Training
```

---

## 📷 Examples and Artifacts (direct links)
- Dataset YAML: [`dados/road_signs_dataset.yaml`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/dados/road_signs_dataset.yaml)
- Annotations CSV: [`dados/road_signs_annotations.csv`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/dados/road_signs_annotations.csv)
- Training artifacts:
  - [`results.png`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/resultados/runs/detect/train/results.png)
  - [`confusion_matrix.png`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/resultados/runs/detect/train/confusion_matrix.png)
  - [`confusion_matrix_normalized.png`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/resultados/runs/detect/train/confusion_matrix_normalized.png)
  - [`labels.jpg`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/resultados/runs/detect/train/labels.jpg)
  - [`results.csv`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/resultados/runs/detect/train/results.csv)
- Model weights (large file):
  - [`modelos/best.pt`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/modelos/best.pt)
  - [`resultados/runs/detect/train/weights/best.pt`](https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/resultados/runs/detect/train/weights/best.pt)
- Examples (replace with actual name in `dados/image_examples/`):
  - `https://raw.githubusercontent.com/sidnei-almeida/road_sign_detection_yolo/main/dados/image_examples/road0.jpg`

---

## 📈 App – Pages
- **Home**: system status, classes and mAP summary, training highlights
- **Detection**: upload | camera | examples | batch; inference presets; class filters; annotated image download
- **Training**: interactive graphs from `results.csv` + main images
- **Data**: dataset YAML visualization and annotation sample
- **About**: project information and contact

---

## 🧪 Examples
- Place images in `dados/image_examples/` to appear in the Examples tab.
- If the folder is empty, the app tries to use `dados/examples/` (legacy) or validation images from `resultados/runs/detect/train`.

---

## 📬 Contact
- GitHub: [sidnei-almeida](https://github.com/sidnei-almeida)
- E-mail: <sidnei.almeida1806@gmail.com>

```text
If this project was useful to you, leave a star on the repository ⭐
```
