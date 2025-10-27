# 🚦 Road Sign Detection App - YOLOv8

Professional Streamlit application for real-time road sign detection using YOLOv8.

## 🎯 Overview

This project implements a complete road sign detection system with:
- **4 Classes**: Speed limit, crosswalk, traffic light, and stop sign
- **YOLOv8 Model**: Nano architecture optimized for speed
- **Professional Interface**: Modern and responsive design
- **Complete Analysis**: Interactive training graphs and metrics

## 🚀 Installation and Execution

### Option 1: Quick Execution
```bash
# Clone the repository
git clone <repository-url>
cd road_sign_detection_yolo

# Run the automatic script
./run_app.sh
```

### Option 2: Manual Installation
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements_lite.txt

# 3. Run the app
streamlit run app_lite.py
```

## 📱 Features

### 🔍 Image Detection
- Image upload (PNG, JPG, JPEG)
- Real-time detection with bounding boxes
- Confidence threshold adjustment
- Results table with confidence

### 📊 Model Analysis
- Real-time performance metrics
- Detailed class information
- Dataset statistics

### 📈 Training Visualization
- Interactive charts with Plotly
- Metric curves (Precision, Recall, mAP)
- Loss function analysis
- Detailed process statistics

### ℹ️ Documentation
- Complete project information
- Technologies used
- Model performance

## 🏗️ Architecture

```
road_sign_detection_yolo/
├── app.py                    # Complete app with YOLO
├── app_lite.py              # Demo app (without YOLO)
├── requirements.txt         # Complete dependencies
├── requirements_lite.txt    # Minimum dependencies
├── run_app.sh              # Execution script
├── demo.py                 # Dependency verification
├── dados/
│   ├── road_signs_annotations.csv
│   └── road_signs_dataset.yaml
├── dataset/
│   ├── train/ (701 images)
│   └── val/ (176 images)
├── modelos/
│   ├── best.pt
│   └── last.pt
├── resultados/
│   └── runs/detect/train/
└── notebooks/
    ├── 1_Exploratory_Data_Analysis_.ipynb
    ├── 2_Data_Pre_Processing.ipynb
    └── 3_Model_Training.ipynb
```

## 📊 Dataset

- **Total**: 1,244 sign annotations
- **Training**: 701 images (984 annotations)
- **Validation**: 176 images (260 annotations)
- **Classes**:
  - 🚦 Speed Limit: 783 annotations
  - 🚶 Crosswalk: 200 annotations
  - 🔴 Traffic Light: 170 annotations
  - 🛑 Stop Sign: 91 annotations

## 📈 Performance

- **mAP@0.5**: ~92%
- **mAP@0.5:0.95**: ~77%
- **Precision**: ~94%
- **Recall**: ~88%

## 🛠️ Technologies

### Backend
- **Python 3.13**
- **Streamlit** - Web interface
- **OpenCV** - Image processing
- **PIL/Pillow** - Image manipulation

### Machine Learning
- **YOLOv8** - Object detection
- **Ultralytics** - YOLO framework
- **PyTorch** - ML backend (optional)

### Visualization
- **Plotly** - Interactive charts
- **Pandas** - Data manipulation
- **NumPy** - Numerical computation

## 🎨 Interface

### Professional Design
- Corporate colors (blue and gradients)
- Responsive layout
- Interactive components
- Real-time visual feedback

### Navigation
- Organized tab system
- Sidebar with settings
- Real-time metrics
- Interactive charts

## 🔧 Settings

### Confidence Threshold
- Range: 0.1 - 1.0
- Default: 0.5
- Adjustable in real-time

### Supported Formats
- PNG, JPG, JPEG
- Automatic resolution
- Optimization for 640x640

## 📝 Versions

### app.py (Complete)
- Requires YOLOv8 installed
- Real detection with trained model
- Dependencies: PyTorch, Ultralytics

### app_lite.py (Demo)
- Simulated detections
- No heavy dependencies
- Ideal for demonstration

## 🚀 Deploy

### Streamlit Cloud
1. Connect the repository
2. Configure `requirements_lite.txt`
3. Run `streamlit run app_lite.py`

### Local
```bash
./run_app.sh
```

## 📖 Documentation

- **README_FINAL.md** - This file
- **README_app.md** - App documentation
- **notebooks/** - Development process
- **demo.py** - Dependency verification

## 🤝 Contribution

1. Fork the project
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is under the MIT license. See the `LICENSE` file for more details.

## 👨‍💻 Author

Developed with ❤️ using Streamlit and YOLOv8

---

**🎉 Ready to use! Run `./run_app.sh` and start detecting road signs!**
