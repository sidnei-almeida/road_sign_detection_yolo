# 🚦 Road Sign Detection App

Professional Streamlit application for real-time road sign detection using YOLOv8.

## 🚀 Features

- **Real-Time Detection**: Upload images and instant road sign detection
- **4 Sign Classes**: Speed limit, crosswalk, traffic light, and stop sign
- **Interactive Interface**: Adjust confidence threshold and visualize results
- **Model Analysis**: Interactive training graphs and performance metrics
- **Professional Design**: Modern and responsive interface

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd road_sign_detection_yolo
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

## 🎯 How to Use

1. **Go to the "Detection" tab**
2. **Upload an image** with road signs
3. **Adjust the confidence threshold** in the sidebar (optional)
4. **Click "Detect Signs"**
5. **View the results** with bounding boxes and confidence

## 📊 Available Tabs

### 🔍 Detection
- Image upload and processing
- Result visualization with bounding boxes
- Detection table with confidence

### 📊 Model Analysis
- Final performance metrics
- Information about detected classes
- Model statistics

### 📈 Training
- Interactive metric graphs
- Loss curves during training
- Detailed statistics of the process

### ℹ️ About
- Project information
- Technologies used
- Model performance

## 🛠️ Technologies

- **Frontend**: Streamlit
- **ML**: YOLOv8 (Ultralytics)
- **Visualization**: Plotly
- **Processing**: OpenCV, PIL
- **Data**: Pandas, NumPy

## 📈 Performance

- **mAP@0.5**: ~92%
- **mAP@0.5:0.95**: ~77%
- **Precision**: ~94%
- **Recall**: ~88%

## 🎨 Interface Features

- Responsive and modern design
- Professional colors (blue and gradients)
- Interactive charts with Plotly
- Real-time visual feedback
- Sidebar with settings
- Organized tab system

## 🔧 Settings

- **Confidence Threshold**: 0.1 - 1.0 (default: 0.5)
- **Supported Formats**: PNG, JPG, JPEG
- **Resolution**: Automatic (optimized for 640x640)

## 📝 Notes

- The model must be in the `modelos/best.pt` folder
- Training data must be in `resultados/runs/detect/train/results.csv`
- Dataset configuration must be in `dados/road_signs_dataset.yaml`
