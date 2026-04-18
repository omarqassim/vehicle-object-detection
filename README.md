# YOLO Object Detection - Streamlit Deployment

A clean, modular, and production-ready project scaffold for deploying a YOLO object detection model via Streamlit Cloud.

## 📁 Project Structure

```
yolo-terminal-deployment/
│
├── app.py                 # Local CLI script for testing inference via terminal
├── streamlit_app.py       # Streamlit web application interface
├── requirements.txt       # Python dependencies (for pip and Streamlit Cloud)
├── packages.txt           # OS-level packages (required for OpenCV in Streamlit Cloud)
├── README.md              # This documentation file
│
├── model/                 # Directory containing the trained weights and labels
│   ├── best.pt            # The exported YOLO weights
│   └── labels.txt         # Class names, one per line
│
├── utils/                 # Modular utility functions
│   ├── detector.py        # YOLOModel wrapper class for robust inference
│   └── visualization.py   # OpenCV functions to draw beautiful bounding boxes
│
└── assets/                # Static assets for the app
    └── demo.png           # Demo image for testing
```

## 🚀 How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Test via Terminal (CLI)
Verify the model loads and inference works correctly without GUI dependencies:
```bash
python app.py
```
This will run inference on `assets/demo.png`, print the detection results to the terminal, and save an annotated `output.jpg` to the root folder.

### 3. Run the Streamlit Web App
Launch the interactive web interface locally:
```bash
streamlit run streamlit_app.py
```

## ☁️ Streamlit Cloud Deployment Steps

1. Push this entire `yolo-terminal-deployment` repository to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and log in.
3. Click **New app** and connect your GitHub account if you haven't already.
4. Select your repository, the correct branch, and set the **Main file path** to `streamlit_app.py`.
5. Click **Deploy!** 

*(Streamlit Cloud will automatically detect `packages.txt` and use it to install `libgl1` required for OpenCV to work on Linux without a display, and use `requirements.txt` to install the required Python packages).*

## ⚠️ Important Notes
- The project is fully compatible with Linux environments (Streamlit Cloud).
- We use `opencv-python-headless` to avoid Qt/GUI library issues on servers.
- The app gracefully falls back to CPU if a GPU is not available.
"# vehicle-object-detection" 
"# vehicle-object-detection" 
