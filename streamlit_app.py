import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import json
 
# Use relative imports as required
from utils.detector import YOLOModel
from utils.visualization import draw_boxes
 
# --- Page Config ---
st.set_page_config(
    page_title="YOLO Object Detection", 
    page_icon="🎯", 
    layout="centered"
)
 
# --- Helper: JSON Serialization Fix ---
def make_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj
 
# --- Model Initialization ---
@st.cache_resource
def load_model():
    """
    Load the YOLO model safely with caching.
    Ensures model is only loaded once per session.
    """
    try:
        model = YOLOModel(model_path="model/best.pt", labels_path="model/labels.txt")
        return model, True
    except Exception as e:
        return str(e), False
 
# Initialize model
with st.spinner("Loading Model..."):
    model_or_error, is_loaded = load_model()
 
if not is_loaded:
    st.error(f"Failed to load model. Error: {model_or_error}")
    st.stop()
    
model = model_or_error
 
# --- UI Layout ---
st.title("🎯 YOLO Object Detection App")
st.markdown("Upload an image below or click **Run Detection** to test with the demo image.")
 
# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("Adjust the confidence threshold to filter out weak detections.")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.25, 
    step=0.01,
    help="Minimum confidence score for a bounding box to be drawn."
)
 
st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** Processing time may vary depending on image size and whether a GPU is available."
)
 
# --- Main App ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
 
image_to_process = None
demo_path = os.path.join("assets", "demo.png")
 
# Determine which image to process
if uploaded_file is not None:
    try:
        image_to_process = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
elif os.path.exists(demo_path):
    st.info("Using default demo image. Upload an image above to run your own.")
    image_to_process = Image.open(demo_path).convert("RGB")
else:
    st.warning("Please upload an image to begin.")
 
# Proceed if we have a valid image
if image_to_process is not None:
    # Display the original image
    st.markdown("### 🖼️ Original Image")
    st.image(image_to_process, use_container_width=True)
    
    # Centered Run Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_btn = st.button("Run YOLO Detection 🚀", use_container_width=True)
        
    if run_btn:
        with st.spinner("Running YOLO inference..."):
            try:
                start_time = time.time()
                
                # 1. Convert to numpy array
                img_array = np.array(image_to_process)
                
                # 2. Run inference
                detections = model.predict(img_array, conf_threshold=conf_threshold)
                
                # 3. Draw bounding boxes
                annotated_img = draw_boxes(img_array, detections)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # 4. Display Result
                st.markdown("---")
                st.markdown("### 📊 Annotated Result")
                st.image(annotated_img, use_container_width=True)
                
                # 5. Display metrics and raw data
                st.success(f"Detection complete! Found {len(detections)} objects in {inference_time:.2f} seconds.")
                
                if len(detections) > 0:
                    with st.expander("🔍 View Raw Detection Data"):
                        # ✅ FIX: Convert numpy types before passing to st.json()
                        st.json(make_serializable(detections))
                        
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")
