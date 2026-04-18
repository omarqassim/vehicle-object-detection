import os
import cv2
import numpy as np
from PIL import Image

# Import from our custom modules
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

def main():
    print("\n--- YOLO Terminal Deployment ---")
    print("[1/4] Initializing YOLO Model...")
    try:
        model = YOLOModel(model_path="model/best.pt", labels_path="model/labels.txt")
        print(f"      Model loaded successfully on device: {model.device.upper()}")
    except Exception as e:
        print(f"      Error loading model: {e}")
        print("      Please ensure 'model/best.pt' exists.")
        return

    # Look for the demo image
    demo_image_path = os.path.join("assets", "demo.png")
    if not os.path.exists(demo_image_path):
        print(f"\n[!] Error: Demo image not found at '{demo_image_path}'.")
        print("    Please place a test image there or modify the script path.")
        return
        
    print(f"\n[2/4] Loading image from '{demo_image_path}'...")
    try:
        image = Image.open(demo_image_path).convert("RGB")
        image_np = np.array(image)
        print(f"      Image loaded. Size: {image_np.shape[1]}x{image_np.shape[0]}")
    except Exception as e:
        print(f"      Error reading image: {e}")
        return

    print("\n[3/4] Running YOLO inference...")
    detections = model.predict(image_np, conf_threshold=0.25)
    
    print(f"      Inference complete. Found {len(detections)} objects.")
    
    if len(detections) > 0:
        print("\n--- Detections ---")
        for i, d in enumerate(detections):
            box = [int(c) for c in d['box']]
            print(f"  {i+1}. {d['class_name']:<15} | Confidence: {d['confidence']:.2f} | Box: {box}")
    else:
        print("      No objects detected above the confidence threshold.")
        
    print("\n[4/4] Generating output image...")
    annotated_image = draw_boxes(image_np, detections)
    
    # Convert RGB back to BGR for saving with cv2
    output_path = "output.jpg"
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    success = cv2.imwrite(output_path, annotated_image_bgr)
    if success:
        print(f"\n✅ Success! Annotated image saved to '{output_path}'")
    else:
        print(f"\n❌ Error saving image to '{output_path}'")

if __name__ == "__main__":
    main()
