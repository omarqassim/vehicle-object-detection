import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path="model/best.pt", labels_path="model/labels.txt"):
        """
        Initialize the YOLO model.
        Automatically detects if GPU (CUDA) is available.
        Loads classes from labels.txt if available.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # We need absolute paths for robust deployment, but the prompt asked for relative paths.
        # We will keep paths relative as requested, but we can resolve them relative to the current working directory.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Load labels dynamically
        self.labels = self._load_labels(labels_path)
        
    def _load_labels(self, labels_path):
        """
        Reads class names from labels.txt.
        Each line corresponds to one class name.
        """
        if not os.path.exists(labels_path):
            print(f"Warning: Labels file not found at {labels_path}. Using model's default classes.")
            return self.model.names if hasattr(self.model, 'names') else {}
            
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
            
        return {i: label for i, label in enumerate(labels)}

    def predict(self, image, conf_threshold=0.25):
        """
        Run inference on the given image.
        Returns a structured list of detections.
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Run inference
        results = self.model.predict(source=image, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence score
                conf = float(box.conf[0])
                
                # Get class ID
                cls_id = int(box.cls[0])
                
                # Map to class name
                cls_name = self.labels.get(cls_id, f"class_{cls_id}")
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })
                
        return detections
