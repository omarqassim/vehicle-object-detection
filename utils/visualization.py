import cv2
import numpy as np

def draw_boxes(image, detections):
    """
    Draw bounding boxes and class labels with confidence on the image.
    
    Args:
        image: numpy array representing the image (RGB or BGR)
        detections: list of dictionaries containing box, confidence, class_name, and class_id
        
    Returns:
        Annotated image as numpy array
    """
    # Create a copy so we don't modify the original image in-place
    annotated_image = np.array(image, copy=True)
    
    # Generate distinct, pleasing colors based on class_id
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = np.random.randint(50, 255, size=(100, 3), dtype=np.uint8)
    
    # Calculate adaptive thickness based on image resolution
    img_h, img_w = annotated_image.shape[:2]
    thickness = max(1, int(max(img_h, img_w) * 0.003))
    
    # Calculate optimal font scale based on image dimensions
    font_scale = max(0.4, max(img_h, img_w) * 0.0008)
    font_thickness = max(1, int(font_scale * 2))
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["confidence"]
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        
        # Get consistent color for this class
        color = tuple(int(c) for c in colors[cls_id % 100])
        
        # Draw the bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label = f"{cls_name} {conf:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Ensure text doesn't go above the image
        text_y_pos = max(y1, text_height + 5)
        
        # Draw filled rectangle for text background to improve readability
        cv2.rectangle(
            annotated_image, 
            (x1, text_y_pos - text_height - baseline - 5), 
            (x1 + text_width + 4, text_y_pos), 
            color, 
            -1
        )
        
        # Determine text color (black or white) based on background color brightness
        bg_brightness = sum(color)
        text_color = (0, 0, 0) if bg_brightness > 450 else (255, 255, 255)
            
        # Draw the label text
        cv2.putText(
            annotated_image,
            label,
            (x1 + 2, text_y_pos - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
        
    return annotated_image
