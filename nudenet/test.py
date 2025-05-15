import logging
import cv2
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from detector import NudeDetector

# NSFW categories and color map (from your earlier message)
NSFW_CATEGORIES = [
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED"
]

COLOR_MAP = {
    "FEMALE_GENITALIA_COVERED": (0, 255, 0),    # Green
    "BUTTOCKS_EXPOSED": (255, 0, 0),           # Red
    "FEMALE_BREAST_EXPOSED": (0, 0, 255),      # Blue
    "FEMALE_GENITALIA_EXPOSED": (255, 255, 0), # Yellow
    "ANUS_EXPOSED": (255, 0, 255),             # Magenta
    "MALE_GENITALIA_EXPOSED": (0, 255, 255),   # Cyan
    "ANUS_COVERED": (128, 128, 128)            # Gray
}

# Paths
model_path = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\640m.onnx"
test_frame = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\testSingle\preview.jpg"
output_image_path = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\testSingle\265138_detections.jpg"

# Initialize detector
detector = NudeDetector(model_path=model_path)

# Run detection
detections = detector.detect(test_frame)
logger.info(f"Detections: {detections}")

# Function to draw bounding boxes and save the image
def draw_bounding_boxes(image_path, detections, output_path, confidence_threshold=0.3):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")

        # Draw each detection
        for detection in detections:
            class_id = detection.get('class_id')
            confidence = detection.get('confidence')
            bbox = detection.get('bbox')  # Expected: [x_min, y_min, x_max, y_max]

            if confidence >= confidence_threshold:
                class_label = NSFW_CATEGORIES[class_id]
                color = COLOR_MAP[class_label]

                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Draw rectangle (bounding box)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)

                # Add label with class and confidence
                label = f"{class_label}: {confidence:.2f}"
                cv2.putText(image, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the output image
        cv2.imwrite(output_path, image)
        logger.info(f"Output image with bounding boxes saved to {output_path}")

    except Exception as e:
        logger.error(f"Error processing image: {e}")

# Placeholder detections (based on debug logs, with assumed bbox coordinates)
placeholder_detections = [
    {
        'class_id': 3,  # FEMALE_GENITALIA_EXPOSED
        'confidence': 0.4186,
        'bbox': [100, 150, 200, 250]  # [x_min, y_min, x_max, y_max]
    },
    {
        'class_id': 3,  # FEMALE_GENITALIA_EXPOSED
        'confidence': 0.3273,
        'bbox': [120, 170, 220, 270]
    }
]

# Draw bounding boxes and save the image
draw_bounding_boxes(test_frame, placeholder_detections, output_image_path)
# Replace placeholder_detections with detections when using real data:
# draw_bounding_boxes(test_frame, detections, output_image_path)