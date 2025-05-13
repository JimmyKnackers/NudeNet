# image_nsfw_detector.py
import os
import logging
import cv2
import numpy as np
import sys

# Add nudenet directory to Python path
sys.path.insert(0, r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet")
from detector import NudeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r"C:\Users\Jimmy\Desktop\nsfw_detector_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Target NSFW categories
NSFW_CATEGORIES = [
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED"
]

# Color map for bounding boxes
COLOR_MAP = {
    "FEMALE_GENITALIA_COVERED": (0, 255, 0),  # Green
    "BUTTOCKS_EXPOSED": (255, 0, 0),  # Red
    "FEMALE_BREAST_EXPOSED": (0, 0, 255),  # Blue
    "FEMALE_GENITALIA_EXPOSED": (255, 255, 0),  # Cyan
    "ANUS_EXPOSED": (255, 0, 255),  # Magenta
    "MALE_GENITALIA_EXPOSED": (0, 255, 255),  # Yellow
    "ANUS_COVERED": (128, 128, 128)  # Gray
}


class ImageNSFWDetector:
    def __init__(self, model_path=r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\640m.onnx",
                 output_dir=r"C:\Users\Jimmy\Desktop\detected_frames\test"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.min_score = 0.5
        try:
            self.detector = NudeDetector(model_path=self.model_path)
            logger.info("NudeDetector initialized with 640m.onnx")
        except Exception as e:
            logger.error(f"Failed to initialize NudeDetector: {e}")
            raise

    def visualize_detections(self, frame, detections, output_path):
        """Draw bounding boxes for detected categories."""
        for detection in detections:
            x, y, w, h = detection['box']
            label = f"{detection['class']} ({detection['score']:.3f})"
            color = COLOR_MAP.get(detection['class'], (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(output_path, frame)
        logger.info(f"Saved visualized frame: {output_path}")

    def process_image(self, image_path, image_name="test"):
        """Process a single image for NSFW detections."""
        logger.info(f"Processing image: {image_path}")

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Detect
        try:
            detections = self.detector.detect(image_path)
            logger.info(f"Raw detections: {detections}")
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return

        # Filter NSFW detections
        nsfw_detections = [d for d in detections if d['class'] in NSFW_CATEGORIES and d['score'] >= self.min_score]
        logger.info(f"NSFW detections: {nsfw_detections}")

        if nsfw_detections:
            # Select top detection
            top_detection = max(nsfw_detections, key=lambda d: d['score'])
            top_category = top_detection['class']
            top_score = top_detection['score']
            score_str = f"{top_score:.3f}".replace('.', '_')

            # Save screenshots
            plain_path = os.path.join(self.output_dir, f"{image_name}_{top_category}_{score_str}.jpg")
            vis_path = plain_path.replace(".jpg", "_detected.jpg")
            cv2.imwrite(plain_path, frame)
            self.visualize_detections(frame.copy(), nsfw_detections, vis_path)
            logger.info(f"Saved screenshots: {plain_path}, {vis_path}")
        else:
            logger.info("No NSFW detections found")


def main():
    try:
        detector = ImageNSFWDetector()
        # Test with a single image (replace with your test image path)
        test_image = r"C:\Users\Jimmy\Desktop\Delinquent School Girls (1975)_FEMALE_GENITALIA_COVERED_0_598_68000.jpg"
        detector.process_image(test_image, image_name="test")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()