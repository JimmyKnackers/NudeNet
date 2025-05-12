# test_detector.py
import os
import sys
import logging
import cv2

# Add nudenet directory to Python path
sys.path.insert(0, r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet")
from detector import NudeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r"C:\Users\Jimmy\Desktop\detector_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

try:
    detector = NudeDetector()
    logger.info("NudeDetector initialized successfully")
    image_path = r"C:\Users\Jimmy\Desktop\265138.jpg"
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        exit(1)
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Invalid image: {image_path}")
        exit(1)
    height, width = img.shape[:2]
    logger.info(f"Image size: {width}x{height}")
    detections = detector.detect(image_path)
    logger.info(f"Detections for {image_path}: {detections}")
    print("Detections:", detections)
    batch_detections = detector.detect_batch([image_path], batch_size=1)
    logger.info(f"Batch detections: {batch_detections}")
    print("Batch detections:", batch_detections)
except Exception as e:
    logger.error(f"Error: {e}")
    raise