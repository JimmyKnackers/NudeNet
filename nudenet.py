# video_processor.py
import configparser
import logging
import os
import sys
import time

import cv2
import mysql.connector

# Add nudenet directory to Python path
sys.path.insert(0, r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet")
from detector import NudeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r"C:\Users\Jimmy\PycharmProjects\NudeNetClassifier\processing_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Read database credentials from Integration.Properties
config = configparser.ConfigParser()
config.read(r"C:\Users\Jimmy\Documents\Integration.Properties")
db_config = {
    'user': config.get('Database', 'db_user', fallback='your_db_user'),
    'password': config.get('Database', 'db_password', fallback='your_db_password'),
    'host': 'localhost',
    'database': config.get('Database', 'db_nn', fallback='nudenet')
}

# Output directory for visualized frames
output_dir = r"C:\Users\Jimmy\Desktop\detected_frames"
os.makedirs(output_dir, exist_ok=True)

def extract_frames(video_path, frame_skip=10):
    """Extract frames from a video using OpenCV (fallback for ScreenshotExtractor)."""
    logger.info(f"Extracting frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []

    frames = []
    frame_paths = []
    frame_numbers = []
    count = 0
    video_name = os.path.basename(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame)
            frame_paths.append(frame_path)
            frame_numbers.append(count)
        count += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames, frame_paths, frame_numbers

def visualize_detections(frame, detections, output_path):
    """Draw bounding boxes and labels on the frame and save it."""
    for detection in detections:
        x, y, w, h = detection['box']
        label = f"{detection['class']} ({detection['score']:.2f})"
        color = (0, 255, 0)  # Green for NSFW detections
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_path, frame)
    logger.info(f"Saved visualized frame: {output_path}")

def process_video(video_path, detector, conn, cursor, frame_skip=10):
    """Process a single video with NudeDetector."""
    video_name = os.path.basename(video_path)
    logger.info(f"Processing video: {video_name}")

    # Extract frames (replace with ScreenshotExtractor if available)
    frames, frame_paths, frame_numbers = extract_frames(video_path, frame_skip)
    if not frames:
        logger.warning(f"No frames extracted from {video_name}")
        return

    # Process frames in batches
    batch_size = 4
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        batch_frames = frames[i:i + batch_size]
        batch_numbers = frame_numbers[i:i + batch_size]

        start_time = time.time()
        batch_detections = detector.detect_batch(batch_paths, batch_size=batch_size)
        end_time = time.time()
        logger.info(f"Processed batch of {len(batch_paths)} frames in {end_time - start_time:.3f} seconds")

        # Save detections and visualize
        for frame, detections, frame_path, frame_number in zip(batch_frames, batch_detections, batch_paths, batch_numbers):
            if detections:
                # Visualize detections
                vis_path = frame_path.replace(".jpg", "_detected.jpg")
                visualize_detections(frame, detections, vis_path)

                # Store detections in database
                for detection in detections:
                    x, y, w, h = detection['box']
                    query = """
                    INSERT INTO detections (video_id, frame_number, class, score, box_x, box_y, box_w, box_h)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (
                        video_name, frame_number, detection['class'], detection['score'],
                        x, y, w, h
                    )
                    try:
                        cursor.execute(query, values)
                        conn.commit()
                    except mysql.connector.Error as e:
                        logger.error(f"Database error: {e}")
                        continue

def main():
    try:
        # Initialize NudeDetector with 640m.onnx
        detector = NudeDetector(model_path=r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\640m.onnx")
        logger.info("NudeDetector initialized with 640m.onnx")

        # Connect to MySQL
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        logger.info("Connected to MySQL database")

        # Process videos
        video_dir = r"C:\Users\Jimmy\Documents\TestFolder"
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
        for video_file in os.listdir(video_dir):
            if video_file.lower().endswith(video_extensions):
                video_path = os.path.join(video_dir, video_file)
                process_video(video_path, detector, conn, cursor, frame_skip=10)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()