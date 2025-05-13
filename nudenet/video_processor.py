# video_processor.py
import os
import sys
import logging
import cv2
import numpy as np
import mysql.connector
import json
import re
import time
from datetime import datetime
from pathlib import Path
import shutil

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

# Color map for category-specific bounding boxes
COLOR_MAP = {
    "FEMALE_GENITALIA_COVERED": (0, 255, 0),  # Green
    "BUTTOCKS_EXPOSED": (255, 0, 0),          # Red
    "FEMALE_BREAST_EXPOSED": (0, 0, 255),     # Blue
    "FEMALE_GENITALIA_EXPOSED": (255, 255, 0),# Cyan
    "ANUS_EXPOSED": (255, 0, 255),            # Magenta
    "MALE_GENITALIA_EXPOSED": (0, 255, 255),  # Yellow
    "ANUS_COVERED": (128, 128, 128)           # Gray
}

# Output directory for visualized frames
output_dir = r"C:\Users\Jimmy\PycharmProjects\NudeNetClassifier\detected_frames"
os.makedirs(output_dir, exist_ok=True)

class ScreenshotExtractor:
    def __init__(self, video_dir, output_dir, properties_file, video_name, min_free_space_mb=5000, max_screenshots=None, crop_to_box=False, check_processed=True):
        self.video_name = re.sub(r'[^a-zA-Z0-9_]', '_', video_name)
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.properties_file = properties_file
        self.min_free_space_mb = min_free_space_mb
        self.max_screenshots = max_screenshots
        self.crop_to_box = crop_to_box
        self.check_processed = check_processed
        self.logger = logging.getLogger(__name__)

        os.makedirs(self.output_dir, exist_ok=True)
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpeg', '.mpg'}
        self.logger.info(f"Initialized ScreenshotExtractor for video_name={self.video_name}, crop_to_box={self.crop_to_box}")

        if not os.path.exists(self.properties_file):
            self.logger.error(f"Properties file not found: {self.properties_file}")
            raise FileNotFoundError(f"Properties file not found: {self.properties_file}")

    def read_properties_file(self):
        try:
            properties = {}
            with open(self.properties_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        properties[key.strip()] = value.strip()
            required_keys = ['db_user', 'db_password', 'db_nn']
            if not all(key in properties for key in required_keys):
                missing = [key for key in required_keys if key not in properties]
                raise ValueError(f"Missing keys in {self.properties_file}: {missing}")
            return properties['db_user'], properties['db_password'], properties['db_nn']
        except Exception as e:
            self.logger.error(f"Failed to read {self.properties_file}: {e}")
            raise

    def init_db_connection(self):
        try:
            db_user, db_password, db_name = self.read_properties_file()
            connection = mysql.connector.connect(
                host='localhost',
                port=3306,
                database=db_name,
                user=db_user,
                password=db_password
            )
            if connection.is_connected():
                self.logger.info(f"Connected to MySQL database: {db_name}")
                return connection
        except mysql.connector.Error as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def is_video_processed(self, connection):
        if not self.check_processed:
            return True
        try:
            cursor = connection.cursor()
            query = """
            SELECT processed FROM video_processing
            WHERE video_name = %s
            """
            cursor.execute(query, (self.video_name,))
            result = cursor.fetchone()
            cursor.close()
            if result is None:
                self.logger.warning(f"No record found for {self.video_name} in video_processing")
                return False
            if result[0] == 0:
                self.logger.warning(f"Video {self.video_name} is marked as unprocessed")
                return False
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Failed to check if video {self.video_name} is processed: {e}")
            return False

    def get_unsafe_records(self, connection):
        try:
            cursor = connection.cursor()
            table_name = self.video_name
            query = f"""
            SELECT filename, timestamp, class, score, box, classifier_unsafe_score, classifier_safe_score
            FROM `{table_name}`
            WHERE processed = 0
            """
            cursor.execute(query)
            records = cursor.fetchall()
            cursor.close()
            return [(filename, timestamp, class_name, score, json.loads(box), unsafe_score, safe_score)
                    for filename, timestamp, class_name, score, box, unsafe_score, safe_score in records]
        except mysql.connector.Error as e:
            self.logger.error(f"Failed to retrieve records from table {table_name}: {e}")
            return []

    def capture_screenshot(self, video_path, timestamp, filename, class_name, score, box, classifier_unsafe_score, classifier_safe_score):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                self.logger.error(f"Invalid FPS for {video_path}")
                cap.release()
                return False

            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret:
                self.logger.error(f"Failed to read frame at {timestamp}s in {video_path}")
                cap.release()
                return False

            video_subfolder = os.path.join(self.output_dir, self.video_name)
            os.makedirs(video_subfolder, exist_ok=True)

            if self.crop_to_box and box:
                x, y, w, h = box
                frame = frame[y:y+h, x:x+w]
                if frame.size == 0:
                    self.logger.warning(f"Empty crop for box {box} in {video_path}")
                    cap.release()
                    return False

            output_filename = filename.replace('_frame_', '_screenshot_')
            output_path = os.path.join(video_subfolder, output_filename)

            cv2.imwrite(output_path, frame)
            self.logger.info(f"Saved screenshot: {output_path}")
            cap.release()
            return True
        except Exception as e:
            self.logger.error(f"Error capturing screenshot from {video_path} at {timestamp}s: {e}")
            return False

    def extract_frames(self, video_path, frame_skip=10, max_frames=None):
        logger.info(f"Extracting temporary frames from {video_path} for detection")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return [], [], [], []

        frames = []
        frame_paths = []
        frame_numbers = []
        timestamps = []
        count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS is invalid
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]

        video_subfolder = os.path.join(self.output_dir, video_name)
        os.makedirs(video_subfolder, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_skip == 0:
                frame_path = os.path.join(video_subfolder, f"{video_name}_frame_{count}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame)
                frame_paths.append(frame_path)
                frame_numbers.append(count)
                timestamps.append(count / fps)
                if max_frames and len(frames) >= max_frames:
                    break
            count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} temporary frames from {video_path}")
        return frames, frame_paths, frame_numbers, timestamps

    def check_disk_space(self):
        try:
            total, used, free = shutil.disk_usage(self.output_dir)
            free_mb = free / (1024 * 1024)  # Convert bytes to MB
            return free_mb
        except Exception as e:
            self.logger.error(f"Failed to check disk space: {e}")
            return float('inf')

def visualize_detections(frame, detections, output_path):
    """Draw bounding boxes for all detected categories."""
    for detection in detections:
        x, y, w, h = detection['box']
        label = f"{detection['class']} ({detection['score']:.3f})"
        color = COLOR_MAP.get(detection['class'], (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_path, frame)
    logger.info(f"Saved visualized frame: {output_path}")

def process_video(video_path, detector, conn, cursor, extractor, frame_skip=10, min_score=0.5, model_label="640m", max_frames=None):
    """Process a single video with NudeDetector."""
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    logger.info(f"Processing video: {video_name} with model {model_label}")

    # Check disk space
    free_space_mb = extractor.check_disk_space()
    if free_space_mb < extractor.min_free_space_mb:
        logger.error(f"Insufficient disk space: {free_space_mb:.2f} MB available, {extractor.min_free_space_mb} MB required")
        return 0

    # Create movie-specific folder for detection screenshots
    movie_folder = os.path.join(output_dir, video_name, model_label)
    os.makedirs(movie_folder, exist_ok=True)

    # Extract frames
    start_time = time.time()
    frames, frame_paths, frame_numbers, timestamps = extractor.extract_frames(video_path, frame_skip, max_frames)
    if not frames:
        logger.warning(f"No frames extracted from {video_name}")
        return 0

    # Ensure video-specific table exists
    table_name = f"{video_name}_{model_label}"
    try:
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            timestamp FLOAT NOT NULL,
            class VARCHAR(50),
            score FLOAT,
            box JSON,
            classifier_unsafe_score FLOAT,
            classifier_safe_score FLOAT,
            processed TINYINT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
    except mysql.connector.Error as e:
        logger.error(f"Failed to create table {table_name}: {e}")
        return 0

    # Process frames in batches
    batch_size = 4
    detection_count = 0
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        batch_frames = frames[i:i + batch_size]
        batch_numbers = frame_numbers[i:i + batch_size]
        batch_timestamps = timestamps[i:i + batch_size]

        batch_start = time.time()
        try:
            batch_detections = detector.detect_batch(batch_paths, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            continue
        batch_end = time.time()
        logger.info(f"Processed batch of {len(batch_paths)} frames in {batch_end - batch_start:.3f} seconds with {model_label}")

        # Process detections
        for frame, detections, frame_path, frame_number, timestamp in zip(
            batch_frames, batch_detections, batch_paths, batch_numbers, batch_timestamps
        ):
            filename = os.path.basename(frame_path)
            nsfw_detections = [d for d in detections if d['class'] in NSFW_CATEGORIES and d['score'] >= min_score]

            if nsfw_detections:
                detection_count += 1
                logger.info(f"NSFW detections for frame {frame_number} with {model_label}: {nsfw_detections}")
                # Select category with highest score
                top_detection = max(nsfw_detections, key=lambda d: d['score'])
                top_category = top_detection['class']
                top_score = top_detection['score']
                box_json = json.dumps(top_detection['box'])

                # Save annotated screenshot
                score_str = f"{top_score:.3f}".replace('.', '_')
                vis_path = os.path.join(movie_folder, f"{video_name}_{top_category}_{score_str}_{frame_number}.jpg")
                visualize_detections(frame.copy(), nsfw_detections, vis_path)
                logger.info(f"Saved annotated screenshot: {vis_path}")

                # Insert into video-specific table
                try:
                    cursor.execute(f"""
                    INSERT INTO `{table_name}` (filename, timestamp, class, score, box, processed)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (filename, timestamp, top_category, top_score, box_json, 1))
                    conn.commit()
                    logger.info(f"Inserted NSFW record into {table_name} for {filename}")
                except mysql.connector.Error as e:
                    logger.error(f"Failed to insert into {table_name} for {filename}: {e}")
                    continue

                # Store in detections table
                category_flags = {cat: 0 for cat in NSFW_CATEGORIES}
                category_scores = {cat: 0.0 for cat in NSFW_CATEGORIES}
                for detection in nsfw_detections:
                    category_flags[detection['class']] = 1
                    category_scores[detection['class']] = max(category_scores[detection['class']], detection['score'])

                query = """
                INSERT INTO detections (
                    video_id, timestamp,
                    FEMALE_GENITALIA_COVERED, FEMALE_GENITALIA_COVERED_score,
                    BUTTOCKS_EXPOSED, BUTTOCKS_EXPOSED_score,
                    FEMALE_BREAST_EXPOSED, FEMALE_BREAST_EXPOSED_score,
                    FEMALE_GENITALIA_EXPOSED, FEMALE_GENITALIA_EXPOSED_score,
                    ANUS_EXPOSED, ANUS_EXPOSED_score,
                    MALE_GENITALIA_EXPOSED, MALE_GENITALIA_EXPOSED_score,
                    ANUS_COVERED, ANUS_COVERED_score,
                    model
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    f"{video_name}_{model_label}", timestamp,
                    category_flags["FEMALE_GENITALIA_COVERED"], category_scores["FEMALE_GENITALIA_COVERED"],
                    category_flags["BUTTOCKS_EXPOSED"], category_scores["BUTTOCKS_EXPOSED"],
                    category_flags["FEMALE_BREAST_EXPOSED"], category_scores["FEMALE_BREAST_EXPOSED"],
                    category_flags["FEMALE_GENITALIA_EXPOSED"], category_scores["FEMALE_GENITALIA_EXPOSED"],
                    category_flags["ANUS_EXPOSED"], category_scores["ANUS_EXPOSED"],
                    category_flags["MALE_GENITALIA_EXPOSED"], category_scores["MALE_GENITALIA_EXPOSED"],
                    category_flags["ANUS_COVERED"], category_scores["ANUS_COVERED"],
                    model_label
                )
                try:
                    cursor.execute(query, values)
                    conn.commit()
                    logger.info(f"Inserted NSFW record into detections for {filename}")
                except mysql.connector.Error as e:
                    logger.error(f"Failed to insert into detections for {filename}: {e}")
                    continue
            else:
                logger.info(f"No NSFW detections for frame {frame_number} with {model_label}")

    # Mark video as processed
    try:
        cursor.execute("""
        INSERT INTO video_processing (video_name, processed)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE processed=%s
        """, (f"{video_name}_{model_label}", 1, 1))
        conn.commit()
        logger.info(f"Marked {video_name}_{model_label} as processed in video_processing")
    except mysql.connector.Error as e:
        logger.error(f"Failed to update video_processing for {video_name}_{model_label}: {e}")

    end_time = time.time()
    logger.info(f"Total processing time for {video_name} with {model_label}: {end_time - start_time:.3f} seconds, {detection_count} detections")
    return detection_count

def main(test_mode=False):
    try:
        # Initialize ScreenshotExtractor
        extractor = ScreenshotExtractor(
            video_dir=r"C:\Users\Jimmy\Documents\TestFolder",
            output_dir=r"C:\Users\Jimmy\PycharmProjects\NudeNetClassifier\Screenshots",
            properties_file=r"C:\Users\Jimmy\Documents\Integration.Properties",
            video_name="placeholder",
            crop_to_box=False,
            check_processed=False
        )

        # Connect to MySQL
        db_user, db_password, db_name = extractor.read_properties_file()
        conn = mysql.connector.connect(
            host='localhost',
            port=3306,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        logger.info("Connected to MySQL database")

        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            video_id VARCHAR(255) NOT NULL,
            timestamp FLOAT NOT NULL,
            FEMALE_GENITALIA_COVERED TINYINT DEFAULT 0,
            FEMALE_GENITALIA_COVERED_score FLOAT DEFAULT 0.0,
            BUTTOCKS_EXPOSED TINYINT DEFAULT 0,
            BUTTOCKS_EXPOSED_score FLOAT DEFAULT 0.0,
            FEMALE_BREAST_EXPOSED TINYINT DEFAULT 0,
            FEMALE_BREAST_EXPOSED_score FLOAT DEFAULT 0.0,
            FEMALE_GENITALIA_EXPOSED TINYINT DEFAULT 0,
            FEMALE_GENITALIA_EXPOSED_score FLOAT DEFAULT 0.0,
            ANUS_EXPOSED TINYINT DEFAULT 0,
            ANUS_EXPOSED_score FLOAT DEFAULT 0.0,
            MALE_GENITALIA_EXPOSED TINYINT DEFAULT 0,
            MALE_GENITALIA_EXPOSED_score FLOAT DEFAULT 0.0,
            ANUS_COVERED TINYINT DEFAULT 0,
            ANUS_COVERED_score FLOAT DEFAULT 0.0,
            model VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_detection (video_id, timestamp, model)
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_processing (
            video_name VARCHAR(255) PRIMARY KEY,
            processed TINYINT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

        # Process videos
        video_dir = r"C:\Users\Jimmy\Documents\TestFolder"
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov')

        if test_mode:
            # Test mode: Compare 320n.onnx and 640m.onnx on one video
            video_file = "Delinquent School Girls (1975).mkv"  # Change to your test video
            video_path = os.path.join(video_dir, video_file)
            if not os.path.exists(video_path):
                logger.error(f"Test video not found: {video_path}")
                return

            extractor.video_name = video_file.rsplit('.', 1)[0]
            models = [
                {"path": r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\320n.onnx", "label": "320n"},
                {"path": r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\640m.onnx", "label": "640m"}
            ]

            for model in models:
                if not os.path.exists(model["path"]):
                    logger.error(f"Model not found: {model['path']}")
                    continue
                logger.info(f"Initializing NudeDetector with {model['label']}")
                detector = NudeDetector(model_path=model["path"])
                process_video(video_path, detector, conn, cursor, extractor, frame_skip=10, model_label=model["label"], max_frames=50)
        else:
            # Normal mode: Process all videos with 640m.onnx
            detector = NudeDetector(model_path=r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\640m.onnx")
            logger.info("NudeDetector initialized with 640m.onnx")
            for video_file in os.listdir(video_dir):
                if video_file.lower().endswith(video_extensions):
                    video_path = os.path.join(video_dir, video_file)
                    extractor.video_name = video_file.rsplit('.', 1)[0]
                    process_video(video_path, detector, conn, cursor, extractor, frame_skip=10)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        # Clean up all temporary frames
        temp_dir = r"C:\Users\Jimmy\PycharmProjects\NudeNetClassifier\Screenshots"
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Deleted all temporary frames in {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                logger.info(f"Verified: {temp_dir} is empty")
            else:
                logger.warning(f"Cleanup incomplete: {temp_dir} contains files")
        except Exception as e:
            logger.error(f"Failed to delete temporary frames in {temp_dir}: {e}")

        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main(test_mode=False)  # Set to False for normal mode