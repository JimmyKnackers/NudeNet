import os
import sys
import logging
import cv2
import numpy as np
import mysql.connector
import json
import re
import time
import multiprocessing
import psutil
import os
from datetime import datetime
from pathlib import Path
import shutil
import multiprocessing
from functools import partial

# Add nudenet directory to Python path
sys.path.insert(0, r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet")
from detector import NudeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r"C:\Users\Jimmy\Documents\GitHub\NudeNet\processing_log.txt"),
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
    "BUTTOCKS_EXPOSED": (255, 0, 0),  # Red
    "FEMALE_BREAST_EXPOSED": (0, 0, 255),  # Blue
    "FEMALE_GENITALIA_EXPOSED": (255, 255, 0),  # Cyan
    "ANUS_EXPOSED": (255, 0, 255),  # Magenta
    "MALE_GENITALIA_EXPOSED": (0, 255, 255),  # Yellow
    "ANUS_COVERED": (128, 128, 128)  # Gray
}

# Output directory for visualized frames
output_dir = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\detected_frames"
os.makedirs(output_dir, exist_ok=True)


class ScreenshotExtractor:
    def __init__(self, video_dir, output_dir, properties_file, video_name, min_free_space_mb=5000, max_screenshots=None,
                 crop_to_box=False, check_processed=True):
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
        self.logger.info(
            f"Initialized ScreenshotExtractor for video_name={self.video_name}, crop_to_box={self.crop_to_box}")

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

    def capture_screenshot(self, video_path, timestamp, filename, class_name, score, box, classifier_unsafe_score,
                           classifier_safe_score):
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
                frame = frame[y:y + h, x:x + w]
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


def process_video(video_path, detector, conn, cursor, extractor, frame_skip=10, min_score=0.5, model_label="640m",
                  max_frames=None):
    """Process a single video with NudeDetector."""
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    logger.info(f"Processing video: {video_name} with model {model_label}")

    # Check disk space
    free_space_mb = extractor.check_disk_space()
    if free_space_mb < extractor.min_free_space_mb:
        logger.error(
            f"Insufficient disk space: {free_space_mb:.2f} MB available, {extractor.min_free_space_mb} MB required")
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
        logger.info(
            f"Processed batch of {len(batch_paths)} frames in {batch_end - batch_start:.3f} seconds with {model_label}")

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
    logger.info(
        f"Total processing time for {video_name} with {model_label}: {end_time - start_time:.3f} seconds, {detection_count} detections")
    return detection_count

def process_single_video(video_file, video_dir, detector_model_path, extractor_params, video_extensions, frame_skip=10,
                         min_score=0.5, model_label="640m", max_frames=None):
    """Process a single video in a separate process."""
    video_path = os.path.join(video_dir, video_file)
    video_name = video_file.rsplit('.', 1)[0]

    process_logger = logging.getLogger(f"{__name__}.{video_name}")
    process_logger.info(f"Starting process for video: {video_name}")

    try:
        # Validate file readability
        try:
            with open(video_path, 'rb') as f:
                f.read(1)
        except Exception as e:
            process_logger.error(f"Cannot read video file {video_path}: {e}")
            # Log error in database
            video_name_db = f"{video_name}_640m"
            conn = mysql.connector.connect(
                host='localhost',
                port=3306,
                database=extractor_params['db_name'],
                user=extractor_params['db_user'],
                password=extractor_params['db_password']
            )
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO video_processing (video_name, processed, error_message)
                VALUES (%s, 0, %s)
                ON DUPLICATE KEY UPDATE error_message = %s, processed = 0
            """, (video_name_db, str(e), str(e)))
            conn.commit()
            cursor.close()
            conn.close()
            return video_name, False

        # Test video file with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"OpenCV cannot open video file {video_path}"
            process_logger.error(error_msg)
            # Log error in database
            video_name_db = f"{video_name}_640m"
            conn = mysql.connector.connect(
                host='localhost',
                port=3306,
                database=extractor_params['db_name'],
                user=extractor_params['db_user'],
                password=extractor_params['db_password']
            )
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO video_processing (video_name, processed, error_message)
                VALUES (%s, 0, %s)
                ON DUPLICATE KEY UPDATE error_message = %s, processed = 0
            """, (video_name_db, error_msg, error_msg))
            conn.commit()
            cursor.close()
            conn.close()
            cap.release()
            return video_name, False
        cap.release()

        # Initialize ScreenshotExtractor
        extractor = ScreenshotExtractor(
            video_dir=video_dir,
            output_dir=extractor_params['output_dir'],
            properties_file=extractor_params['properties_file'],
            video_name=video_name,
            crop_to_box=extractor_params['crop_to_box'],
            check_processed=extractor_params['check_processed']
        )

        # Create database connection
        db_user, db_password, db_name = extractor.read_properties_file()
        conn = mysql.connector.connect(
            host='localhost',
            port=3306,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        process_logger.info(f"Process for {video_name} connected to MySQL database")

        # Initialize NudeDetector
        detector = NudeDetector(model_path=detector_model_path)

        # Process the video
        detection_count = process_video(
            video_path=video_path,
            detector=detector,
            conn=conn,
            cursor=cursor,
            extractor=extractor,
            frame_skip=frame_skip,
            min_score=min_score,
            model_label=model_label,
            max_frames=max_frames
        )
        process_logger.info(f"Completed processing {video_name} with {detection_count} detections")

        # Log completion to processed_videos.txt
        processed_file = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\processed_videos.txt"
        with open(processed_file, "a") as f:
            f.write(f"{video_name}\n")

        # Update video_processing to mark as processed
        video_name_db = f"{video_name}_640m"
        cursor.execute("""
            INSERT INTO video_processing (video_name, processed, error_message)
            VALUES (%s, 1, NULL)
            ON DUPLICATE KEY UPDATE processed = 1, error_message = NULL
        """, (video_name_db,))
        conn.commit()

    except Exception as e:
        process_logger.error(f"Error processing {video_name}: {e}")
        # Log error in database
        video_name_db = f"{video_name}_640m"
        cursor.execute("""
            INSERT INTO video_processing (video_name, processed, error_message)
            VALUES (%s, 0, %s)
            ON DUPLICATE KEY UPDATE error_message = %s, processed = 0
        """, (video_name_db, str(e), str(e)))
        conn.commit()
        return video_name, False
    finally:
        temp_dir = os.path.join(extractor_params['output_dir'], video_name)
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                process_logger.info(f"Deleted temporary frames for {video_name} in {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)
        except Exception as e:
            process_logger.error(f"Failed to clean up temporary frames for {video_name}: {e}")

        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            process_logger.info(f"Database connection closed for {video_name}")

    return video_name, True

import multiprocessing
import psutil
import os
from functools import partial
import mysql.connector
import shutil
import logging

def main(test_mode=False):
    try:
        # Set lower process priority to reduce system impact
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # Windows: Lower priority
        logger.info("Set process priority to below normal")

        # Initialize ScreenshotExtractor parameters
        extractor_params = {
            'video_dir': r"E:\Prol",
            'output_dir': r"C:\Users\Jimmy\Documents\GitHub\NudeNet\Screenshots",
            'properties_file': r"C:\Users\Jimmy\Documents\Integration.Properties",
            'crop_to_box': False,
            'check_processed': True
        }

        # Connect to MySQL
        extractor = ScreenshotExtractor(
            video_dir=extractor_params['video_dir'],
            output_dir=extractor_params['output_dir'],
            properties_file=extractor_params['properties_file'],
            video_name="placeholder",
            crop_to_box=extractor_params['crop_to_box'],
            check_processed=extractor_params['check_processed']
        )
        db_user, db_password, db_name = extractor.read_properties_file()
        conn = mysql.connector.connect(
            host='localhost',
            port=3306,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        logger.info("Main process connected to MySQL database")

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
            error_message TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

        # Process videos
        video_dir = extractor_params['video_dir']
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.mpg', '.mpeg', '.wmv')
        detector_model_path = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\640m.onnx"

        if test_mode:
            video_file = "Delinquent School Girls (1975).mkv"
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
                logger.info(f"Processing test video with {model['label']}")
                result = process_single_video(
                    video_file=video_file,
                    video_dir=video_dir,
                    detector_model_path=model["path"],
                    extractor_params=extractor_params,
                    video_extensions=video_extensions,
                    frame_skip=10,
                    model_label=model["label"],
                    max_frames=50
                )
                logger.info(f"Test result for {result[0]} with {model['label']}: {'Success' if result[1] else 'Failed'}")
        else:
            # Normal mode: Process exactly 5 unprocessed videos
            video_files = []
            for root, _, files in os.walk(video_dir):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in video_extensions):
                        video_path = os.path.join(root, f)
                        # Store relative path to handle subfolders
                        relative_path = os.path.relpath(video_path, video_dir)
                        video_files.append(relative_path)
                    else:
                        logger.debug(f"Skipping {f} in {root}: Invalid extension")

            video_files = sorted(video_files)
            logger.info(f"Found {len(video_files)} video files: {video_files}")

            if not video_files:
                logger.warning(f"No videos found in {video_dir} or its subfolders")
                return

            # Load processed videos
            processed_file = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\processed_videos.txt"
            processed_videos_file = set()
            if os.path.exists(processed_file):
                with open(processed_file, "r") as f:
                    processed_videos_file = {line.strip() for line in f if line.strip()}
                logger.info(f"Loaded {len(processed_videos_file)} processed videos from {processed_file}")

            cursor.execute("SELECT video_name FROM video_processing WHERE processed = 1")
            processed_videos_db = {f"{row[0]}" for row in cursor.fetchall()}
            # Strip '_640m' and convert to relative path format
            processed_videos = processed_videos_file | {
                os.path.splitext(v.rsplit('_640m', 1)[0])[0].replace(os.sep, '/')
                for v in processed_videos_db if v.endswith('_640m')
            }
            logger.info(f"Total processed videos: {len(processed_videos)}")

            # Filter unprocessed videos
            unprocessed_videos = []
            for f in video_files:
                base_name = os.path.splitext(f)[0].replace(os.sep, '/')
                if base_name not in processed_videos:
                    video_path = os.path.join(video_dir, f)
                    try:
                        with open(video_path, 'rb') as test_file:
                            test_file.read(1)
                        unprocessed_videos.append(f)
                    except Exception as e:
                        logger.error(f"Cannot read {f}: {e}")
                        video_name_db = f"{base_name}_640m".replace('/', '_')
                        cursor.execute("""
                            INSERT INTO video_processing (video_name, processed, error_message)
                            VALUES (%s, 0, %s)
                            ON DUPLICATE KEY UPDATE error_message = %s, processed = 0
                        """, (video_name_db, str(e), str(e)))
                        conn.commit()
                else:
                    logger.info(f"Skipping {f}: Already processed")

            # Prioritize BabysitterMassacre.avi
            target_video = 'BabysitterMassacre.avi'
            if target_video in unprocessed_videos:
                unprocessed_videos.remove(target_video)
                unprocessed_videos.insert(0, target_video)
                logger.info(f"Prioritized {target_video} for processing")
            else:
                # Check subfolders for BabysitterMassacre.avi
                for f in unprocessed_videos:
                    if os.path.basename(f) == target_video:
                        unprocessed_videos.remove(f)
                        unprocessed_videos.insert(0, f)
                        logger.info(f"Prioritized {f} for processing")
                        break

            video_files = unprocessed_videos[:5]
            logger.info(f"Selected {len(video_files)} unprocessed videos for processing: {video_files}")

            if not video_files:
                logger.info("No unprocessed videos available within the limit of 5")
                return

            max_processes = 3  # Optimized for 24-core Ultra 9 285K
            logger.info(f"Processing videos with {max_processes} concurrent processes")
            with multiprocessing.Pool(processes=max_processes) as pool:
                process_func = partial(
                    process_single_video,
                    video_dir=video_dir,
                    detector_model_path=detector_model_path,
                    extractor_params=extractor_params,
                    video_extensions=video_extensions,
                    frame_skip=30,
                    min_score=0.5,
                    model_label="640m"
                )
                results = pool.map(process_func, video_files)

            for video_name, success in results:
                logger.info(f"Processing {'succeeded' if success else 'failed'} for {video_name}")

    except KeyboardInterrupt:
        logger.warning("Script interrupted by user, cleaning up...")
        raise
    except Exception as e:
        logger.error(f"Main process error: {e}")
        raise
    finally:
        temp_dir = extractor_params['output_dir']
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Deleted main temporary directory {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                logger.info(f"Verified: {temp_dir} is empty")
            else:
                logger.warning(f"Cleanup incomplete: {temp_dir} contains files")
        except Exception as e:
            logger.error(f"Failed to clean up {temp_dir}: {e}")

        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            logger.info("Main database connection closed")

if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Windows
    multiprocessing.freeze_support()
    main(test_mode=False)  # Set to False for normal mode