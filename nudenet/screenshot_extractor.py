import os
import cv2
import re
import mysql.connector
from mysql.connector import Error
import logging
import shutil
from pathlib import Path
import json

class ScreenshotExtractor:
    def __init__(self, video_dir, output_dir, properties_file, video_name, min_free_space_mb=5000, max_screenshots=None, crop_to_box=False, check_processed=True):
        """Initialize the ScreenshotExtractor.

        Args:
            video_dir (str): Directory containing videos (e.g., 'C:\\Users\\Jimmy\\Documents\\TestFolder').
            output_dir (str): Directory to save screenshots (e.g., 'C:\\Users\\Jimmy\\PycharmProjects\\NudeNetClassifier\\Screenshots').
            properties_file (str): Path to Integration.Properties file.
            video_name (str): Specific video name to process (e.g., 'video1').
            min_free_space_mb (int): Minimum free disk space in MB to continue processing (default: 5000MB = 5GB).
            max_screenshots (int, optional): Maximum number of screenshots to process (default: None for unlimited).
            crop_to_box (bool): Crop screenshots to bounding box if True, else save full frame (default: False).
            check_processed (bool): Check video_processing table to ensure video was processed (default: True).
        """
        self.video_name = re.sub(r'[^a-zA-Z0-9_]', '_', video_name)
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.properties_file = properties_file
        self.min_free_space_mb = min_free_space_mb
        self.max_screenshots = max_screenshots
        self.crop_to_box = crop_to_box
        self.check_processed = check_processed
        self.logger = logging.getLogger(__name__)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Supported video extensions
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpeg', '.mpg'}

        # Log initialization
        self.logger.info(f"Initialized ScreenshotExtractor for video_name={self.video_name}, crop_to_box={self.crop_to_box}")

        # Validate properties file
        if not os.path.exists(self.properties_file):
            self.logger.error(f"Properties file not found: {self.properties_file}")
            raise FileNotFoundError(f"Properties file not found: {self.properties_file}")

    def read_properties_file(self):
        """Read database credentials from Integration.Properties."""
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
        """Initialize database connection."""
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
        except Error as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def is_video_processed(self, connection):
        """Check if video is processed in video_processing table."""
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
        except Error as e:
            self.logger.error(f"Failed to check if video {self.video_name} is processed: {e}")
            return False

    def get_unsafe_records(self, connection):
        """Retrieve records from the video-specific table."""
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
        except Error as e:
            self.logger.error(f"Failed to retrieve records from table {table_name}: {e}")
            return []

    def capture_screenshot(self, video_path, timestamp, filename, class_name, score, box, classifier_unsafe_score, classifier_safe_score):
        """Capture a screenshot from the video at the given timestamp."""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return False

            # Get FPS and calculate frame number
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                self.logger.error(f"Invalid FPS for {video_path}")
                cap.release()
                return False

            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                self.logger.error(f"Failed to read frame at {timestamp}s in {video_path}")
                cap.release()
                return False

            # Create subfolder
            video_subfolder = os.path.join(self.output_dir, self.video_name)
            os.makedirs(video_subfolder, exist_ok=True)

            # Crop to bounding box if enabled
            if self.crop_to_box and box:
                x, y, w, h = box
                frame = frame[y:y+h, x:x+w]
                if frame.size == 0:
                    self.logger.warning(f"Empty crop for box {box} in {video_path}")
                    cap.release()
                    return False

            # Generate output filename
            output_filename = filename.replace('_frame_', '_screenshot_')
            output_path = os.path.join(video_subfolder, output_filename)

            # Save screenshot
            cv2.imwrite(output_path, frame)
            self.logger.info(f"Saved screenshot: {output_path}")
            cap.release()
            return True
        except Exception as e:
            self.logger.error(f"Error capturing screenshot from {video_path} at {timestamp}s: {e}")
            return False

    def process_screenshots(self):
        """Process records from the video-specific table and capture screenshots."""
        processed_count = 0
        try:
            # Initialize database connection
            connection = self.init_db_connection()
            try:
                # Check if video was processed
                if not self.is_video_processed(connection):
                    self.logger.error(f"Cannot process screenshots: Video {self.video_name} not processed or unprocessed")
                    return

                # Get records from video-specific table
                records = self.get_unsafe_records(connection)
                self.logger.info(f"Found {len(records)} records to process for {self.video_name}")

                for filename, timestamp, class_name, score, box, classifier_unsafe_score, classifier_safe_score in records:
                    # Check disk space
                    free_space_mb = self.check_disk_space()
                    if free_space_mb < self.min_free_space_mb:
                        self.logger.warning(
                            f"Stopping: Free disk space ({free_space_mb:.2f} MB) below threshold ({self.min_free_space_mb} MB)")
                        break

                    # Check screenshot limit
                    if self.max_screenshots is not None and processed_count >= self.max_screenshots:
                        self.logger.warning(f"Stopping: Reached maximum screenshots ({self.max_screenshots})")
                        break

                    # Find video file
                    video_path = None
                    for ext in self.video_extensions:
                        test_path = os.path.join(self.video_dir, self.video_name + ext)
                        if os.path.exists(test_path):
                            video_path = test_path
                            break

                    if video_path is None:
                        self.logger.warning(f"Video not found for {self.video_name}")
                        break

                    # Capture screenshot
                    if self.capture_screenshot(video_path, timestamp, filename, class_name, score, box, classifier_unsafe_score, classifier_safe_score):
                        processed_count += 1
                        # Mark record as processed
                        cursor = connection.cursor()
                        table_name = self.video_name
                        cursor.execute(f"UPDATE `{table_name}` SET processed=1 WHERE filename=%s", (filename,))
                        connection.commit()
                        cursor.close()

            finally:
                if connection.is_connected():
                    connection.close()
                    self.logger.info("Database connection closed")

        except Exception as e:
            self.logger.error(f"Failed to process screenshots: {e}")

        self.logger.info(f"Processed {processed_count} screenshots")

    def check_disk_space(self):
        """Check available disk space in MB."""
        try:
            total, used, free = shutil.disk_usage(self.output_dir)
            free_mb = free / (1024 * 1024)  # Convert bytes to MB
            return free_mb
        except Exception as e:
            self.logger.error(f"Failed to check disk space: {e}")
            return float('inf')

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(r"C:\Users\Jimmy\PycharmProjects\NudeNetClassifier\screenshot_log.txt"),
            logging.StreamHandler()
        ]
    )

    # Example usage for multiple videos
    def get_processed_videos(properties_file):
        try:
            properties = {}
            with open(properties_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        properties[key.strip()] = value.strip()
            db_user = properties['db_user']
            db_password = properties['db_password']
            db_name = properties['db_nn']

            connection = mysql.connector.connect(
                host='localhost',
                port=3306,
                database=db_name,
                user=db_user,
                password=db_password
            )

            cursor = connection.cursor()
            query = """
            SELECT video_name FROM video_processing
            WHERE processed = 1
            """
            cursor.execute(query)
            video_names = [row[0] for row in cursor.fetchall()]
            cursor.close()
            connection.close()
            return video_names
        except Exception as e:
            logging.error(f"Failed to get processed videos: {e}")
            return []

    video_names = get_processed_videos(r"C:\Users\Jimmy\Documents\Integration.Properties")
    logging.info(f"Found {len(video_names)} processed videos: {video_names}")

    for video_name in video_names:
        logging.info(f"Processing screenshots for {video_name}")
        extractor = ScreenshotExtractor(
            video_dir=r"C:\Users\Jimmy\Documents\TestFolder",
            output_dir=r"C:\Users\Jimmy\PycharmProjects\NudeNetClassifier\Screenshots",
            properties_file=r"C:\Users\Jimmy\Documents\Integration.Properties",
            video_name=video_name,
            min_free_space_mb=5000,
            crop_to_box=True,  # Crop to bounding boxes
            check_processed=True
        )
        extractor.process_screenshots()