import unittest
from unittest.mock import patch, MagicMock
import os
import cv2
import numpy as np
import mysql.connector
from pathlib import Path
import shutil
import json
import logging
from screenshot_extractor import ScreenshotExtractor, process_single_video, visualize_detections, NSFW_CATEGORIES, \
    COLOR_MAP

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestScreenshotExtractor(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.video_dir = r"E:\Prol"
        self.output_dir = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\Screenshots"
        self.properties_file = r"C:\Users\Jimmy\Documents\Integration.Properties"
        self.video_file = "Delinquent School Girls (1975).mkv"
        self.video_name = "Delinquent_School_Girls_1975"
        self.model_path = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\nudenet\640m.onnx"
        self.extractor_params = {
            'video_dir': self.video_dir,
            'output_dir': self.output_dir,
            'properties_file': self.properties_file,
            'crop_to_box': False,
            'check_processed': True,
            'db_user': 'test_user',
            'db_password': 'test_password',
            'db_name': 'test_db'
        }
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpeg', '.mpg'}

        # Create temporary output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize extractor
        self.extractor = ScreenshotExtractor(
            video_dir=self.video_dir,
            output_dir=self.output_dir,
            properties_file=self.properties_file,
            video_name=self.video_name,
            crop_to_box=False,
            check_processed=True
        )

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        logger.info("Cleaned up test output directory")

    @patch('cv2.VideoCapture')
    @patch('mysql.connector.connect')
    @patch('screenshot_extractor.NudeDetector')
    def test_process_single_video(self, mock_detector, mock_db_connect, mock_video_capture):
        """Test processing a single video with mock dependencies."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # Mock FPS
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Mock frame
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True
        mock_video_capture.return_value = mock_cap

        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.is_connected.return_value = True
        mock_db_connect.return_value = mock_conn
        mock_cursor.fetchall.side_effect = [
            [],  # For is_video_processed
            []  # For get_unsafe_records
        ]

        # Mock NudeDetector
        mock_detector_instance = MagicMock()
        mock_detector.return_value = mock_detector_instance
        mock_detections = [
            {
                'class': 'FEMALE_BREAST_EXPOSED',
                'score': 0.85,
                'box': [100, 100, 200, 200]
            }
        ]
        mock_detector_instance.detect_batch.return_value = [mock_detections]

        # Mock file operations
        with patch('cv2.imwrite') as mock_imwrite, \
                patch('os.path.exists', return_value=True), \
                patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'test'

            # Call process_single_video
            video_name, success = process_single_video(
                video_file=self.video_file,
                video_dir=self.video_dir,
                detector_model_path=self.model_path,
                extractor_params=self.extractor_params,
                video_extensions=self.video_extensions,
                frame_skip=10,
                min_score=0.5,
                model_label="640m",
                max_frames=50
            )

        # Assertions
        self.assertTrue(success, "Video processing should succeed")
        self.assertEqual(video_name, self.video_name, "Video name should match")
        mock_video_capture.assert_called_once_with(os.path.join(self.video_dir, self.video_file))
        mock_db_connect.assert_called()
        mock_detector.assert_called_once_with(model_path=self.model_path)
        mock_imwrite.assert_called()  # Ensure frames are saved
        mock_cursor.execute.assert_called()  # Ensure database operations occurred
        mock_conn.commit.assert_called()  # Ensure database commits occurred

        # Verify logging
        with patch('logging.Logger.info') as mock_log_info:
            logger.info("Test logging")
            mock_log_info.assert_called_with("Test logging")

    def test_visualize_detections(self):
        """Test visualize_detections function."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {
                'class': 'FEMALE_BREAST_EXPOSED',
                'score': 0.85,
                'box': [100, 100, 200, 200]
            }
        ]
        output_path = os.path.join(self.output_dir, "test_frame.jpg")

        with patch('cv2.rectangle') as mock_rectangle, \
                patch('cv2.putText') as mock_put_text, \
                patch('cv2.imwrite') as mock_imwrite:
            visualize_detections(frame, detections, output_path)

        mock_rectangle.assert_called_once_with(
            frame, (100, 100), (300, 300), COLOR_MAP['FEMALE_BREAST_EXPOSED'], 2
        )
        mock_put_text.assert_called_once_with(
            frame, 'FEMALE_BREAST_EXPOSED (0.850)', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            COLOR_MAP['FEMALE_BREAST_EXPOSED'], 2
        )
        mock_imwrite.assert_called_once_with(output_path, frame)


if __name__ == '__main__':
    unittest.main()