import os
import _io
import math
import cv2
import numpy as np
import onnxruntime
import logging
from onnxruntime.capi import _pybind_state as C

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__labels = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
    "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
    "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

def _read_image(image_path, target_size=640):
    logger.info(f"Reading image: {image_path}")
    if isinstance(image_path, str):
        mat = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        mat = image_path
    elif isinstance(image_path, bytes):
        mat = cv2.imdecode(np.frombuffer(image_path, np.uint8), -1)
    elif isinstance(image_path, _io.BufferedReader):
        mat = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), -1)
    else:
        raise ValueError("Image_path must be str, np.ndarray, bytes, or BufferedReader")

    if mat is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_original_width, image_original_height = mat.shape[1], mat.shape[0]
    logger.info(f"Original image size: {image_original_width}x{image_original_height}")

    mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

    max_size = max(mat_c3.shape[:2])
    x_pad = max_size - mat_c3.shape[1]
    x_ratio = max_size / mat_c3.shape[1]
    y_pad = max_size - mat_c3.shape[0]
    y_ratio = max_size / mat_c3.shape[0]

    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)
    logger.info(f"Padded image size: {mat_pad.shape[1]}x{mat_pad.shape[0]}")

    input_blob = cv2.dnn.blobFromImage(
        mat_pad, 1 / 255.0, (target_size, target_size), (0, 0, 0), swapRB=True, crop=False,
    )
    logger.info(f"Input blob shape: {input_blob.shape}")

    return (
        input_blob, x_ratio, y_ratio, x_pad, y_pad, image_original_width, image_original_height,
    )

def _postprocess(
    output, x_pad, y_pad, x_ratio, y_ratio, image_original_width, image_original_height,
    model_width, model_height,
):
    logger.info(f"Postprocessing output shape: {output[0].shape}")
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.1:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0:4]
            x = x - w / 2
            y = y - h / 2
            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height
            x = max(0, min(x, image_original_width))
            y = max(0, min(y, image_original_height))
            w = min(w, image_original_width - x)
            h = min(h, image_original_height - y)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])

    logger.info(f"Pre-NMS detections: {len(boxes)}")
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.1, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        x, y, w, h = box
        detections.append(
            {
                "class": __labels[class_id],
                "score": float(score),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )

    logger.info(f"Final detections: {detections}")
    return detections

class NudeDetector:
    def __init__(self, model_path=None, providers=None, inference_resolution=640):
        default_model_path = os.path.join(os.path.dirname(__file__), "640m.onnx")
        model_path = model_path or default_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info(f"Loading ONNX model from: {model_path}")
        self.onnx_session = onnxruntime.InferenceSession(
            model_path, providers=C.get_available_providers() if not providers else providers,
        )
        logger.info(f"Available providers: {C.get_available_providers()}")
        model_inputs = self.onnx_session.get_inputs()
        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.input_name = model_inputs[0].name

    def detect(self, image_path):
        (
            preprocessed_image, x_ratio, y_ratio, x_pad, y_pad,
            image_original_width, image_original_height,
        ) = _read_image(image_path, self.input_width)
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(
            outputs, x_pad, y_pad, x_ratio, y_ratio,
            image_original_width, image_original_height,
            self.input_width, self.input_height,
        )
        return detections

    def detect_batch(self, image_paths, batch_size=4):
        logger.info(f"Processing batch of {len(image_paths)} images")
        all_detections = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            batch_inputs = []
            batch_metadata = []
            for image_path in batch:
                (
                    preprocessed_image, x_ratio, y_ratio, x_pad, y_pad,
                    image_original_width, image_original_height,
                ) = _read_image(image_path, self.input_width)
                batch_inputs.append(preprocessed_image)
                batch_metadata.append(
                    (x_ratio, y_ratio, x_pad, y_pad, image_original_width, image_original_height)
                )
            batch_input = np.vstack(batch_inputs)
            logger.info(f"Batch input shape: {batch_input.shape}")
            outputs = self.onnx_session.run(None, {self.input_name: batch_input})
            logger.info(f"Batch output shape: {outputs[0].shape}")
            for j, metadata in enumerate(batch_metadata):
                (
                    x_ratio, y_ratio, x_pad, y_pad,
                    image_original_width, image_original_height,
                ) = metadata
                detections = _postprocess(
                    [outputs[0][j : j + 1]], x_pad, y_pad, x_ratio, y_ratio,
                    image_original_width, image_original_height,
                    self.input_width, self.input_height,
                )
                all_detections.append(detections)
        return all_detections