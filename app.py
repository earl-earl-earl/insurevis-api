# app.py
import os
import io
import time
import traceback
import numpy as np
import cv2
import onnxruntime as ort
# from shapely.geometry import Polygon # Keep if needed, else remove
from PIL import Image
from typing import List, Dict, Tuple
import json

# --- Flask Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS # <-- Added CORS Import

# --- Detectron2 Imports ---
try:
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
except ImportError as e:
    print(f"Error importing Detectron2. Please ensure it is installed along with its dependencies (like torch).")
    print(f"Details: {e}")
    exit(1) # Exit if core components are missing

# --- gdown Import ---
# gdown import is removed as download functionality is removed
# try:
#     import gdown
# except ImportError:
#     print("Error importing gdown. Please install it using: pip install gdown")
#     exit(1)


# ===========================
# ⚙️ Load Configuration from JSON
# ===========================
CONFIG_FILE_PATH = "config.json"

def load_config(path: str) -> Dict:
    """Loads configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded successfully from {path}")
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file not found at {path}")
        print("Please create a config.json file based on the required structure.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Failed to decode JSON from {path}: {e}")
        print("Please check your config.json for syntax errors.")
        exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred while loading config: {e}")
        traceback.print_exc()
        exit(1)

# Load the configuration globally
CONFIG = load_config(CONFIG_FILE_PATH)

# ===========================
# ⚙️ Extract Config Values into Constants (or use CONFIG directly)
# ===========================

# --- Model Paths (Keep using environment variables or direct paths for flexibility) ---
MODEL_DIR = os.environ.get("MODEL_DIR", ".")
# Ensure MODEL_DIR exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created model directory: {MODEL_DIR}")

PART_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Parts Segmentation Model.pth")
DAMAGE_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Damage Segmentation Model.pth")
SEVERITY_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "Severity Classification Model.onnx")
DAMAGE_TYPE_DETECT_MODEL_PATH = os.path.join(MODEL_DIR, "Damage Type Object Detection Model.onnx")

# --- Google Drive IDs (Removed - No longer used for download) ---
# !! IMPORTANT: Ensure models are present at the paths defined above !!


# --- Extract Model Parameters from CONFIG ---
MASKRCNN_CONFIG_FILE = CONFIG['model_params'].get('detectron2_base_config') # e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
if MASKRCNN_CONFIG_FILE is None:
    print("FATAL ERROR: 'detectron2_base_config' missing in config.json under 'model_params'.")
    exit(1)

PART_SEG_CONF_THRES = CONFIG['model_params'].get('part_seg_conf_thres', 0.5)
DAMAGE_SEG_CONF_THRES = CONFIG['model_params'].get('damage_seg_conf_thres', 0.5)
DAMAGE_DETECTOR_INPUT_SIZE = tuple(CONFIG['model_params'].get('damage_detector_input_size', [640, 640])) # Convert list to tuple
DAMAGE_DETECTOR_CONF_THRESHOLD = CONFIG['model_params'].get('damage_detector_conf_threshold', 0.5)
DAMAGE_DETECTOR_IOU_THRESHOLD = CONFIG['model_params'].get('damage_detector_iou_threshold', 0.4)
SEVERITY_CLASSIFIER_INPUT_SIZE = tuple(CONFIG['model_params'].get('severity_classifier_input_size', [224, 224]))

# --- Extract Class Names from CONFIG ---
car_part_classes = CONFIG['class_names'].get('car_parts', [])
damage_segmentation_class_names = CONFIG['class_names'].get('damage_segmentation', [])
severity_names = CONFIG['class_names'].get('severity', ["Low", "Medium", "High"])
damage_type_names = CONFIG['class_names'].get('damage_types', ["Dent", "Scratch", "Crack", "Broken"])

if not car_part_classes or not damage_segmentation_class_names or not damage_type_names:
     print("FATAL ERROR: Class names lists ('car_parts', 'damage_segmentation', 'damage_types') are missing or empty in config.json.")
     exit(1)


# --- Extract Cost Tables from CONFIG ---
part_base_costs = CONFIG['costs'].get('part_base', {})
damage_multipliers = CONFIG['costs'].get('damage_multipliers', {})
if not part_base_costs or not damage_multipliers:
     print("Warning: Cost tables ('part_base', 'damage_multipliers') are missing or empty in config.json. Cost estimation may be inaccurate.")


# --- Extract Processing Parameters from CONFIG ---
COST_ESTIMATION_IOU_THRESHOLD = CONFIG['processing_params'].get('cost_estimation_iou_threshold', 0.3)
DAMAGE_CLASS_LABEL_IN_SEGMENTER = CONFIG['processing_params'].get('damage_class_label_in_segmenter', 'damage')

# --- Derive Damage Class Index (Important!) ---
try:
    DAMAGE_CLASS_INDEX_IN_SEGMENTER = damage_segmentation_class_names.index(DAMAGE_CLASS_LABEL_IN_SEGMENTER)
    print(f"Derived damage class index for label '{DAMAGE_CLASS_LABEL_IN_SEGMENTER}' is: {DAMAGE_CLASS_INDEX_IN_SEGMENTER}")
except ValueError:
    print(f"FATAL ERROR: Damage label '{DAMAGE_CLASS_LABEL_IN_SEGMENTER}' defined in config "
          f"not found in damage_segmentation class list: {damage_segmentation_class_names}")
    exit(1)

# ===========================
#  Helper Function for Model Download (REMOVED)
# ===========================
# The download_model_if_not_exists function is removed.

# ===========================
#  Download Models (if necessary) (REMOVED)
# ===========================
# The code block checking and downloading models is removed.
# The application now assumes the models are already present at the specified paths.
# Loading code below includes checks for file existence.


# ===========================
# 🧠 Model Loading (Uses constants derived from config)
# ===========================

# --- Detectron2 Model Loader Helper ---
def load_detectron2_model(config_path, weight_path, class_names_for_maskrcnn, conf_thres):
    # Check if weights file exists BEFORE attempting to load
    if not os.path.exists(weight_path):
         print(f"FATAL ERROR: Detectron2 model weights not found at: {weight_path}")
         # This is a fatal error now that download is removed.
         raise FileNotFoundError(f"Detectron2 model weights not found: {weight_path}. Ensure the model is placed in the '{MODEL_DIR}' directory.")
    try:
        cfg = get_cfg()
        # Use a local config file path or a model_zoo path
        if os.path.exists(config_path):
             cfg.merge_from_file(config_path)
             print(f"Using local Detectron2 config file: {config_path}")
        else:
             # If config_path is like "COCO-InstanceSegmentation/...", use model_zoo
             # Note: This requires the config file itself to be available locally or downloadable by model_zoo.
             # If config_path points to a file that was previously downloaded by gdown,
             # you'll need to ensure THAT config file is also present.
             # For simplicity and common practice, assume model_zoo config or local config is available.
             try:
                 cfg.merge_from_file(model_zoo.get_config_file(config_path))
                 print(f"Using Detectron2 model zoo config: {config_path}")
             except Exception as model_zoo_error:
                 print(f"Error accessing Detectron2 model zoo config '{config_path}': {model_zoo_error}")
                 print("Please ensure 'detectron2_base_config' in config.json is a valid model_zoo path or a local config file path.")
                 raise # Re-raise the specific model_zoo error

        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names_for_maskrcnn) # Uses loaded class names list
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Detectron2 model '{os.path.basename(weight_path)}' loading on: {cfg.MODEL.DEVICE} with {len(class_names_for_maskrcnn)} classes.")
        predictor = DefaultPredictor(cfg)
        print(f"Detectron2 model '{os.path.basename(weight_path)}' loaded successfully.")
        return predictor
    except FileNotFoundError: # Catch the specific FileNotFoundError raised above
        raise # Re-raise it to be caught by the main loading block
    except Exception as e:
        print(f"Error loading Detectron2 model from {weight_path}: {e}")
        traceback.print_exc()
        raise # Re-raise the exception

# --- Load All Models ---
print("Loading models...")
start_time = time.time()
try:
    # Note: File existence checks are now primarily within load_detectron2_model
    # and explicit checks before ONNX sessions.

    part_predictor = load_detectron2_model(
        config_path=MASKRCNN_CONFIG_FILE, # From config
        weight_path=PART_SEG_MODEL_PATH,
        class_names_for_maskrcnn=car_part_classes, # From config
        conf_thres=PART_SEG_CONF_THRES # From config
    )

    damage_predictor = load_detectron2_model(
        config_path=MASKRCNN_CONFIG_FILE, # From config
        weight_path=DAMAGE_SEG_MODEL_PATH,
        class_names_for_maskrcnn=damage_segmentation_class_names, # From config
        conf_thres=DAMAGE_SEG_CONF_THRES # From config
    )

    # Explicit check for ONNX models
    if not os.path.exists(SEVERITY_CLASS_MODEL_PATH):
         print(f"FATAL ERROR: Severity Classifier model not found at: {SEVERITY_CLASS_MODEL_PATH}")
         raise FileNotFoundError(f"Severity Classifier model not found: {SEVERITY_CLASS_MODEL_PATH}. Ensure the model is placed in the '{MODEL_DIR}' directory.")
    classifier_session = ort.InferenceSession(SEVERITY_CLASS_MODEL_PATH)
    print(f"ONNX Severity Classifier '{os.path.basename(SEVERITY_CLASS_MODEL_PATH)}' loaded.")

    if not os.path.exists(DAMAGE_TYPE_DETECT_MODEL_PATH):
         print(f"FATAL ERROR: Damage Type Detector model not found at: {DAMAGE_TYPE_DETECT_MODEL_PATH}")
         raise FileNotFoundError(f"Damage Type Detector model not found: {DAMAGE_TYPE_DETECT_MODEL_PATH}. Ensure the model is placed in the '{MODEL_DIR}' directory.")
    detector_session = ort.InferenceSession(DAMAGE_TYPE_DETECT_MODEL_PATH)
    print(f"ONNX Damage Type Detector '{os.path.basename(DAMAGE_TYPE_DETECT_MODEL_PATH)}' loaded.")

    print(f"All models loaded successfully in {time.time() - start_time:.2f} seconds.")

# Specific check for FileNotFoundError raised by loading helpers
except FileNotFoundError as fnf_error:
    print(f"FATAL ERROR: Model file not found: {fnf_error}")
    exit(1)
except Exception as load_error:
    print(f"FATAL ERROR during model loading: {load_error}")
    traceback.print_exc()
    exit(1)

# ===========================
# 🛠️ Helper Functions (Now use global constants derived from JSON)
# ===========================

# --- NMS for ONNX Damage Detector ---
def non_max_suppression_for_damage_detector(boxes, scores, iou_threshold):
    if boxes.shape[0] == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-5)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# --- Post-processing for ONNX Damage Detector ---
def postprocess_onnx_damage_detections(
    outputs, original_width, original_height,
    input_width, input_height, conf_threshold,
    iou_threshold, class_names_list
):
    if not outputs or len(outputs) == 0 or outputs[0].ndim != 3:
        print("Warning: Damage detector returned unexpected output format or is empty.")
        return []

    raw_detections = outputs[0][0] # Shape (num_classes + 4, num_detections)

    num_classes = len(class_names_list)
    expected_rows = 4 + num_classes
    if raw_detections.shape[0] != expected_rows:
        print(f"Error: Damage detector unexpected output shape {raw_detections.shape}. Expected {expected_rows} rows based on {num_classes} classes.")
        return []

    detections_transposed = raw_detections.T # Shape (num_detections, num_classes + 4)
    boxes, scores, class_ids = [], [], []
    scale_x = original_width / input_width
    scale_y = original_height / input_height

    for row in detections_transposed:
        box_coords, class_probs = row[:4], row[4:]
        max_score_index = np.argmax(class_probs)
        max_score = class_probs[max_score_index]

        if max_score >= conf_threshold:
            class_id = max_score_index
            cx, cy, w, h = box_coords
            if w <= 0 or h <= 0: continue # Skip invalid boxes

            # Convert center-wh to min-max and scale to original image size
            x_min_inp = cx - w / 2
            y_min_inp = cy - h / 2
            x_max_inp = cx + w / 2
            y_max_inp = cy + h / 2

            orig_x_min = max(0.0, x_min_inp * scale_x)
            orig_y_min = max(0.0, y_min_inp * scale_y)
            orig_x_max = min(original_width - 1.0, x_max_inp * scale_x)
            orig_y_max = min(original_height - 1.0, y_max_inp * scale_y)

            if orig_x_max > orig_x_min and orig_y_max > orig_y_min:
                 boxes.append([orig_x_min, orig_y_min, orig_x_max, orig_y_max])
                 scores.append(max_score)
                 class_ids.append(class_id)


    if not boxes:
        # print("Info: No detections above confidence threshold.") # Too verbose
        return []

    boxes_np = np.array(boxes).astype(np.float32)
    scores_np = np.array(scores).astype(np.float32)
    class_ids_np = np.array(class_ids)

    indices_to_keep = non_max_suppression_for_damage_detector(
        boxes_np, scores_np, iou_threshold
    )

    final_detections = []
    for i in indices_to_keep:
        class_id_val = class_ids_np[i]
        try:
            class_name = class_names_list[class_id_val]
        except IndexError:
            print(f"Warning: Damage detector class ID {class_id_val} out of bounds for {len(class_names_list)} classes.")
            class_name = "Unknown" # Fallback name

        final_detections.append({
            "box": [int(coord) for coord in boxes_np[i]],
            "class_name": class_name,
            "confidence": float(scores_np[i]),
            "class_id": int(class_id_val)
        })
    return final_detections


# --- Run ONNX Damage Type Detector ---
def run_onnx_damage_type_detector(image_numpy_bgr, detector_session):
    if image_numpy_bgr is None or image_numpy_bgr.size == 0:
        print("Warning: Empty image passed to run_onnx_damage_type_detector.")
        return [] # Return empty list for no detections

    original_height, original_width = image_numpy_bgr.shape[:2]
    try:
        # Convert BGR to RGB for PIL, resize, normalize
        image_rgb = cv2.cvtColor(image_numpy_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        # Use global constant DAMAGE_DETECTOR_INPUT_SIZE
        img_resized_pil = img_pil.resize(DAMAGE_DETECTOR_INPUT_SIZE, Image.LANCZOS)
        img_array = np.array(img_resized_pil, dtype=np.float32) / 255.0
        img_transposed = np.transpose(img_array, (2, 0, 1)) # HWC to CHW
        input_tensor = np.expand_dims(img_transposed, axis=0) # Add batch dim

        input_name = detector_session.get_inputs()[0].name
        output_names = [out.name for out in detector_session.get_outputs()]

        outputs_onnx = detector_session.run(output_names, {input_name: input_tensor})

        # Use global constants for postprocessing parameters and class names
        detections = postprocess_onnx_damage_detections(
            outputs_onnx,
            original_width, original_height,
            DAMAGE_DETECTOR_INPUT_SIZE[0], DAMAGE_DETECTOR_INPUT_SIZE[1],
            DAMAGE_DETECTOR_CONF_THRESHOLD, # Global constant
            DAMAGE_DETECTOR_IOU_THRESHOLD,  # Global constant
            damage_type_names            # Global constant
        )

        detected_class_indices = [det['class_id'] for det in detections] if detections else []
        return detected_class_indices # Return list of class indices

    except Exception as e:
        print(f"Error during ONNX Damage Type Detector inference: {e}")
        traceback.print_exc()
        return [] # Return empty list on error


# --- IoU Calculation ---
def compute_iou(mask1, mask2):
    # Ensure masks are boolean
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    return intersection / union if union != 0 else 0.0 # Return float


# --- Run ONNX Severity Classifier ---
def run_yolo_classifier(image, classifier_session):
    if image is None or image.size == 0:
        print("Warning: Empty image passed to run_yolo_classifier.")
        return 0 # Default to first class (e.g., Low)

    try:
        # Use global constant SEVERITY_CLASSIFIER_INPUT_SIZE
        img_resized = cv2.resize(image, SEVERITY_CLASSIFIER_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Classifier expects RGB
        img_normalized = img_rgb / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1).astype(np.float32)[None, :] # CHW, Add batch dim

        input_name = classifier_session.get_inputs()[0].name
        outputs = classifier_session.run(None, {input_name: img_transposed})

        # Assuming the output is a single tensor of shape (1, num_classes)
        if outputs and len(outputs) > 0 and outputs[0].ndim == 2 and outputs[0].shape[0] == 1:
            return int(np.argmax(outputs[0][0])) # Get index of max probability
        else:
             print(f"Warning: Unexpected output shape from classifier: {outputs[0].shape if outputs else 'None'}")
             return 0 # Default to first class

    except Exception as e:
        print(f"Error during ONNX Severity Classifier inference: {e}")
        traceback.print_exc()
        return 0 # Default to first class on error


# --- Run Detectron2 Mask R-CNN ---
def run_mask_rcnn(image, predictor):
    if image is None or image.size == 0:
        print("Warning: Empty image passed to run_mask_rcnn.")
        return np.array([]), np.array([])

    try:
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        if not instances.has("pred_masks") or not instances.has("pred_classes"):
             print("Warning: Detectron2 output missing 'pred_masks' or 'pred_classes'.")
             return np.array([]), np.array([])
        masks = instances.pred_masks.numpy() # Shape (N, H, W) boolean
        classes = instances.pred_classes.numpy() # Shape (N,) int
        # Convert boolean masks to uint8 (0 or 1) if needed by downstream steps,
        # but compute_iou expects boolean, so keep as boolean here.
        return masks, classes
    except Exception as e:
        print(f"Error during Detectron2 inference: {e}")
        traceback.print_exc()
        return np.array([]), np.array([])


# --- Estimate Repair Cost ---
def estimate_repair_cost(image, part_masks, damage_masks, part_labels,
                         classifier_session, detector_session):
    results = []
    total_cost = 0.0 # Use float for total cost

    if image is None or image.size == 0:
        print("Warning: Empty image provided for cost estimation.")
        return {"total_cost": 0.0, "damages": []}
    if part_masks.size == 0 or damage_masks.size == 0:
        print("Info: No valid part or damage masks provided for cost estimation.")
        return {"total_cost": 0.0, "damages": []}
    if len(part_masks) != len(part_labels):
         print(f"Warning: Mismatch between part masks ({len(part_masks)}) and labels ({len(part_labels)}). Creating generic labels.")
         # Create generic labels if mismatch
         part_labels = [f"Part_{j}" for j in range(len(part_masks))]


    print(f"Estimating cost for {len(damage_masks)} damage masks / {len(part_masks)} part masks...")
    processed_overlaps = 0

    for i, dmg_mask in enumerate(damage_masks):
        for j, part_mask in enumerate(part_masks):
            try:
                # Ensure masks are boolean for accurate IoU calculation
                iou = compute_iou(dmg_mask.astype(bool), part_mask.astype(bool))

                if iou > COST_ESTIMATION_IOU_THRESHOLD:
                    processed_overlaps += 1
                    part_name = part_labels[j] # part_labels length should now match part_masks

                    # --- Create Crop ---
                    # Ensure boolean masks are used for intersection
                    overlap_mask_bool = np.logical_and(dmg_mask.astype(bool), part_mask.astype(bool))
                    rows, cols = np.where(overlap_mask_bool)

                    if rows.size == 0 or cols.size == 0: continue # Should not happen if IoU > 0, but safety check
                    y_min, y_max = np.min(rows), np.max(rows)
                    x_min, x_max = np.min(cols), np.max(cols)

                    padding = 5 # Add padding for better classification/detection
                    img_h, img_w = image.shape[:2]
                    crop_y_min = max(0, y_min - padding)
                    crop_y_max = min(img_h, y_max + 1 + padding) # +1 because slicing is exclusive
                    crop_x_min = max(0, x_min - padding)
                    crop_x_max = min(img_w, x_max + 1 + padding)

                    if crop_y_max <= crop_y_min or crop_x_max <= crop_x_min: continue # Invalid crop size

                    cropped_image_bgr = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy() # Use .copy() to avoid read-only issues
                    if cropped_image_bgr is None or cropped_image_bgr.size == 0: continue

                    # --- Run Severity Classification ---
                    severity_idx = run_yolo_classifier(cropped_image_bgr, classifier_session)
                    # Clamp severity index to valid range
                    safe_severity_idx = max(0, min(severity_idx, len(severity_names) - 1))
                    severity = severity_names[safe_severity_idx]

                    # --- Run Damage Type Detection ---
                    detected_damage_type_idxs = run_onnx_damage_type_detector(
                        cropped_image_bgr, detector_session
                    )

                    # --- Process Each Detected Damage Type ---
                    if not detected_damage_type_idxs:
                        # If no damage type detected, default to 'Unknown' or a specific default damage type
                        processed_types_for_this_overlap = {"Unknown"} # Use a set to avoid duplicates
                        detected_damage_type_names = ["Unknown"]
                    else:
                        processed_types_for_this_overlap = set()
                        detected_damage_type_names = []
                        for dmg_cls_idx in detected_damage_type_idxs:
                             dmg_type = "Unknown"
                             if 0 <= dmg_cls_idx < len(damage_type_names):
                                 dmg_type = damage_type_names[dmg_cls_idx]
                             else:
                                 print(f"Warning: Damage detector idx {dmg_cls_idx} out of bounds for {len(damage_type_names)} types. Using 'Unknown'.")

                             if dmg_type not in processed_types_for_this_overlap:
                                processed_types_for_this_overlap.add(dmg_type)
                                detected_damage_type_names.append(dmg_type)


                    for dmg_type in detected_damage_type_names:
                        # --- Calculate Cost using Base Cost * Multiplier ---
                        base_cost = part_base_costs.get(part_name, part_base_costs.get('default', 0.0)) # Default base cost to 0.0
                        multiplier_list = damage_multipliers.get(dmg_type, damage_multipliers.get('Unknown', [1.0] * len(severity_names))) # Default multiplier list
                        multiplier = 1.0 # Default multiplier if list is invalid

                        if isinstance(multiplier_list, list) and len(multiplier_list) == len(severity_names):
                            # Use the same safe severity index as before
                            multiplier = multiplier_list[safe_severity_idx]
                        else:
                             print(f"Warning: Invalid multiplier list format for type '{dmg_type}': {multiplier_list}. Expected list of length {len(severity_names)}. Using 1.0.")

                        cost = float(base_cost) * float(multiplier) # Ensure float calculation
                        total_cost += cost

                        results.append({
                            "part": part_name,
                            "damage_type": dmg_type,
                            "severity": severity,
                            "base_cost": round(float(base_cost), 2),
                            "multiplier": round(float(multiplier), 2),
                            "calculated_cost": round(float(cost), 2) # Ensure float and round
                        })

            except Exception as cost_calc_error:
                print(f"Error processing overlap between damage mask {i} and part mask {j} (Part: {part_labels[j] if j < len(part_labels) else 'N/A'}): {cost_calc_error}")
                traceback.print_exc()
                continue


    print(f"Finished cost estimation. Processed {processed_overlaps} valid overlaps. Final Total Cost: {total_cost:.2f}")
    return {
        "total_cost": round(total_cost, 2),
        "damages": results
    }

# ===========================
#  Flask Application Setup
# ===========================
app = Flask(__name__)
CORS(app) # Apply CORS to your app

@app.route('/')
def home():
    # Add simple check for models existing on startup
    model_files = [PART_SEG_MODEL_PATH, DAMAGE_SEG_MODEL_PATH, SEVERITY_CLASS_MODEL_PATH, DAMAGE_TYPE_DETECT_MODEL_PATH]
    missing_models = [f for f in model_files if not os.path.exists(f)]
    status_message = "Car Damage Estimation API is running."
    overall_status = "OK"
    if missing_models:
        status_message += f" WARNING: The following model files are missing: {', '.join(missing_models)}. Prediction requests may fail."
        overall_status = "WARNING: Models Missing"

    return jsonify({"message": status_message, "status": overall_status}), 200 if overall_status == "OK" else 500


@app.route('/predict', methods=['POST'])
def predict_damage_cost():
    print("\nReceived request on /predict")
    start_req_time = time.time()

    if 'image_file' not in request.files:
        print("Error: No 'image_file' part in the request.")
        return jsonify({"error": "Missing 'image_file' in request form data"}), 400

    file = request.files['image_file']
    if file.filename == '':
        print("Error: No selected file.")
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        # Use io.BytesIO for safer reading by PIL if needed, but cv2.imdecode is fine here
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_bgr is None:
             # Check if the buffer was actually empty after reading
             if len(image_bytes) == 0:
                 raise ValueError("Uploaded file is empty.")
             else:
                 raise ValueError("Could not decode image. Invalid format?")
        print(f"Image decoded successfully: shape={image_bgr.shape}, dtype={image_bgr.dtype}")
    except Exception as e:
        print(f"Error reading/decoding image: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to read or decode image: {e}"}), 400

    try:
        print("Running Part Segmentation...")
        t1 = time.time()
        part_masks, part_class_idxs = run_mask_rcnn(image_bgr, part_predictor)
        print(f"Part Segmentation found {len(part_masks)} masks in {time.time()-t1:.2f}s.")

        print("Running Damage Segmentation...")
        t2 = time.time()
        all_damage_masks, all_damage_classes = run_mask_rcnn(image_bgr, damage_predictor)

        # Filter masks using the derived DAMAGE_CLASS_INDEX_IN_SEGMENTER
        damage_masks = np.array([]) # Initialize as empty numpy array
        if all_damage_masks.size > 0 and all_damage_classes.size > 0:
            # Ensure shapes match for boolean indexing
            if len(all_damage_masks) == len(all_damage_classes):
                damage_masks = all_damage_masks[all_damage_classes == DAMAGE_CLASS_INDEX_IN_SEGMENTER]
                print(f"Damage Segmentation found {len(all_damage_masks)} total, filtered to {len(damage_masks)} '{DAMAGE_CLASS_LABEL_IN_SEGMENTER}' masks in {time.time()-t2:.2f}s.")
            else:
                 print(f"Warning: Mismatch between total damage masks ({len(all_damage_masks)}) and classes ({len(all_damage_classes)}). Cannot filter damage masks effectively.")
        else:
             print(f"Damage Segmentation found no masks or classes in {time.time()-t2:.2f}s.")


        # Map part class indices to labels using global car_part_classes
        part_labels = []
        if part_class_idxs.size > 0:
            for idx in part_class_idxs:
                if 0 <= idx < len(car_part_classes):
                    part_labels.append(car_part_classes[idx])
                else:
                    part_labels.append(f"Unknown_Part_Idx_{idx}")
                    print(f"Warning: Part class index {idx} out of bounds for car_part_classes list ({len(car_part_classes)} items).")
        # Note: estimate_repair_cost now handles potential mismatch between part_masks and part_labels lengths

        if not part_masks.any() or not damage_masks.any():
            print("No part or damage masks found matching criteria. Returning zero cost.")
            result = {"total_cost": 0.0, "damages": []}
        else:
            print("Estimating repair cost...")
            t3 = time.time()
            result = estimate_repair_cost(
                image_bgr, part_masks, damage_masks, part_labels, # Pass part_labels
                classifier_session, detector_session
            )
            print(f"Cost estimation completed in {time.time()-t3:.2f}s.")

        total_processing_time = time.time() - start_req_time
        print(f"Request processed successfully in {total_processing_time:.2f} seconds.")
        return jsonify(result), 200

    except Exception as e:
        print(f"Error during prediction pipeline: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during processing.", "details": str(e)}), 500

# ===========================
# Main Driver
# ===========================
if __name__ == '__main__':
    print("Starting Flask development server...")
    # Note: In production, use a WSGI server like Gunicorn or uWSGI
    # Add a quick check on startup to remind the user if models are missing
    model_files = [PART_SEG_MODEL_PATH, DAMAGE_SEG_MODEL_PATH, SEVERITY_CLASS_MODEL_PATH, DAMAGE_TYPE_DETECT_MODEL_PATH]
    missing_models = [f for f in model_files if not os.path.exists(f)]
    if missing_models:
        print("\n" + "="*50)
        print("!!! WARNING: Missing Model Files !!!")
        print("The application is configured to *load* models from specific paths,")
        print("but the following expected model files were NOT FOUND:")
        for mf in missing_models:
            print(f"- {mf}")
        print(f"Please ensure the models are placed in the '{MODEL_DIR}' directory.")
        print("Prediction requests will likely fail until these files are present.")
        print("="*50 + "\n")
    else:
         print("\nAll required model files appear to be present.")


    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)