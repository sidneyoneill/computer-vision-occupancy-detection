import cv2
import time
import torch
import json
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# --- Import configuration ---
from config_new import VIDEO_PATH, MODEL_PATH, OUTPUT_PATH, DETECTION_INTERVAL, ANNOTATIONS_PATH, ROI

def load_annotations(json_path):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load annotations: {e}")
        return None

# --- Initialize Video Capture and Writer ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video file!")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# --- Load YOLOv8 Model for Detection ---
print("MPS Available:", torch.backends.mps.is_available())
model = YOLO(MODEL_PATH)
if torch.backends.mps.is_available():
    model.to("mps")

# --- Optionally, load counting line from annotations (for visualization) ---
annotations = load_annotations(ANNOTATIONS_PATH)
if annotations and "counting_line" in annotations and "points" in annotations["counting_line"]:
    pts = annotations["counting_line"]["points"]
    line_pt1 = (int(pts[0][0]), int(pts[0][1]))
    line_pt2 = (int(pts[1][0]), int(pts[1][1]))
else:
    # Default to a horizontal line at half the frame height.
    line_pt1 = (0, height // 2)
    line_pt2 = (width, height // 2)

# --- Variables for FPS Measurement and Counting ---
frame_count = 0
prev_time = time.time()
last_results = None

total_detections = 0  # Global counter for total detections in the video.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Create a copy of the frame for display.
    display_frame = frame.copy()

    # If ROI is defined as a polygon, process only that region.
    if ROI and len(ROI) == 4:
        roi_pts = np.array(ROI, dtype=np.int32)
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(roi_pts)
        roi_crop = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        shifted_pts = roi_pts - [roi_x, roi_y]
        cv2.fillPoly(mask, [shifted_pts], 255)
        roi_crop_masked = cv2.bitwise_and(roi_crop, roi_crop, mask=mask)
    else:
        roi_crop_masked = frame.copy()
        roi_x, roi_y = 0, 0

    # Run detection on the raw masked ROI every DETECTION_INTERVAL frames.
    if frame_count % DETECTION_INTERVAL == 0:
        results = model(roi_crop_masked, conf=0.6, iou=0.5)
        last_results = results
    else:
        results = last_results

    # Create an annotator for drawing on the full frame.
    annotator = Annotator(display_frame, line_width=2)

    # Draw the polygon ROI on the full frame.
    cv2.polylines(display_frame, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=2)
    # (Optional) Draw the counting line.
    cv2.line(display_frame, line_pt1, line_pt2, (0, 0, 255), 4)

    # Process detection results.
    if results is not None:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Boxes are in ROI crop coordinates.
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, conf, class_id in zip(boxes, confs, class_ids):
                if int(class_id) != 0:
                    continue  # Process only persons (assumed class 0)
                # Adjust box coordinates from ROI crop to full frame.
                x1 = int(box[0] + roi_x)
                y1 = int(box[1] + roi_y)
                x2 = int(box[2] + roi_x)
                y2 = int(box[3] + roi_y)
                label = f"Conf: {conf:.2f}"
                annotator.box_label([x1, y1, x2, y2], label, color=colors(0, True))
                total_detections += 1

    # Calculate and display FPS.
    current_time = time.time()
    computed_fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(display_frame, f"FPS: {computed_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Write display frame to output video and show it.
    out.write(display_frame)
    cv2.imshow("YOLOv8 Detection with Polygon ROI (No Enhancement)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total frames processed: {frame_count}")
print(f"Total detections over the entire video: {total_detections}")
