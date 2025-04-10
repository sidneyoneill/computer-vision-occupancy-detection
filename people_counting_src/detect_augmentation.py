import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# --- Import configuration ---
from config_new import VIDEO_PATH, MODEL_PATH, OUTPUT_PATH, DETECTION_INTERVAL

def enhance_image(img):
    """
    Enhance the image by boosting contrast, brightness, and sharpness.
    """
    # Convert to LAB color space.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE on the L channel.
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Gamma correction.
    gamma = 3
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    # Sharpening.
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

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

# --- Load YOLOv8 Model ---
print("MPS Available:", torch.backends.mps.is_available())
model = YOLO(MODEL_PATH)
if torch.backends.mps.is_available():
    model.to("mps")

# --- Variables ---
frame_count = 0
prev_time = time.time()
last_results = None
total_detections = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    # Enhance the full frame
    enhanced_frame = enhance_image(frame)

    # Run detection on enhanced frame every DETECTION_INTERVAL frames
    if frame_count % DETECTION_INTERVAL == 0:
        results = model(enhanced_frame, conf=0.25, iou=0.5)
        last_results = results
    else:
        results = last_results

    annotator = Annotator(display_frame, line_width=2)

    # Process detection results
    if results is not None:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, conf, class_id in zip(boxes, confs, class_ids):
                if int(class_id) != 0:
                    continue  # Only detect people (class 0)
                x1, y1, x2, y2 = map(int, box)
                label = f"Conf: {conf:.2f}"
                annotator.box_label([x1, y1, x2, y2], label, color=colors(0, True))
                total_detections += 1

    # Calculate and display FPS
    current_time = time.time()
    computed_fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(display_frame, f"FPS: {computed_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write and show
    out.write(display_frame)
    cv2.imshow("YOLOv8 Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total frames processed: {frame_count}")
print(f"Total detections over the entire video: {total_detections}")
