import cv2
import time
import torch
import json
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from scipy.optimize import linear_sum_assignment

# --- Import configuration ---
from config_new import VIDEO_PATH, MODEL_PATH, OUTPUT_PATH, DETECTION_INTERVAL, ANNOTATIONS_PATH, ROI

def load_annotations(json_path):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load annotations: {e}")
        return None

def enhance_image(img):
    """
    Enhance the image by boosting contrast, brightness, and sharpness.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    gamma = 5  # Adjust gamma as needed.
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# ----------------------------
# Kalman Filter-Based Tracker with Global Assignment
# ----------------------------
class KalmanTracker:
    def __init__(self, initial_point, bbox, conf, track_id):
        self.track_id = track_id
        self.kf = cv2.KalmanFilter(4, 2)
        # State: [x, y, dx, dy]; Measurement: [x, y]
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePre = np.array([[initial_point[0]], [initial_point[1]], [0], [0]], np.float32)
        self.bbox = bbox  # [x1, y1, x2, y2] in full-frame coordinates.
        self.conf = conf
        self.time_since_update = 0
        self.prediction = initial_point
        self.last_measurement = initial_point  # used for association

    def predict(self):
        pred = self.kf.predict()
        self.prediction = (int(pred[0]), int(pred[1]))
        self.time_since_update += 1
        return self.prediction

    def update(self, measured_point, bbox, conf):
        measurement = np.array([[np.float32(measured_point[0])],
                                [np.float32(measured_point[1])]])
        self.kf.correct(measurement)
        self.bbox = bbox
        self.conf = conf
        self.time_since_update = 0
        self.prediction = (int(self.kf.statePost[0]), int(self.kf.statePost[1]))
        self.last_measurement = measured_point
        return self.prediction

class KalmanTrackerManager:
    def __init__(self, max_distance=350, max_age=15):
        self.trackers = []  # list of KalmanTracker instances.
        self.next_id = 0
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, detections):
        """
        detections: list of tuples (cx, cy, bbox, conf)
        Returns a dictionary mapping track_id to (predicted cx, cy, bbox, conf, last_measurement)
        Uses the Hungarian algorithm for global assignment.
        Association uses each tracker's last_measurement.
        """
        n = len(self.trackers)
        m = len(detections)
        for tracker in self.trackers:
            tracker.predict()
        if n > 0 and m > 0:
            cost_matrix = np.zeros((n, m), dtype=np.float32)
            for i, tracker in enumerate(self.trackers):
                for j, det in enumerate(detections):
                    d = np.linalg.norm(np.array(tracker.last_measurement) - np.array([det[0], det[1]]))
                    cost_matrix[i, j] = d
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned = [False] * m
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.max_distance:
                    self.trackers[i].update((detections[j][0], detections[j][1]),
                                             detections[j][2],
                                             detections[j][3])
                    assigned[j] = True
            for j, det in enumerate(detections):
                if not assigned[j]:
                    new_tracker = KalmanTracker((det[0], det[1]), det[2], det[3], self.next_id)
                    self.trackers.append(new_tracker)
                    self.next_id += 1
        else:
            for det in detections:
                new_tracker = KalmanTracker((det[0], det[1]), det[2], det[3], self.next_id)
                self.trackers.append(new_tracker)
                self.next_id += 1

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        tracks = {}
        for t in self.trackers:
            tracks[t.track_id] = (t.prediction[0], t.prediction[1], t.bbox, t.conf, t.last_measurement)
        return tracks

# ----------------------------
# End of Tracker Classes
# ----------------------------

# --- Counting Line Definition ---
# Fixed counting line coordinates (as provided):

line_point1 = (1058, 62)
line_point2 = (945, 788)

def side_of_line(point, p1, p2):
    # f(x,y) = (x - p1x)*(p2y - p1y) - (y - p1y)*(p2x - p1x)
    return (point[0] - p1[0]) * (p2[1] - p1[1]) - (point[1] - p1[1]) * (p2[0] - p1[0])

# Initialize counting variables.
counter_in = 0
counter_out = 0
prev_side = {}  # Previous side for each track.

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

# --- (Optional) Load Counting Line from Annotations for Visualization ---
annotations = load_annotations(ANNOTATIONS_PATH)
if annotations and "counting_line" in annotations and "points" in annotations["counting_line"]:
    pts = annotations["counting_line"]["points"]
    # Here we keep our fixed counting line; you can optionally visualize the annotations.
    vis_line_pt1 = (int(pts[0][0]), int(pts[0][1]))
    vis_line_pt2 = (int(pts[1][0]), int(pts[1][1]))
else:
    vis_line_pt1 = line_point1
    vis_line_pt2 = line_point2

# --- Variables for FPS Measurement ---
frame_count = 0
prev_time = time.time()
last_results = None

# Initialize our Kalman tracker manager.
tracker_manager = KalmanTrackerManager(max_distance=350, max_age=15)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    display_frame = frame.copy()

    # Process ROI.
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

    # Enhance the ROI.
    enhanced_roi = enhance_image(roi_crop_masked)

    # Run detection on the enhanced ROI every DETECTION_INTERVAL frames.
    if frame_count % DETECTION_INTERVAL == 0:
        results = model(enhanced_roi, conf=0.3, iou=0.7)
        last_results = results
    else:
        results = last_results

    detections_list = []
    if results is not None:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # In ROI crop coordinates.
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, conf, class_id in zip(boxes, confs, class_ids):
                if int(class_id) != 0:
                    continue
                x1 = int(box[0] + roi_x)
                y1 = int(box[1] + roi_y)
                x2 = int(box[2] + roi_x)
                y2 = int(box[3] + roi_y)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections_list.append((cx, cy, [x1, y1, x2, y2], conf))

    # Update tracker with current detections.
    tracks = tracker_manager.update(detections_list)

    annotator = Annotator(display_frame, line_width=2)
    cv2.polylines(display_frame, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=2)
    
    # Draw the fixed counting line using our defined points.
    cv2.line(display_frame, line_point1, line_point2, (0, 0, 255), 4)

    # Process each tracked object.
    for track_id, data in tracks.items():
        pred_cx, pred_cy, bbox, conf, meas = data
        label = f"ID: {track_id} Conf: {conf:.2f}"
        annotator.box_label(bbox, label, color=colors(track_id, True))
        cv2.circle(display_frame, (pred_cx, pred_cy), 4, (255, 0, 0), -1)  # predicted centroid (blue)
        cv2.circle(display_frame, (int(meas[0]), int(meas[1])), 4, (0, 0, 255), -1)  # measured centroid (red)
        
        # Determine side for current measured centroid.
        current_side = np.sign(side_of_line((meas[0], meas[1]), line_point1, line_point2))
        if track_id in prev_side:
            previous_side = prev_side[track_id]
            if current_side != previous_side and current_side != 0:
                if previous_side > 0 and current_side < 0:
                    counter_in += 1
                elif previous_side < 0 and current_side > 0:
                    counter_out += 1
        prev_side[track_id] = current_side

    current_time = time.time()
    computed_fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(display_frame, f"FPS: {computed_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(display_frame, f"Out: {counter_in}  In: {counter_out}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    out.write(display_frame)
    cv2.imshow("Tracking with Counting Line", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total frames processed: {frame_count}")


