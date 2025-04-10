# tracker.py
# tracker.py
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class KalmanTracker:
    def __init__(self, initial_position, track_id):
        self.track_id = track_id
        self.kf = cv2.KalmanFilter(4, 2)
        # State vector: [x, y, dx, dy]; Measurement: [x, y]
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        # Process noise covariance: lower value to trust the motion model more.
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        # Increased measurement noise covariance to reduce the impact of noisy detections.
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e0

        # Initialize state with the initial detected position and zero velocity.
        self.kf.statePre = np.array([[initial_position[0]],
                                     [initial_position[1]],
                                     [0],
                                     [0]], np.float32)
        self.kf.statePost = np.array([[initial_position[0]],
                                      [initial_position[1]],
                                      [0],
                                      [0]], np.float32)
        self.prediction = initial_position
        self.time_since_update = 0

    def predict(self):
        pred = self.kf.predict()
        self.prediction = (int(pred[0]), int(pred[1]))
        self.time_since_update += 1
        return self.prediction

    def update(self, measurement):
        # measurement: (x, y)
        measurement_np = np.array([[np.float32(measurement[0])],
                                   [np.float32(measurement[1])]])
        self.kf.correct(measurement_np)
        self.prediction = (int(self.kf.statePost[0]), int(self.kf.statePost[1]))
        self.time_since_update = 0
        return self.prediction

class KalmanTrackerManager:
    def __init__(self, max_distance=70, max_age=20):
        self.trackers = []  # List of active KalmanTracker instances.
        self.next_id = 0
        self.max_distance = max_distance  # Maximum allowed distance for association.
        self.max_age = max_age            # Maximum frames allowed without an update.

    def update(self, detections):
        """
        Update trackers with new detections.
        
        detections: list of bounding boxes in [x1, y1, x2, y2] format.
        """
        # Compute centroids for new detections.
        detection_centroids = []
        for box in detections:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            detection_centroids.append((cx, cy))
        
        # Predict new positions for existing trackers.
        predictions = [tracker.predict() for tracker in self.trackers]
        
        if len(predictions) > 0 and len(detection_centroids) > 0:
            # Build cost matrix (Euclidean distance).
            cost_matrix = np.zeros((len(detection_centroids), len(predictions)), dtype=np.float32)
            for i, det in enumerate(detection_centroids):
                for j, pred in enumerate(predictions):
                    cost_matrix[i, j] = np.linalg.norm(np.array(det) - np.array(pred))
            
            # Solve assignment with the Hungarian algorithm.
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_detection_indices = set()
            for i, j in zip(row_ind, col_ind):
                # Accept association only if cost is below threshold.
                if cost_matrix[i, j] < self.max_distance:
                    self.trackers[j].update(detection_centroids[i])
                    assigned_detection_indices.add(i)
            
            # Create new trackers for detections not associated with any tracker.
            for i, det in enumerate(detection_centroids):
                if i not in assigned_detection_indices:
                    new_tracker = KalmanTracker(det, self.next_id)
                    self.trackers.append(new_tracker)
                    self.next_id += 1
        else:
            # No existing trackers; create a new one for each detection.
            for det in detection_centroids:
                new_tracker = KalmanTracker(det, self.next_id)
                self.trackers.append(new_tracker)
                self.next_id += 1

        # Remove trackers that haven't been updated recently.
        self.trackers = [tracker for tracker in self.trackers if tracker.time_since_update <= self.max_age]
        
        # Return dictionary mapping track_id to current predicted centroid.
        tracks = {tracker.track_id: tracker.prediction for tracker in self.trackers}
        return tracks
