import cv2
import numpy as np
from config_new import VIDEO_PATH

# Global list to store the ROI points.
roi_points = []

def click_event(event, x, y, flags, param):
    global roi_points
    image = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the point coordinates to the list.
        roi_points.append((x, y))
        # Draw a small circle at the clicked point.
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select ROI", image)
        # If four points are selected, draw the polygon.
        if len(roi_points) == 4:
            pts = np.array(roi_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Select ROI", image)

# Open a video file or camera stream.
cap = cv2.VideoCapture(VIDEO_PATH)  # Replace with your video path.
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read frame from video.")
    cap.release()
    exit(1)

# Create a clone for drawing.
clone = frame.copy()
cv2.imshow("Select ROI", clone)
cv2.setMouseCallback("Select ROI", click_event, clone)

print("Please click four points to define the ROI.")

# Wait until 4 points are selected or a key is pressed.
while len(roi_points) < 1:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyWindow("Select ROI")

print("Selected ROI points:", roi_points)
