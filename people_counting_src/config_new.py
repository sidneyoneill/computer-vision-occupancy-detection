import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = os.path.join(BASE_DIR, "input_vid_counting", "IMG_3388.MOV")
MODEL_PATH = os.path.join(BASE_DIR, "seat_detection_env", "yolov8m.pt")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_counting", "live_counting_file.MOV")
DETECTION_INTERVAL = 3
ANNOTATIONS_PATH = os.path.join(BASE_DIR, "annotations", "counting_line.json")

# Define a polygon ROI as a list of four points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
# The ROI is the area inside this polygon.
ROI = [(444, 203), (452, 718), (1553, 915), (1572, 129)]
