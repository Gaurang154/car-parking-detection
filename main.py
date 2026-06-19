"""CLI desktop viewer — classical engine only (no model download)."""

import cv2

from detector import ParkingDetector
from storage import OccupancyHistory, PositionStore

position_store = PositionStore()
history = OccupancyHistory()
detector = ParkingDetector(position_store, history, enable_yolo=False)

cap = cv2.VideoCapture(detector.video_source)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {detector.video_source}")

print("PARKX Desktop Detector")
print("Press 'q' to quit | Press 's' to save screenshot")

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, frame = cap.read()
    if not success:
        break

    annotated, stats = detector.process_single_frame(frame)
    cv2.imshow("PARKX — Parking Detector", annotated)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        print("Exiting...")
        break
    if key == ord("s"):
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        filename = f"screenshot_frame{frame_num}.png"
        cv2.imwrite(filename, annotated)
        print(f"Screenshot saved: {filename}")

cap.release()
cv2.destroyAllWindows()
