"""Legacy OpenCV space picker — prefer the web picker at /picker."""

import cv2

from config import IMG_PATH, SPACE_HEIGHT, SPACE_WIDTH
from storage import PositionStore

store = PositionStore()


def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        store.add(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        store.remove_at(x, y, SPACE_WIDTH, SPACE_HEIGHT)


print("PARKX Space Picker (CLI)")
print("Left-click  → add space")
print("Right-click → remove space")
print("Press 'q'   → quit")
print("Tip: use http://localhost:8080/picker for the web picker")

while True:
    img = cv2.imread(str(IMG_PATH))
    for x, y in store.list():
        cv2.rectangle(img, (x, y), (x + SPACE_WIDTH, y + SPACE_HEIGHT), (255, 0, 255), 2)

    cv2.putText(
        img,
        f"Spaces: {store.count()}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.imshow("PARKX Space Picker", img)
    cv2.setMouseCallback("PARKX Space Picker", mouse_click)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(f"Saved {store.count()} parking spaces.")
        break

cv2.destroyAllWindows()
