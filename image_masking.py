import cv2
import numpy as np
import json
import os

IMAGE_PATH = "frame_img2.png"
SAVE_FILE = "hsv_thresholds.json"

# ---------------- Load image ----------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found or failed to load")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ---------------- Load saved thresholds (if exist) ----------------
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "r") as f:
        data = json.load(f)

    mask1_lower = np.array(data["mask1"]["lower"])
    mask1_upper = np.array(data["mask1"]["upper"])
    mask2_lower = np.array(data["mask2"]["lower"])
    mask2_upper = np.array(data["mask2"]["upper"])
else:
    # defaults = full range
    mask1_lower = np.array([0, 0, 0])
    mask1_upper = np.array([179, 255, 255])
    mask2_lower = np.array([0, 0, 0])
    mask2_upper = np.array([179, 255, 255])

# ---------------- Trackbars ----------------
cv2.namedWindow("Trackbars 1")
cv2.namedWindow("Trackbars 2")

def nothing(x):
    pass

# Mask 1 sliders
for name, maxv in [
    ("LH1", 179), ("LS1", 255), ("LV1", 255),
    ("UH1", 179), ("US1", 255), ("UV1", 255)
]:
    cv2.createTrackbar(name, "Trackbars 1", 0, maxv, nothing)

# Mask 2 sliders
for name, maxv in [
    ("LH2", 179), ("LS2", 255), ("LV2", 255),
    ("UH2", 179), ("US2", 255), ("UV2", 255)
]:
    cv2.createTrackbar(name, "Trackbars 2", 0, maxv, nothing)

# ---------------- Set initial values from JSON ----------------
cv2.setTrackbarPos("LH1", "Trackbars 1", mask1_lower[0])
cv2.setTrackbarPos("LS1", "Trackbars 1", mask1_lower[1])
cv2.setTrackbarPos("LV1", "Trackbars 1", mask1_lower[2])
cv2.setTrackbarPos("UH1", "Trackbars 1", mask1_upper[0])
cv2.setTrackbarPos("US1", "Trackbars 1", mask1_upper[1])
cv2.setTrackbarPos("UV1", "Trackbars 1", mask1_upper[2])

cv2.setTrackbarPos("LH2", "Trackbars 2", mask2_lower[0])
cv2.setTrackbarPos("LS2", "Trackbars 2", mask2_lower[1])
cv2.setTrackbarPos("LV2", "Trackbars 2", mask2_lower[2])
cv2.setTrackbarPos("UH2", "Trackbars 2", mask2_upper[0])
cv2.setTrackbarPos("US2", "Trackbars 2", mask2_upper[1])
cv2.setTrackbarPos("UV2", "Trackbars 2", mask2_upper[2])

# ---------------- Main loop ----------------
while True:

    # ---- read mask 1 ----
    lower1 = np.array([
        cv2.getTrackbarPos("LH1", "Trackbars 1"),
        cv2.getTrackbarPos("LS1", "Trackbars 1"),
        cv2.getTrackbarPos("LV1", "Trackbars 1")
    ])

    upper1 = np.array([
        cv2.getTrackbarPos("UH1", "Trackbars 1"),
        cv2.getTrackbarPos("US1", "Trackbars 1"),
        cv2.getTrackbarPos("UV1", "Trackbars 1")
    ])

    mask1 = cv2.inRange(hsv, lower1, upper1)

    # ---- read mask 2 ----
    lower2 = np.array([
        cv2.getTrackbarPos("LH2", "Trackbars 2"),
        cv2.getTrackbarPos("LS2", "Trackbars 2"),
        cv2.getTrackbarPos("LV2", "Trackbars 2")
    ])

    upper2 = np.array([
        cv2.getTrackbarPos("UH2", "Trackbars 2"),
        cv2.getTrackbarPos("US2", "Trackbars 2"),
        cv2.getTrackbarPos("UV2", "Trackbars 2")
    ])

    mask2 = cv2.inRange(hsv, lower2, upper2)

    # ---- FINAL MASK (AND logic: must pass both) ----
    final_mask = cv2.bitwise_and(mask1, mask2)

    result = cv2.bitwise_and(img, img, mask=final_mask)

    # ---------------- display ----------------
    cv2.imshow("Original", img)
    cv2.imshow("Mask 1", mask1)
    cv2.imshow("Mask 2", mask2)
    cv2.imshow("Final Mask", final_mask)
    cv2.imshow("Result", result)

    # ---------------- controls ----------------
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    if key == ord('s'):
        data = {
            "mask1": {
                "lower": lower1.tolist(),
                "upper": upper1.tolist()
            },
            "mask2": {
                "lower": lower2.tolist(),
                "upper": upper2.tolist()
            }
        }

        with open(SAVE_FILE, "w") as f:
            json.dump(data, f, indent=4)

        print("Saved to", SAVE_FILE)

cv2.destroyAllWindows()