import cv2
import numpy as np

# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Unable to open webcam")
    exit()

# -----------------------------
# Select ROIS
# -----------------------------
roi = cv2.selectROI("Select Target", frame, False)

if roi == (0, 0, 0, 0):
    print("No ROI Selected")
    cap.release()
    cv2.destroyAllWindows()
    exit()

x, y, w, h = map(int, roi)

# Convert first frame to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Crop ROI
roi_hsv = hsv[y:y+h, x:x+w]

# Create mask for ROI
mask = cv2.inRange(
    roi_hsv,
    np.array((0., 60., 32.)),
    np.array((180., 255., 255.))
)

# Histogram
roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

track_window = (x, y, w, h)

term_crit = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    10,
    1
)

# -----------------------------
# Tracking Loop
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Back Projection
    dst = cv2.calcBackProject(
        [hsv],
        [0],
        roi_hist,
        [0, 180],
        1
    )

    # CamShift
    ret_val, track_window = cv2.CamShift(
        dst,
        track_window,
        term_crit
    )

    # Rotated Rectangle
    pts = cv2.boxPoints(ret_val)
    pts = pts.astype(int)

    tracking = frame.copy()

    cv2.polylines(
        tracking,
        [pts],
        True,
        (0, 255, 0),
        2
    )

    x, y, w, h = track_window

    cv2.circle(
        tracking,
        (int(x + w/2), int(y + h/2)),
        5,
        (0, 0, 255),
        -1
    )

    # -----------------------------
    # Display 4 Outputs
    # -----------------------------

    # 1. Original Image
    cv2.imshow("Select Target", frame)

    # 2. Black & White Mask
    mask_display = cv2.inRange(
        hsv,
        np.array((0., 60., 32.)),
        np.array((180., 255., 255.))
    )
    cv2.imshow("Mask", mask_display)

    # 3. HSV Image
    cv2.imshow("HSV Space", hsv)

    # 4. Final Tracking
    cv2.imshow("CamShift Tracking", tracking)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()