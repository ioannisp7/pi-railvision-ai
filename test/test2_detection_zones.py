# Script starts live camera preview with overlayed detection zones and occupancy text
from picamera2 import Picamera2
import cv2

# -----------------------------
# Camera setup
# -----------------------------
picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720)}
    )
)

picam2.start()

# -----------------------------
# Detection zone coordinates
# (x, y, width, height)
# -----------------------------
zone = (400, 250, 300, 120)

while True:

    # Capture frame
    frame = picam2.capture_array()

    # Unpack zone
    x, y, w, h = zone

    # Draw rectangle
    # Last argument "2" is used for thickness
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Zone label
    # Last argument "2" is used for thickness
    cv2.putText(
        frame,
        "CROSSING_ZONE",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Fake occupancy text for now
    # Last argument "2" is used for thickness
    cv2.putText(
        frame,
        "FREE",
        (x, y + h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show video
    cv2.imshow("Railway Vision", frame)

    # Quit with Q
    # 0xFF = Keep only 8 bits for keyboard key value and not extra keyboard metadata
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

