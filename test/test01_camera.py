# Initial test. Verifies that camera is accessible by code

from picamera2 import Picamera2
import cv2

# Initialize camera
picam2 = Picamera2()

# Configure preview resolution
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720)}
    )
)

# Start camera
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Show video
    cv2.imshow("Railway Camera", frame)

    # Exit with Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()