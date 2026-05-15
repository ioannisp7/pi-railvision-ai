# Script starts live camera preview and waits B key from user to set a background image.
# Then it shows overlayed detection zones and occupancy text and in a second window shows
# foreground mask for troubleshooting purposes. Any number of zones can be added. S key
# stops image rendering for freeing up CPU.
from picamera2 import Picamera2
import cv2
import numpy as np

# =========================================================
# Variables Configuration
# =========================================================

# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection zones
# Format:
# (x, y, width, height)
ZONES = [

    {
        "name": "CROSSING_A",
        "rect": (400, 250, 300, 120)
    },
    {
        "name": "STATION_B",
        "rect": (750, 250, 300, 120)
    }
]

# Occupancy threshold in zone level. How MANY changed pixels are required before we declare occupancy
# Larger value = less sensitive
OCCUPANCY_THRESHOLD = 2000

# Difference threshold in pixel level. How different must ONE pixel become before it counts as changed
# Larger value = less sensitive to image changes
DIFF_THRESHOLD = 25

# Enable/disable image cleanup
USE_CLEANUP = True

# Morphology kernel size. Defines aggressiveness of cleanup
# Lower values remove tiny noise but keeps more detail. Higher values remove much more noise but damage image. 5 is
# moderate cleanup strength
KERNEL_SIZE = 5

# Show debug windows
windowsEnabled = True


# =========================================================
# Camera Initialazation
# =========================================================

def initialize_camera():
    picam2 = Picamera2()

    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
        )
    )

    picam2.start()

    return picam2


# =========================================================
# CONVERT FRAME TO GRAYSCALE
# =========================================================

def convert_to_grayscale(frame):
    # Convert color image to grayscale
    # Easier and faster for image comparison
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray


# =========================================================
# CREATE DIFFERENCE MASK
# =========================================================

def create_difference_mask(backgroundFrame, grayFrame):
    # Compare current image against background reference
    # Bright pixels = changed areas
    diff = cv2.absdiff(backgroundFrame, grayFrame)

    # Convert grayscale difference image into:
    # BLACK = no important change
    # WHITE = significant change

    _, thresh = cv2.threshold(
        diff,
        DIFF_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    return thresh


# =========================================================
# CLEANUP MASK
# =========================================================

def cleanup_mask(mask):
    # Create kernel matrix
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    # Morphological opening:
    # removes small white noise
    cleanMask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleanMask


# =========================================================
# EXTRACT ZONE FROM IMAGE
# =========================================================

def extract_zone(image, zone):
    x, y, w, h = zone

    # Crop image to zone area
    zoneImage = image[y:y+h, x:x+w]

    return zoneImage


# =========================================================
# CHECK OCCUPANCY
# =========================================================

def check_occupancy(zoneMask):
    # Count white pixels inside zone
    occupancyPixels = cv2.countNonZero(zoneMask)

    # Convert result to True/False
    occupied = occupancyPixels > OCCUPANCY_THRESHOLD

    # occupancyPixels variable is dropped after returned. It is returned here for debugging and tuning
    return occupied, occupancyPixels


# =========================================================
# DRAW OVERLAYS
# =========================================================

def draw_overlay(frame, zone, zoneName, occupied, status):
    x, y, w, h = zone

    # Choose color
    if occupied:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    # Draw rectangle
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    # Draw zone name text
    cv2.putText(
        frame,
        zoneName,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Draw status text
    cv2.putText(
        frame,
        status,
        (x, y + h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

picam2 = initialize_camera()

backgroundFrame = None
railwayWindowInitialized = False
maskWindowInitialized = False

while True:

    # -----------------------------------------------------
    # CAPTURE FRAME
    # -----------------------------------------------------

    frame = picam2.capture_array()

    # -----------------------------------------------------
    # CONVERT TO GRAYSCALE
    # -----------------------------------------------------

    gray = convert_to_grayscale(frame)

    # -----------------------------------------------------
    # RUN DETECTION ONLY IF BACKGROUND EXISTS
    # -----------------------------------------------------

    if backgroundFrame is not None:

        # Create threshold mask
        mask = create_difference_mask(
            backgroundFrame,
            gray
        )

        # Optional cleanup
        if USE_CLEANUP:
            mask = cleanup_mask(mask)

        for zoneData in ZONES:
            zoneRect = zoneData["rect"]
            # Extract only detection zone
            zoneMask = extract_zone(mask, zoneRect)
            occupied, _ = check_occupancy(zoneMask)

            # Create status text
            if occupied:
                status = "OCCUPIED"
                print(zoneData["name"] + " is occupied")
            else:
                status = "FREE"

            # Draw overlays
            if windowsEnabled:

                draw_overlay(
                    frame,
                    zoneRect,
                    zoneData["name"],
                    occupied,
                    status
                )


    if windowsEnabled:
        
        message = "Press B to capture background, S to stop live preview, Q to quit"
        cv2.putText(
        frame,
        message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

        # -----------------------------------------------------
        # SHOW CAMERA WINDOW
        # -----------------------------------------------------

        cv2.imshow("Railway Vision", frame)
        if not railwayWindowInitialized:
            cv2.moveWindow("Railway Vision", 50, 50)
            railwayWindowInitialized = True

        # -----------------------------------------------------
        # SHOW MASK WINDOW
        # -----------------------------------------------------

        if backgroundFrame is not None:
            cv2.imshow("Threshold Mask", mask)
            if not maskWindowInitialized:
                cv2.moveWindow("Threshold Mask", 1400, 50)
                maskWindowInitialized = True


    # -----------------------------------------------------
    # KEYBOARD INPUT
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # Capture background reference
    if key == ord('b'):
        backgroundFrame = gray.copy()
        print("Background reference captured")
    
    # Hide debug windows
    if key == ord('s'):
        windowsEnabled = False
        print("Debug windows disabled")

    # Quit program
    if key == ord('q'):
        print("Exiting")
        break


# =========================================================
# CLEANUP
# =========================================================

cv2.destroyAllWindows()
