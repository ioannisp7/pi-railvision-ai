# The same as previous test "test3_background_subtraction.py" with noise reduction in captured images.
from picamera2 import Picamera2
import cv2
import numpy as np

# =========================================================
# CAMERA INITIALIZATION
# =========================================================

picam2 = Picamera2()

# Configure camera resolution
# Lower resolution = lower CPU usage
# 1280x720 is a good compromise

picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720)}
    )
)

picam2.start()

# =========================================================
# DETECTION ZONE
# =========================================================

# Define one virtual detection zone
# Format:
# (x, y, width, height)

zone = (400, 250, 300, 120)

# =========================================================
# BACKGROUND/REFERENCE FRAME
# =========================================================

# This variable will later store:
# "empty layout image"
#
# Initially there is no reference yet
#
backgroundFrame = None

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # -----------------------------------------------------
    # CAPTURE CURRENT CAMERA FRAME
    # -----------------------------------------------------

    frame = picam2.capture_array()

    # -----------------------------------------------------
    # CONVERT IMAGE TO GRAYSCALE
    # -----------------------------------------------------

    # Convert image to grayscale
    # Easier and faster processing
    # We only care about shape differences
    # not color information yet
    #
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # GET ZONE COORDINATES
    # -----------------------------------------------------

    x, y, w, h = zone

    # -----------------------------------------------------
    # DEFAULT STATUS
    # -----------------------------------------------------

    # Before reference image exists:
    #
    # system status = NOT READY
    #
    occupied = False
    status = "PRESS B TO CAPTURE BACKGROUND"
    color = (255, 255, 0)

    # =====================================================
    # ONLY RUN DETECTION IF REFERENCE EXISTS
    # =====================================================

    if backgroundFrame is not None:

        # -------------------------------------------------
        # CALCULATE DIFFERENCE IMAGE
        # -------------------------------------------------

        # Compare:
        # current frame VS reference background frame
        # Result: bright pixels = changed areas

        diff = cv2.absdiff(backgroundFrame, gray)

        # -------------------------------------------------
        # THRESHOLD IMAGE
        # -------------------------------------------------

        # Convert grayscale difference image into:
        # BLACK = no important change
        # WHITE = significant change
        #
        # Threshold value = 25
        # Smaller value: more sensitive
        # Larger value: less sensitive
        #
        # Function threshold return 2 values and we drop the first

        _, thresh = cv2.threshold(
            diff,
            25,
            255,
            cv2.THRESH_BINARY
        )

        # -------------------------------------------------
        # CLEANUP FILTER
        # -------------------------------------------------

        # Create small kernel matrix
        #
        # Kernel size:
        # 5x5 pixels
        #
        kernel = np.ones((5, 5), np.uint8)

        # Morphological opening:
        #
        # Removes tiny white noise pixels
        # while keeping larger shapes
        #
        cleanMask = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel
        )

        # -------------------------------------------------
        # EXTRACT ONLY DETECTION ZONE
        # -------------------------------------------------

        # Crop threshold image
        # to only the virtual railway zone
        zoneMask = cleanMask[y:y+h, x:x+w]


        # -------------------------------------------------
        # COUNT WHITE PIXELS
        # -------------------------------------------------

        # Count changed pixels inside zone
        # White pixels mean something different exists here
        occupancyPixels = cv2.countNonZero(zoneMask)

        # -------------------------------------------------
        # OCCUPANCY DECISION
        # -------------------------------------------------

        # If enough changed pixels exist zone becomes occupied
        # If occupancyPixels > 2000 then occupied = True else occupied = False

        occupied = occupancyPixels > 2000

        # -------------------------------------------------
        # STATUS DISPLAY
        # -------------------------------------------------

        if occupied:
            color = (0, 0, 255)
            status = "OCCUPIED"
        else:
            color = (0, 255, 0)
            status = "FREE"

        # -------------------------------------------------
        # SHOW DEBUG MASK
        # -------------------------------------------------

        cv2.imshow("Threshold Mask", cleanMask)
        cv2.moveWindow("Threshold Mask", 1400, 50)

    # =====================================================
    # DRAW VISUAL OVERLAYS
    # =====================================================

    # Draw detection rectangle
    # Last argument "2" is used for thickness
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2
    )


    # Draw status text
    # Last argument "2" is used for thickness
    cv2.putText(
        frame,
        status,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # =====================================================
    # SHOW CAMERA IMAGE
    # =====================================================

    cv2.imshow("Railway Vision", frame)
    cv2.moveWindow("Railway Vision", 50, 50)

    # =====================================================
    # KEYBOARD INPUT
    # =====================================================
    # 0xFF = Keep only 8 bits for keyboard key value and not extra keyboard metadata
    
    key = cv2.waitKey(1) & 0xFF

    # -----------------------------------------------------
    # CAPTURE BACKGROUND REFERENCE
    # -----------------------------------------------------

    # Press B key when:
    # - layout is empty
    # - no train inside detection zone
    #
    # This stores the current grayscale image as official background reference

    if key == ord('b'):

        backgroundFrame = gray.copy()

        print("Background reference captured")

    # -----------------------------------------------------
    # EXIT PROGRAM
    # -----------------------------------------------------

    if key == ord('q'):
        break

# =========================================================
# CLEANUP
# =========================================================

cv2.destroyAllWindows()
