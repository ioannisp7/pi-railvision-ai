# Added the ability to set zones in every quadrilateral shape
from picamera2 import Picamera2
import cv2
import numpy as np

# =========================================================
# Variables Configuration
# =========================================================

# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection zones. Provide coordinates for each of the 4 points
ZONES = [

    {
        "name": "CROSSING_A",
        "points": [
            [400, 250],
            [700, 250],
            [700, 370],
            [400, 370]
        ],
        "occupied": False,
        "previousOccupied": False
    },

    {
        "name": "STATION_B",
        "points": [
            [750, 250],
            [1050, 250],
            [1050, 370],
            [750, 370]
        ],
        "occupied": False,
        "previousOccupied": False
    }
]


ui_state = {

    # Mouse state
    "mouseX": 0,
    "mouseY": 0,

    # Selected zone
    "selected_zone_index": 0,

    # GUI visibility
    "show_mask": True,

    # Show debug windows
    "windows_enabled": True,
    # Start with detection disarmed
    "detection_enabled": False
}

BUTTONS = {
    # Buttons format (x, y, width, height)
    "capture_background": (20, 20, 250, 50),
    "arm_detection": (20, 90, 250, 50),
    "quit": (20, 160, 250, 50)

}

# GUI panel size
GUI_WIDTH = 400
GUI_HEIGHT = 720

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

def extract_zone(mask, points):

    # Create empty black mask
    polygonMask = np.zeros_like(mask)

    # Convert coordinates to integer format
    points = np.int32(points)

    # Draw filled polygon
    cv2.fillPoly(
        polygonMask,
        [points],
        255
    )

    # Keep only pixels inside polygon
    result = cv2.bitwise_and(
        mask,
        polygonMask
    )

    return result


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
# PROCESS ALL ZONES
# =========================================================

def process_all_zones(mask, ui_state):

    for zoneData in ZONES:
        zonePoints = zoneData["points"]
        # Extract only detection zone area
        zoneMask = extract_zone(mask, zonePoints)

        # While detection is disarmed continue to next loop
        if not ui_state["detection_enabled"]:
            zoneData["occupied"] = False
            continue

        # When detection is armed
        # Check occupance status
        occupied, occupancyPixels = check_occupancy(zoneMask)
        # Store previous state of Zone's occupied status to check if changed
        previousOccupied = zoneData["occupied"]
        # Current state of Zone's occupied status
        zoneData["occupied"] = occupied

        # Check if occupancy changed
        # ENTER event
        if not previousOccupied and occupied:
            print("Train ENTERED " + zoneData["name"])

        # EXIT event
        if previousOccupied and not occupied:
            print("Train EXITED " + zoneData["name"])


# =========================================================
# DRAW ZONE OVERLAYS
# =========================================================

def draw_zone_ovelays(frame, points, zoneName, occupied, status):

    points = np.int32(points)

    # Choose color
    if occupied:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    # Draw polygon outline
    cv2.polylines(
        frame,
        [points],
        True,
        color,
        2
    )

    # Use first point for text anchor
    textX = points[0][0]
    textY = points[0][1]

    # Draw zone name
    cv2.putText(
        frame,
        zoneName,
        (textX, textY - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Draw status
    cv2.putText(
        frame,
        status,
        (textX, textY + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


# =========================================================
# Draw help test on the screen
# =========================================================


def draw_help_text(frame, ui_state):

    message = "B: capture background | A: arm detection | S: stop preview | Q: quit"

    cv2.putText(
        frame,
        message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    mouseText = (f"Mouse: {ui_state['mouseX']}, {ui_state['mouseY']}"
)

    cv2.putText(
        frame,
        mouseText,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


# =========================================================
# MOUSE CALLBACK
# =========================================================

def mouse_callback(event, x, y, flags, ui_state):

    # Left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        ui_state["mouseX"] = x
        ui_state["mouseY"] = y
        print(f"Mouse click: x={x}, y={y}")


# =========================================================
# CREATE GUI PANEL
# =========================================================

def create_gui_panel():

    # Dark gray background
    panel = np.zeros((GUI_HEIGHT, GUI_WIDTH, 3), dtype=np.uint8)

    # BGR color coding
    panel[:] = (40, 40, 40)

    return panel


# =========================================================
# DRAW BUTTON
# =========================================================

def draw_button(panel, rect, text, color):

    x, y, w, h = rect

    # Button background
    cv2.rectangle(
        panel,
        (x, y),
        (x + w, y + h),
        color,
        -1
    )

    # Button border
    cv2.rectangle(
        panel,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        2
    )

    # Button text
    cv2.putText(
        panel,
        text,
        (x + 15, y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


# =========================================================
# DRAW GUI PANEL CONTENT
# =========================================================

def draw_gui_panel(panel):

    # Title
    cv2.putText(
        panel,
        "Railway Control Panel",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Buttons
    draw_button(
        panel,
        BUTTONS["capture_background"],
        "Capture Background",
        # BGR color coding
        (80, 80, 200)
    )

    draw_button(
        panel,
        BUTTONS["arm_detection"],
        "Arm Detection",
        (80, 150, 80)
    )

    draw_button(
        panel,
        BUTTONS["quit"],
        "Quit",
        (80, 80, 200)
    )


# =========================================================
# DRAW GUI
# =========================================================

def draw_gui(frame, mask, ui_state):
    global railwayWindowInitialized
    global maskWindowInitialized

    # Draw GUI only if debug windows are enabled
    if not ui_state["windows_enabled"]:
        return

    # Draw all zones
    for zoneData in ZONES:
        if zoneData["occupied"]:
            status = "OCCUPIED"
        else:
            status = "FREE"

        draw_zone_ovelays(
            frame,
            zoneData["points"],
            zoneData["name"],
            zoneData["occupied"],
            status
        )
    
    # Create GUI panel
    panel = create_gui_panel()

    # Draw GUI controls
    draw_gui_panel(panel)

    # Show GUI panel
    cv2.imshow("Control Panel", panel)

    cv2.setMouseCallback(
        "Control Panel",
        gui_mouse_callback
    )

    # Draw help text
    draw_help_text(frame, ui_state)

    # Main camera window
    cv2.imshow("Railway Vision", frame)
    if not railwayWindowInitialized:
        cv2.moveWindow("Railway Vision",50,50)
        railwayWindowInitialized = True

    # Threshold mask window
    if backgroundFrame is not None and ui_state["detection_enabled"] and mask is not None:
        cv2.imshow("Threshold Mask",mask)
        if not maskWindowInitialized:
            cv2.moveWindow("Threshold Mask",1400,50)
            maskWindowInitialized = True
    
    # Mouse callback
    cv2.setMouseCallback("Railway Vision", mouse_callback, ui_state)


# =========================================================
# GUI PANEL CALLBACK
# =========================================================

def gui_mouse_callback(event, x, y, flags, param):

    global backgroundFrame
    global detectionEnabled

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Capture background button
    bx, by, bw, bh = BUTTONS["capture_background"]

    if bx <= x <= bx + bw and by <= y <= by + bh:
        backgroundFrame = gray.copy()
        print("Background reference captured")
        return

    # Arm detection button
    bx, by, bw, bh = BUTTONS["arm_detection"]

    if bx <= x <= bx + bw and by <= y <= by + bh:
        detectionEnabled = True
        print("Occupancy detection armed")
        return

    # Quit button
    bx, by, bw, bh = BUTTONS["quit"]

    if bx <= x <= bx + bw and by <= y <= by + bh:
        print("Exiting")
        exit()


# =========================================================
# MAIN PROGRAM
# =========================================================

picam2 = initialize_camera()

# Initialise variables
backgroundFrame = None
railwayWindowInitialized = False
maskWindowInitialized = False
mask = None

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
        
        process_all_zones(mask, ui_state)

    draw_gui(frame, mask, ui_state)

    # -----------------------------------------------------
    # KEYBOARD INPUT
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # Capture background reference
    if key == ord('b'):
        backgroundFrame = gray.copy()
        print("Background reference captured")

    # Arm occupancy detection
    if key == ord('a'):
        ui_state["detection_enabled"] = True
        print("Occupancy detection armed")
    
    # Hide debug windows
    if key == ord('s'):
        ui_state["windows_enabled"] = False
        print("Debug windows disabled")

    # Quit program
    if key == ord('q'):
        print("Exiting")
        break


# =========================================================
# CLEANUP
# =========================================================

cv2.destroyAllWindows()
