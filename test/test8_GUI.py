# Added GUI
from picamera2 import Picamera2
import cv2
import numpy as np
import tkinter as tk
from time import sleep
import json
from tkinter import filedialog

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
    "detection_enabled": False,

    # Initialization of the windows
    "railway_window_initialized": False,
    "mask_window_initialized": False
}

app_state = {
    "background_frame": None,
    "mask": None,
    "gray_frame": None
}

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
    app_state["gray_frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# =========================================================
# CREATE DIFFERENCE MASK
# =========================================================

def create_difference_mask():
    # Compare current image against background reference
    # Bright pixels = changed areas
    diff = cv2.absdiff(app_state["background_frame"], app_state["gray_frame"])

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

def cleanup_mask():
    # Create kernel matrix
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    # Morphological opening:
    # removes small white noise
    cleanMask = cv2.morphologyEx(
        app_state["mask"],
        cv2.MORPH_OPEN,
        kernel
    )

    return cleanMask


# =========================================================
# EXTRACT ZONE FROM IMAGE
# =========================================================

def extract_zone(points):
    # Create empty black mask
    polygonMask = np.zeros_like(app_state["mask"])

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
        app_state["mask"],
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

def process_all_zones():
    for zoneData in ZONES:
        zonePoints = zoneData["points"]
        # Extract only detection zone area
        zoneMask = extract_zone(zonePoints)

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

def draw_zone_overlays(frame, points, zoneName, occupied, status):
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
# Draw mouse coordinates on screen
# =========================================================

def draw_mouse_coordinates(frame):
    mouseText = (f"Mouse: {ui_state['mouseX']}, {ui_state['mouseY']}")

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

def mouse_callback(event, x, y, flags, param):
    # Left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        ui_state["mouseX"] = x
        ui_state["mouseY"] = y
        print(f"Mouse click: x={x}, y={y}")


# =========================================================
# DRAW GUI
# =========================================================

def draw_gui(frame):    
    # Draw GUI only if debug windows are enabled
    if not ui_state["windows_enabled"]:
        return

    # Draw all zones
    for zoneData in ZONES:
        if zoneData["occupied"]:
            status = "OCCUPIED"
        else:
            status = "FREE"

        draw_zone_overlays(
            frame,
            zoneData["points"],
            zoneData["name"],
            zoneData["occupied"],
            status
        )

    # Draw help text
    draw_mouse_coordinates(frame)

    # Main camera window
    cv2.imshow("Railway Vision", frame)
    if not ui_state["railway_window_initialized"]:
        cv2.moveWindow("Railway Vision",50,50)
        ui_state["railway_window_initialized"] = True
        # Mouse callback
        cv2.setMouseCallback("Railway Vision", mouse_callback)

    # Threshold mask window
    if app_state["background_frame"] is not None and ui_state["detection_enabled"] and app_state["mask"] is not None:
        cv2.imshow("Threshold Mask",app_state["mask"])
        if not ui_state["mask_window_initialized"]:
            cv2.moveWindow("Threshold Mask",1400,50)
            ui_state["mask_window_initialized"] = True
    
    # # Mouse callback
    # cv2.setMouseCallback("Railway Vision", mouse_callback)


# =========================================================
# GUI BUTTON FUNCTIONS
# =========================================================

def capture_background():
    app_state["background_frame"] = app_state["gray_frame"].copy()
    print("Background reference captured")


def arm_detection():
    ui_state["detection_enabled"] = True
    print("Occupancy detection armed")


def stop_preview():
    ui_state["windows_enabled"] = False
    cv2.destroyAllWindows()
    print("Preview disabled")


def quit_program():
    global running
    running = False
    print("Exiting")


# =========================================================
# CREATE TKINTER GUI
# =========================================================

def create_tkinter_gui():

    root = tk.Tk()

    root.title("Railway Control Panel")
    root.geometry("400x500")

    captureButton = tk.Button(
        root,
        text="Capture Background",
        command=capture_background,
        height=2,
        width=25
    )
    captureButton.pack(pady=10)

    armButton = tk.Button(
        root,
        text="Arm Detection",
        command=arm_detection,
        height=2,
        width=25
    )
    armButton.pack(pady=10)

    stopButton = tk.Button(
        root,
        text="Stop Preview",
        command=stop_preview,
        height=2,
        width=25
    )
    stopButton.pack(pady=10)

    saveButton = tk.Button(
        root,
        text="Save Layout",
        command=save_layout,
        height=2,
        width=25
    )
    saveButton.pack(pady=10)

    loadButton = tk.Button(
        root,
        text="Load Layout",
        command=load_layout,
        height=2,
        width=25
    )
    loadButton.pack(pady=10)

    quitButton = tk.Button(
        root,
        text="Quit",
        command=quit_program,
        height=2,
        width=25
    )
    quitButton.pack(pady=10)

    return root


# =========================================================
# SAVE LAYOUT
# =========================================================

def save_layout():
    filePath = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")]
    )

    if not filePath:
        return

    layoutData = {
        "zones": ZONES
    }

    with open(filePath, "w") as file:
        json.dump(layoutData, file, indent=4)

    print("Layout saved")


# =========================================================
# LOAD LAYOUT
# =========================================================

def load_layout():
    filePath = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json")]
    )

    if not filePath:
        return

    with open(filePath, "r") as file:
        layoutData = json.load(file)

    ZONES.clear()

    for zone in layoutData["zones"]:
        ZONES.append(zone)

    print("Layout loaded")


# =========================================================
# MAIN PROGRAM
# =========================================================

picam2 = initialize_camera()

running = True
root = create_tkinter_gui()

while running:
    # CAPTURE FRAME
    frame = picam2.capture_array()

    # CONVERT TO GRAYSCALE
    convert_to_grayscale(frame)

    # RUN DETECTION ONLY IF BACKGROUND EXISTS
    if app_state["background_frame"] is not None:

        # Create threshold mask
        app_state["mask"] = create_difference_mask()

        # Optional cleanup
        if USE_CLEANUP:
            app_state["mask"] = cleanup_mask()
        
        process_all_zones()

    # Limit CPU stress 
    sleep(0.01)

    draw_gui(frame)

    root.update_idletasks()
    root.update()

    cv2.waitKey(1)

# =========================================================
# CLEANUP
# =========================================================

cv2.destroyAllWindows()
picam2.stop()
root.destroy()