# Added GUI
from picamera2 import Picamera2
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from time import sleep
import json
from tkinter import filedialog

# =========================================================
# Variables Configuration
# =========================================================

ADD_NEW_ZONE_OPTION = "ADD NEW ZONE"

# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection zones. Provide coordinates for each of the 4 points
ZONES = []

ui_state = {
    # Mouse state
    "mouseX": 0,
    "mouseY": 0,

    # Selected zone
    "selected_zone_index": 0,

    "active_point_index": None,

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
# SELECTED TEXTBOX
# =========================================================

def select_point_entry(index):

    ui_state["active_point_index"] = index

    print(f"Selected Point {index + 1}")


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
            add_log("Train ENTERED " + zoneData["name"])

        # EXIT event
        if previousOccupied and not occupied:
            add_log("Train EXITED " + zoneData["name"])


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

    # Draw zone name and status
    overlayText = f"{zoneName} {status}"

    cv2.putText(
        frame,
        overlayText,
        (textX, textY - 10),
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

        activeIndex = ui_state["active_point_index"]

        if activeIndex is not None:

            pointEntries[activeIndex].delete(0, tk.END)

            pointEntries[activeIndex].insert(
                0,
                f"{x},{y}"
            )


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
# GUI LOG SYSTEM
# =========================================================

def add_log(message):

    logText.insert(
        tk.END,
        message + "\n"
    )

    # Auto-scroll to latest message
    logText.see(tk.END)


# =========================================================
# GUI BUTTON FUNCTIONS
# =========================================================

def capture_background():

    app_state["background_frame"] = app_state["gray_frame"].copy()

    armButton.config(
        state=tk.NORMAL
    )

    add_log("Background reference captured")


def arm_detection():

    ui_state["detection_enabled"] = True

    stopButton.config(
        state=tk.NORMAL
    )

    add_log("Occupancy detection armed")


def stop_preview():
    ui_state["windows_enabled"] = False
    cv2.destroyAllWindows()
    add_log("Preview disabled")


def quit_program():
    global running
    running = False
    add_log("Exiting")


# =========================================================
# LOAD ZONE COODINATES IN GUI
# =========================================================

def load_zone_into_editor(event=None):

    selectedIndex = zoneSelector.current()

    # ADD NEW ZONE selected
    if selectedIndex >= len(ZONES):
        ui_state["selected_zone_index"] = -1

        zoneNameEntry.delete(0, tk.END)

        for entry in pointEntries:
            entry.delete(0, tk.END)

        return

    ui_state["selected_zone_index"] = selectedIndex

    zoneData = ZONES[selectedIndex]

    zoneNameEntry.delete(0, tk.END)

    zoneNameEntry.insert(
        0,
        zoneData["name"]
    )

    for i in range(4):

        x = zoneData["points"][i][0]
        y = zoneData["points"][i][1]

        pointEntries[i].delete(0, tk.END)

        pointEntries[i].insert(
            0,
            f"{x},{y}"
        )


# =========================================================
# APPLY ZONE COORDINATES CHANGES
# =========================================================

def apply_zone_changes():

    selectedIndex = zoneSelector.current()

    newZoneName = zoneNameEntry.get()

    newPoints = []

    for i in range(4):

        textValue = pointEntries[i].get()

        splitValues = textValue.split(",")

        x = int(splitValues[0])
        y = int(splitValues[1])

        newPoints.append([x, y])

    # ============================================
    # ADD NEW ZONE
    # ============================================

    if selectedIndex >= len(ZONES):

        newZone = {
            "name": newZoneName,
            "points": newPoints,
            "occupied": False
        }

        ZONES.append(newZone)

        add_log("New zone added")

    # ============================================
    # UPDATE EXISTING ZONE
    # ============================================

    else:

        ZONES[selectedIndex]["name"] = newZoneName

        ZONES[selectedIndex]["points"] = newPoints

        add_log("Zone updated")

    # ============================================
    # Refresh combobox
    # ============================================

    zoneNames = []

    for zone in ZONES:
        zoneNames.append(zone["name"])

    zoneNames.append(ADD_NEW_ZONE_OPTION)

    zoneSelector["values"] = zoneNames

    # Select latest zone
    zoneSelector.current(len(ZONES) - 1)


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

    add_log("Layout saved")


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

    zoneNames = []

    for zone in ZONES:
        zoneNames.append(zone["name"])

    zoneNames.append(ADD_NEW_ZONE_OPTION)

    zoneSelector["values"] = zoneNames

    zoneSelector.current(0)

    load_zone_into_editor()

    add_log("Layout loaded")


# =========================================================
# CREATE TKINTER GUI
# =========================================================

def create_tkinter_gui():

    root = tk.Tk()

    root.title("Railway Control Panel")
    root.geometry("700x500")

    root.lift()
    root.attributes("-topmost", True)
    root.after(100, lambda: root.attributes("-topmost", False))

    # =====================================================
    # LEFT PANEL
    # =====================================================

    leftFrame = tk.Frame(root)
    leftFrame.pack(
        side=tk.LEFT,
        fill=tk.Y,
        padx=20,
        pady=20
    )

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    rightFrame = tk.Frame(root)
    rightFrame.pack(
        side=tk.LEFT,
        fill=tk.BOTH,
        expand=True,
        padx=20,
        pady=20
    )

    # =====================================================
    # LEFT SIDE BUTTONS
    # =====================================================

    captureButton = tk.Button(
        leftFrame,
        text="Capture Background",
        command=capture_background,
        height=2,
        width=25
    )
    captureButton.pack(pady=10)

    global armButton

    armButton = tk.Button(
        leftFrame,
        text="Arm Detection",
        command=arm_detection,
        height=2,
        width=25,
        state=tk.DISABLED
    )
    armButton.pack(pady=10)

    global stopButton

    stopButton = tk.Button(
        leftFrame,
        text="Stop Preview",
        command=stop_preview,
        height=2,
        width=25,
        state=tk.DISABLED
    )
    stopButton.pack(pady=10)

    loadButton = tk.Button(
        leftFrame,
        text="Load Layout",
        command=load_layout,
        height=2,
        width=25
    )
    loadButton.pack(pady=10)

    saveButton = tk.Button(
        leftFrame,
        text="Save Layout",
        command=save_layout,
        height=2,
        width=25
    )
    saveButton.pack(pady=10)

    quitButton = tk.Button(
        leftFrame,
        text="Quit",
        command=quit_program,
        height=2,
        width=25
    )
    quitButton.pack(pady=10)

    # =====================================================
    # RIGHT SIDE ZONE EDITOR
    # =====================================================

    tk.Label(
        rightFrame,
        text="Select Zone"
    ).pack()

    zoneNames = [ADD_NEW_ZONE_OPTION]

    global zoneSelector

    zoneSelector = ttk.Combobox(
        rightFrame,
        values=zoneNames,
        state="readonly",
        width=25
    )

    zoneSelector.pack(pady=5)

    tk.Label(
        rightFrame,
        text="Zone Name"
    ).pack()

    global zoneNameEntry

    zoneNameEntry = tk.Entry(
        rightFrame,
        width=25
    )

    zoneNameEntry.pack(pady=5)

    zoneSelector.current(0)

    zoneSelector.bind(
        "<<ComboboxSelected>>",
        load_zone_into_editor
    )

    global pointEntries

    pointEntries = []

    for i in range(4):

        label = tk.Label(
            rightFrame,
            text=f"Point {i+1} (x,y)"
        )

        label.pack()

        entry = tk.Entry(
            rightFrame,
            width=20
        )

        entry.bind(
            "<Button-1>",
            lambda event, index=i: select_point_entry(index)
        )

        entry.pack(pady=2)

        pointEntries.append(entry)

    applyButton = tk.Button(
        rightFrame,
        text="Apply Zone Changes",
        command=apply_zone_changes,
        height=2,
        width=25
    )
    applyButton.pack(pady=20)

    # =====================================================
    # LOG AREA
    # =====================================================

    tk.Label(
        root,
        text="System Log"
    ).pack()

    global logText

    logText = tk.Text(
        root,
        height=10,
        width=80
    )

    logText.pack(
        padx=10,
        pady=10
    )

    return root


# =========================================================
# MAIN PROGRAM
# =========================================================

picam2 = initialize_camera()

running = True
zoneSelector = None
pointEntries = []
zoneNameEntry = None
logText = None
armButton = None
stopButton = None
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

    draw_gui(frame)

    try:
        root.update_idletasks()
        root.update()
    except tk.TclError:
        running = False

    cv2.waitKey(1)

    # Limit CPU stress 
    sleep(0.01)

# =========================================================
# CLEANUP
# =========================================================

cv2.destroyAllWindows()
picam2.stop()
root.destroy()