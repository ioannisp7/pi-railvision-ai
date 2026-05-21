# Consolidate all windows to one GUI
from picamera2 import Picamera2
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
from tkinter import filedialog
from PIL import Image, ImageTk

# =========================================================
# Variables Configuration
# =========================================================

# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Percentage of changed pixels required to mark a zone as occupied
# 0.08 = 8%
OCCUPANCY_PERCENT = 0.08

# Difference threshold in pixel level. How different must ONE pixel become before it counts as changed
# Larger value = less sensitive to image changes
DIFF_THRESHOLD = 25

# Enable/disable image cleanup
USE_CLEANUP = True

# Morphology kernel size. Defines aggressiveness of cleanup
# Lower values remove tiny noise but keeps more detail. Higher values remove much more noise but damage image. 5 is
# moderate cleanup strength
KERNEL_SIZE = 5

# Pre-created morphology kernel matrix
MORPH_KERNEL = np.ones(
    (KERNEL_SIZE, KERNEL_SIZE),
    np.uint8
)


ADD_NEW_ZONE_OPTION = "ADD NEW ZONE"

# Detection zones. Provide coordinates for each of the 4 points
ZONES = []

ui_state = {
    # Mouse state

    "active_point_index": None,

    # Show debug windows
    "windows_enabled": True,
    # Start with detection disarmed
    "detection_enabled": False,

    "zone_selector": None,
    "zone_name_entry": None,
    "point_entries": [],

    "log_text": None,
    
    "arm_button": None,
    "stop_button": None,
    "delete_button": None,

    "camera_label": None,
    "mask_label": None,

    "display_width": 800,
    "display_height": 450,
}

app_state = {
    "background_frame": None,
    "mask": None,
    "gray_frame": None,
    "running": True
}

colors = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (255, 255, 0)
}

# =========================================================
# ZONE CLASS
# =========================================================

class Zone:

    def __init__(self, name, points):

        self.name = name

        self.points = points

        # Precomputed NumPy polygon
        self.points_np = np.int32(points)

        # Precomputed polygon mask
        self.polygon_mask = np.zeros(
            (FRAME_HEIGHT, FRAME_WIDTH),
            dtype=np.uint8
        )

        cv2.fillPoly(
            self.polygon_mask,
            [self.points_np],
            255
        )

        self.occupied = False


# =========================================================
# Tooltip CLASS
# =========================================================

class ToolTip:

    def __init__(self, widget, text):

        self.widget = widget
        self.text = text

        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

        self.tooltip = None

    def show_tooltip(self, event):

        x = event.x_root + 10
        y = event.y_root + 10

        self.tooltip = tk.Toplevel(self.widget)

        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip,
            text=self.text,
            background="lightyellow",
            relief="solid",
            borderwidth=1
        )

        label.pack()

    def hide_tooltip(self, event):

        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


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


# =========================================================
# CLEANUP MASK
# =========================================================

def cleanup_mask():
    # Morphological opening:
    # removes small white noise
    cleanMask = cv2.morphologyEx(app_state["mask"],cv2.MORPH_OPEN,MORPH_KERNEL)
    return cleanMask


def extract_zone(zoneData):

    zoneMask = cv2.bitwise_and(
        app_state["mask"],
        zoneData.polygon_mask
    )

    return zoneMask


# =========================================================
# CHECK OCCUPANCY
# =========================================================

def check_occupancy(zoneMask, polygonMask):

    # Changed pixels
    occupancyPixels = cv2.countNonZero(zoneMask)

    # Total zone area
    totalZonePixels = cv2.countNonZero(
        polygonMask
    )

    if totalZonePixels == 0:
        return False, occupancyPixels, 0

    occupancyRatio = (
        occupancyPixels / totalZonePixels
    )

    occupied = (
        occupancyRatio > OCCUPANCY_PERCENT
    )

    return (
        occupied,
        occupancyPixels,
        occupancyRatio
    )


# =========================================================
# GET CAMERA CLICK
# =========================================================

def on_camera_click(event):

    x = int(event.x * FRAME_WIDTH / ui_state["display_width"])
    y = int(event.y * FRAME_HEIGHT / ui_state["display_height"])

    ui_state["mouseX"] = x
    ui_state["mouseY"] = y

    add_log(f"Mouse click: x={x}, y={y}")

    activeIndex = ui_state["active_point_index"]

    if activeIndex is not None:

        ui_state["point_entries"][activeIndex].delete(0, tk.END)

        ui_state["point_entries"][activeIndex].insert(
            0,
            f"{x},{y}"
        )


# =========================================================
# PROCESS ALL ZONES
# =========================================================

def process_all_zones():
    for zoneData in ZONES:
        # Extract only detection zone area
        zoneMask = extract_zone(zoneData)
        polygonMask = zoneData.polygon_mask

        # While detection is disarmed continue to next loop
        if not ui_state["detection_enabled"]:
            zoneData.occupied = False
            continue

        # When detection is armed
        # Check occupance status
        (occupied, occupancyPixels, occupancyRatio) = check_occupancy(zoneMask, polygonMask)
        # Store previous state of Zone's occupied status to check if changed
        previousOccupied = zoneData.occupied
        # Current state of Zone's occupied status
        zoneData.occupied = occupied

        # Check if occupancy changed
        # ENTER event
        if not previousOccupied and occupied:
            add_log("Train ENTERED " + zoneData.name)

        # EXIT event
        if previousOccupied and not occupied:
            add_log("Train EXITED " + zoneData.name)


# =========================================================
# DRAW ZONE OVERLAYS
# =========================================================

def draw_zone_overlays(frame, points, zoneName, occupied, status):
    # Choose color
    if occupied:
        color = colors["red"]
    else:
        color = colors["green"]

    # Draw polygon outline
    cv2.polylines(frame,[points],True,color,2)

    # Use first point for text anchor
    textX = points[0][0]
    textY = points[0][1]

    # Draw zone name and status
    overlayText = f"{zoneName} {status}"

    cv2.putText(frame,overlayText,(textX, textY - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)


# =========================================================
# DRAW GUI
# =========================================================

def update_preview_windows(frame):

    if (
        not ui_state["windows_enabled"]
        or ui_state["camera_label"] is None
    ):
        return

    displayFrame = frame.copy()

    for zoneData in ZONES:

        if zoneData.occupied:
            status = "OCCUPIED"
        else:
            status = "FREE"

        draw_zone_overlays(
            displayFrame,
            zoneData.points_np,
            zoneData.name,
            zoneData.occupied,
            status
        )

    update_camera_preview(displayFrame)

    update_mask_preview()


def update_camera_preview(displayFrame):
    displayFrame = cv2.resize(
        displayFrame,
        (
            ui_state["display_width"],
            ui_state["display_height"]
        )
    )

    rgbFrame = cv2.cvtColor(
        displayFrame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgbFrame)

    photo = ImageTk.PhotoImage(image=image)

    ui_state["camera_label"].config(
        image=photo
    )

    ui_state["camera_label"].image = photo


def update_mask_preview():

    if (
        app_state["background_frame"] is None
        or not ui_state["detection_enabled"]
        or app_state["mask"] is None
    ):
        return

    maskColor = cv2.cvtColor(
        app_state["mask"],
        cv2.COLOR_GRAY2BGR
    )

    maskColor[app_state["mask"] > 0] = colors["white"]

    for zoneData in ZONES:
        points = zoneData.points_np
        if zoneData.occupied:
            color = colors["red"]
        else:
            color = colors["green"]

        zoneMask = extract_zone(zoneData)
        polygonMask = zoneData.polygon_mask

        occupancyPixels = cv2.countNonZero(
            zoneMask
        )

        totalPixels = cv2.countNonZero(
            polygonMask
        )

        if totalPixels > 0:

            occupancyPercent = (
                occupancyPixels / totalPixels
            ) * 100

        else:
            occupancyPercent = 0

        cv2.polylines(
            maskColor,
            [zoneData.points_np],
            True,
            color,
            2
        )

        textX = zoneData.points_np[0][0]
        textY = zoneData.points_np[0][1]

        cv2.putText(
            maskColor,
            f"{occupancyPercent:.1f}%",
            (textX, textY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    maskDisplay = cv2.resize(
        maskColor,
        (
            ui_state["display_width"],
            ui_state["display_height"]
        )
    )

    maskRGB = cv2.cvtColor(
        maskDisplay,
        cv2.COLOR_BGR2RGB
    )

    maskImage = Image.fromarray(maskRGB)

    maskPhoto = ImageTk.PhotoImage(
        image=maskImage
    )

    ui_state["mask_label"].config(
        image=maskPhoto
    )

    ui_state["mask_label"].image = maskPhoto


# =========================================================
# GUI LOG SYSTEM
# =========================================================

def add_log(message):

    ui_state["log_text"].insert(
        tk.END,
        message + "\n"
    )

    # Limit log size
    maxLines = 300

    currentLines = int(
        ui_state["log_text"].index('end-1c').split('.')[0]
    )

    if currentLines > maxLines:
        linesToDelete = currentLines - maxLines
        ui_state["log_text"].delete("1.0", f"{linesToDelete}.0")

    # Auto-scroll to latest message
    ui_state["log_text"].see(tk.END)


# =========================================================
# GUI BUTTON FUNCTIONS
# =========================================================

def capture_background():
    app_state["background_frame"] = app_state["gray_frame"].copy()
    ui_state["arm_button"].config(state=tk.NORMAL)
    add_log("Background reference captured")


def arm_detection():
    ui_state["detection_enabled"] = True
    ui_state["arm_button"].config(state=tk.DISABLED)
    ui_state["stop_button"].config(state=tk.NORMAL)
    add_log("Occupancy detection armed")


def stop_preview():
    ui_state["windows_enabled"] = False

    ui_state["stop_button"].config(
        state=tk.DISABLED
    )

    # Destroy camera preview widget
    if ui_state["camera_label"] is not None:
        ui_state["camera_label"].destroy()
        ui_state["camera_label"] = None

    # Destroy mask preview widget
    if ui_state["mask_label"] is not None:
        ui_state["mask_label"].destroy()
        ui_state["mask_label"] = None

    add_log("Preview destroyed")


def quit_program():
    app_state["running"] = False
    add_log("Exiting")
    picam2.stop()
    root.quit()
    root.destroy()


# =========================================================
# LOAD ZONE COODINATES IN GUI
# =========================================================

def load_zone_into_editor(event=None):

    selectedIndex = ui_state["zone_selector"].current()

    # ADD NEW ZONE selected
    if selectedIndex >= len(ZONES):

        ui_state["selected_zone_index"] = -1

        ui_state["delete_button"].config(
            state=tk.DISABLED
        )

        ui_state["zone_name_entry"].delete(0, tk.END)

        for entry in ui_state["point_entries"]:
            entry.delete(0, tk.END)

        return

    ui_state["selected_zone_index"] = selectedIndex

    zoneData = ZONES[selectedIndex]

    ui_state["delete_button"].config(
        state=tk.NORMAL
    )

    ui_state["zone_name_entry"].delete(0, tk.END)

    ui_state["zone_name_entry"].insert(0,zoneData.name)

    for i in range(4):
        x = zoneData.points[i][0]
        y = zoneData.points[i][1]
        ui_state["point_entries"][i].delete(0, tk.END)
        ui_state["point_entries"][i].insert(0,f"{x},{y}")


# =========================================================
# DELETE SELECTED ZONE
# =========================================================

def delete_selected_zone():
    selectedIndex = ui_state["zone_selector"].current()

    # Prevent deleting ADD NEW ZONE option
    if selectedIndex >= len(ZONES):
        return

    deletedName = ZONES[selectedIndex].name

    del ZONES[selectedIndex]

    # Rebuild combobox
    refresh_zone_selector()

    add_log(f"Deleted zone: {deletedName}")


# =========================================================
# REFRESH ZONE SELECTOR
# =========================================================

def refresh_zone_selector(selectIndex=0):

    zoneNames = []

    for zone in ZONES:
        zoneNames.append(zone.name)

    zoneNames.append(ADD_NEW_ZONE_OPTION)

    ui_state["zone_selector"]["values"] = zoneNames

    ui_state["zone_selector"].current(selectIndex)

    load_zone_into_editor()


# =========================================================
# APPLY ZONE COORDINATES CHANGES
# =========================================================

def apply_zone_changes():
    selectedIndex = ui_state["zone_selector"].current()
    newZoneName = ui_state["zone_name_entry"].get()
    newPoints = []

    for i in range(4):
        textValue = ui_state["point_entries"][i].get()
        splitValues = textValue.split(",")
        try:
            x = int(splitValues[0])
            y = int(splitValues[1])
            newPoints.append([x, y])
        except:
            add_log("Invalid coordinate format")
            return

    # Require exactly 4 valid points
    if len(newPoints) != 4:
        add_log("Zone requires 4 valid points")
        return

    # Prevent empty zone names
    if newZoneName.strip() == "":
        add_log("Zone name cannot be empty")
        return

    # ============================================
    # ADD NEW ZONE
    # ============================================

    if selectedIndex >= len(ZONES):
        newZone = Zone(
            newZoneName,
            newPoints
        )
        ZONES.append(newZone)
        add_log("New zone added")

    # ============================================
    # UPDATE EXISTING ZONE
    # ============================================

    else:
        ZONES[selectedIndex].name = newZoneName
        ZONES[selectedIndex].points = newPoints
        ZONES[selectedIndex].points_np = np.int32(newPoints)
        ZONES[selectedIndex].polygon_mask = np.zeros(
            (FRAME_HEIGHT, FRAME_WIDTH),
            dtype=np.uint8
        )

        cv2.fillPoly(
            ZONES[selectedIndex].polygon_mask,
            [ZONES[selectedIndex].points_np],
            255
        )
        add_log("Zone updated")

    refresh_zone_selector(len(ZONES) - 1)


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

    zonesToSave = []

    for zone in ZONES:

        zonesToSave.append({
            "name": zone.name,
            "points": zone.points
        })

    layoutData = {
        "zones": zonesToSave
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

    for zoneData in layoutData["zones"]:

        zone = Zone(
            zoneData["name"],
            zoneData["points"]
        )

        ZONES.append(zone)

    refresh_zone_selector()
    add_log("Layout loaded")


# =========================================================
# CREATE MAIN FRAMES
# =========================================================

def create_main_frames(root):

    # =====================================================
    # MAIN HORIZONTAL SPLIT
    # =====================================================

    mainFrame = tk.Frame(root)

    mainFrame.pack(
        anchor="nw",
        padx=10,
        pady=10
    )

    # =====================================================
    # LEFT COLUMN
    # =====================================================

    leftColumn = tk.Frame(mainFrame)

    leftColumn.pack(
        side=tk.LEFT,
        anchor="n",
        padx=(0, 10)
    )

    # =====================================================
    # RIGHT COLUMN
    # =====================================================

    rightColumn = tk.Frame(mainFrame)

    rightColumn.pack(
        side=tk.LEFT,
        anchor="n"
    )

    # =====================================================
    # LEFT COLUMN CONTENT
    # =====================================================

    cameraFrame = tk.Frame(leftColumn)

    cameraFrame.pack(
        fill=tk.BOTH,
        expand=True,
        pady=(0, 10)
    )

    maskFrame = tk.Frame(leftColumn)

    maskFrame.pack(
        fill=tk.BOTH,
        expand=True
    )

    # =====================================================
    # RIGHT TOP AREA
    # =====================================================

    rightTopFrame = tk.Frame(rightColumn)

    rightTopFrame.pack(
        fill=tk.X,
        pady=(0, 10)
    )

    # Buttons
    leftFrame = tk.Frame(rightTopFrame)

    leftFrame.pack(
        side=tk.LEFT,
        anchor="n",
        padx=(0, 15)
    )

    # Zone editor
    centerFrame = tk.Frame(rightTopFrame)

    centerFrame.pack(
        side=tk.LEFT,
        anchor="n"
    )

    # =====================================================
    # RIGHT BOTTOM AREA = LOG
    # =====================================================

    rightFrame = tk.Frame(rightColumn)

    rightFrame.pack(
        fill=tk.BOTH,
        expand=True
    )

    return (
        cameraFrame,
        maskFrame,
        leftFrame,
        centerFrame,
        rightFrame
    )


# =========================================================
# CREATE CAMERA AREA
# =========================================================

def create_camera_area(cameraFrame, maskFrame):

    # Prevent frames from resizing larger than content
    cameraFrame.pack_propagate(False)
    maskFrame.pack_propagate(False)

    # Match image sizes
    cameraFrame.config(
        width=ui_state["display_width"],
        height=ui_state["display_height"]
    )

    maskFrame.config(
        width=ui_state["display_width"],
        height=ui_state["display_height"]
    )

    # ============================================
    # Camera preview
    # ============================================

    ui_state["camera_label"] = tk.Label(
        cameraFrame,
        bd=0
    )

    ui_state["camera_label"].pack(
        anchor="w"
    )

    ui_state["camera_label"].bind(
        "<Button-1>",
        on_camera_click
    )

    # ============================================
    # Mask preview
    # ============================================

    ui_state["mask_label"] = tk.Label(
        maskFrame,
        bd=0
    )

    ui_state["mask_label"].pack(
        anchor="w"
    )


# =========================================================
# CREATE BUTTON AREA
# =========================================================

def create_buttons_area(leftFrame):

    captureButton = tk.Button(
        leftFrame,
        text="Capture Background",
        command=capture_background,
        height=2,
        width=25
    )

    captureButton.pack(pady=10)

    ToolTip(
        captureButton,
        "Capture empty railway image as background reference"
    )

    ui_state["arm_button"] = tk.Button(
        leftFrame,
        text="Arm Detection",
        command=arm_detection,
        height=2,
        width=25,
        state=tk.DISABLED
    )

    ui_state["arm_button"].pack(pady=10)

    ToolTip(
        ui_state["arm_button"],
        "Enable train occupancy detection"
    )

    ui_state["stop_button"] = tk.Button(
        leftFrame,
        text="Stop Preview",
        command=stop_preview,
        height=2,
        width=25,
        state=tk.DISABLED
    )

    ui_state["stop_button"].pack(pady=10)

    ToolTip(
        ui_state["stop_button"],
        "Close preview updates"
    )

    loadButton = tk.Button(
        leftFrame,
        text="Load Layout",
        command=load_layout,
        height=2,
        width=25
    )

    loadButton.pack(pady=10)

    ToolTip(
        loadButton,
        "Load layout from a JSON file"
    )

    saveButton = tk.Button(
        leftFrame,
        text="Save Layout",
        command=save_layout,
        height=2,
        width=25
    )

    saveButton.pack(pady=10)

    ToolTip(
        saveButton,
        "Save layout to JSON file"
    )

    quitButton = tk.Button(
        leftFrame,
        text="Quit",
        command=quit_program,
        height=2,
        width=25
    )

    quitButton.pack(pady=10)

    ToolTip(
        quitButton,
        "Quit program"
    )


# =========================================================
# CREATE ZONE EDITOR
# =========================================================

def create_zone_editor(centerFrame):

    tk.Label(
        centerFrame,
        text="Select Zone"
    ).pack()

    zoneNames = [ADD_NEW_ZONE_OPTION]

    ui_state["zone_selector"] = ttk.Combobox(
        centerFrame,
        values=zoneNames,
        state="readonly",
        width=25
    )

    ui_state["zone_selector"].pack(pady=5)

    tk.Label(
        centerFrame,
        text="Zone Name"
    ).pack()

    zoneNameFrame = tk.Frame(centerFrame)

    zoneNameFrame.pack(pady=5)

    ui_state["zone_name_entry"] = tk.Entry(
        zoneNameFrame,
        width=18
    )

    ui_state["zone_name_entry"].pack(
        side=tk.LEFT,
        padx=5
    )

    ui_state["delete_button"] = tk.Button(
        zoneNameFrame,
        text="X",
        width=3,
        fg="red",
        command=delete_selected_zone,
        state=tk.DISABLED
    )

    ui_state["delete_button"].pack(side=tk.LEFT)

    ToolTip(
        ui_state["delete_button"],
        "Delete selected zone"
    )

    ui_state["zone_selector"].current(0)

    ui_state["zone_selector"].bind(
        "<<ComboboxSelected>>",
        load_zone_into_editor
    )

    ui_state["point_entries"].clear()

    for i in range(4):

        label = tk.Label(
            centerFrame,
            text=f"Point {i+1} (x,y)"
        )

        label.pack()

        entry = tk.Entry(
            centerFrame,
            width=20
        )

        entry.bind(
            "<Button-1>",
            lambda event, index=i: select_point_entry(index)
        )

        entry.pack(pady=2)

        ui_state["point_entries"].append(entry)

    applyButton = tk.Button(
        centerFrame,
        text="Apply Zone Changes",
        command=apply_zone_changes,
        height=2,
        width=25
    )

    applyButton.pack(pady=20)

    ToolTip(
        applyButton,
        "Apply changes to layout"
    )


# =========================================================
# CREATE LOG AREA
# =========================================================

def create_log_area(rightFrame):

    tk.Label(
        rightFrame,
        text="System Log"
    ).pack()

    # ============================================
    # Container frame
    # ============================================

    logFrame = tk.Frame(rightFrame)

    logFrame.pack(
        fill=tk.BOTH,
        expand=True,
        padx=10,
        pady=10
    )

    # ============================================
    # Scrollbar
    # ============================================

    scrollbar = tk.Scrollbar(logFrame)

    scrollbar.pack(
        side=tk.RIGHT,
        fill=tk.Y
    )

    # ============================================
    # Text widget
    # ============================================

    ui_state["log_text"] = tk.Text(
        logFrame,
        width=45,
        yscrollcommand=scrollbar.set
    )

    ui_state["log_text"].pack(
        side=tk.LEFT,
        fill=tk.BOTH,
        expand=True
    )

    # Connect scrollbar to text widget

    scrollbar.config(
        command=ui_state["log_text"].yview
    )


# =========================================================
# CREATE TKINTER GUI
# =========================================================

def create_tkinter_gui():

    root = tk.Tk()

    root.title("Railway Control Panel")

    root.geometry("1250x950")

    root.lift()

    root.attributes("-topmost", True)

    root.after(
        100,
        lambda: root.attributes("-topmost", False)
    )

    (
        cameraFrame,
        maskFrame,
        leftFrame,
        centerFrame,
        rightFrame
    ) = create_main_frames(root)

    create_camera_area(
        cameraFrame,
        maskFrame
    )

    create_buttons_area(leftFrame)

    create_zone_editor(centerFrame)

    create_log_area(rightFrame)

    return root


# =========================================================
# MAIN UPDATE LOOP
# =========================================================

def update_loop():

    if not app_state["running"]:
        return

    # Capture frame
    try:
        frame = picam2.capture_array()
    except Exception as error:
        add_log(f"Camera error: {error}")
        root.after(1000, update_loop)
        return

    # Convert to grayscale
    convert_to_grayscale(frame)

    # Detection processing
    if app_state["background_frame"] is not None:

        app_state["mask"] = create_difference_mask()

        if USE_CLEANUP:
            app_state["mask"] = cleanup_mask()

        process_all_zones()

    # GUI preview update
    if ui_state["windows_enabled"]:
        try:
            update_preview_windows(frame)
        except Exception as error:
            add_log(f"GUI error: {error}")

    # Frame rate. Number increase, lowers refresh rate and CPU usage, also lowers detection accuracy
    root.after(30, update_loop)


# =========================================================
# MAIN PROGRAM
# =========================================================

picam2 = initialize_camera()

root = create_tkinter_gui()

# Start update loop
update_loop()

# Start Tkinter event loop
root.mainloop()
