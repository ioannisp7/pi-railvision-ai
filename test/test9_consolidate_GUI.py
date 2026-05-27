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

# Camera window size
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 450

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

colors = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (255, 255, 0)
}

class OccupancyDetector:
    def __init__(self):
        self.background_frame = None
        self.gray_frame = None
        self.mask = None

    # =========================================================
    # CONVERT FRAME TO GRAYSCALE
    # =========================================================

    def convert_to_grayscale(self, frame):
        # Convert color image to grayscale
        # Easier and faster for image comparison
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # =========================================================
    # CREATE DIFFERENCE MASK
    # =========================================================

    def create_difference_mask(self):
        # Compare current image against background reference
        # Bright pixels = changed areas
        diff = cv2.absdiff(self.background_frame, self.gray_frame)

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

    def cleanup_mask(self):
        # Morphological opening:
        # removes small white noise
        cleanMask = cv2.morphologyEx(self.mask,cv2.MORPH_OPEN,MORPH_KERNEL)
        return cleanMask
    
    # =========================================================
    # CHECK OCCUPANCY
    # =========================================================

    def check_occupancy(self, zoneMask, polygonMask):

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

    def process_frame(self, frame):
        self.convert_to_grayscale(frame)

        if self.background_frame is None:
            return

        self.mask = self.create_difference_mask()

        if USE_CLEANUP:
            self.mask = self.cleanup_mask()


class ZoneManager:

    def __init__(self):

        self.zones = []
    
    def extract_zone(self, mask, zone):
        zoneMask = cv2.bitwise_and(
            mask,
            zone.polygon_mask
        )
        return zoneMask
    
    # =========================================================
    # SAVE LAYOUT
    # =========================================================

    def save_layout(self, filePath):

        zonesToSave = []

        for zone in self.zones:

            zonesToSave.append({
                "name": zone.name,
                "points": zone.points
            })

        layoutData = {
            "zones": zonesToSave
        }

        with open(filePath, "w") as file:
            json.dump(layoutData, file, indent=4)


    # =========================================================
    # LOAD LAYOUT
    # =========================================================

    def load_layout(self, filePath):
        with open(filePath, "r") as file:
            layoutData = json.load(file)

        self.zones.clear()

        for zone in layoutData["zones"]:
            zone = Zone(
                zone["name"],
                zone["points"]
            )
            self.zones.append(zone)


class PreviewRenderer:

    def draw_zone_overlays(
        self,
        frame,
        points,
        zoneName,
        occupied,
        status
    ):

        if occupied:
            color = colors["red"]
        else:
            color = colors["green"]

        cv2.polylines(
            frame,
            [points],
            True,
            color,
            2
        )

        textX = points[0][0]
        textY = points[0][1]

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

    def update_camera_preview(
        self,
        camera_label,
        displayFrame
    ):

        displayFrame = cv2.resize(
            displayFrame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )

        rgbFrame = cv2.cvtColor(
            displayFrame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgbFrame)

        photo = ImageTk.PhotoImage(image=image)

        camera_label.config(
            image=photo
        )

        camera_label.image = photo

    def update_mask_preview(
        self,
        mask_label,
        detector,
        zone_manager
    ):
        if (
            detector.background_frame is None
            or detector.mask is None
            or not app.detection_enabled
        ):
            return

        maskColor = cv2.cvtColor(
            detector.mask,
            cv2.COLOR_GRAY2BGR
        )

        maskColor[detector.mask > 0] = colors["white"]

        for zone in zone_manager.zones:

            if zone.occupied:
                color = colors["red"]
            else:
                color = colors["green"]

            zoneMask = zone_manager.extract_zone(
                detector.mask,
                zone
            )

            (
                occupied,
                occupancyPixels,
                occupancyRatio
            ) = detector.check_occupancy(
                zoneMask,
                zone.polygon_mask
            )

            occupancyPercent = occupancyRatio * 100

            cv2.polylines(
                maskColor,
                [zone.points_np],
                True,
                color,
                2
            )

            textX = zone.points_np[0][0]
            textY = zone.points_np[0][1]

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
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
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

        mask_label.config(
            image=maskPhoto
        )

        mask_label.image = maskPhoto


# =========================================================
# MAIN APPLICATION CLASS
# =========================================================

class RailwayDetectorApp:
    def __init__(self):
        self.active_point_index = None

        self.windows_enabled = True
        self.detection_enabled = False

        self.zone_selector = None
        self.zone_name_entry = None
        self.point_entries = []

        self.log_text = None

        self.arm_button = None
        self.stop_button = None
        self.delete_button = None

        self.camera_label = None
        self.mask_label = None

        self.running = True
        self.detector = OccupancyDetector()
        self.zone_manager = ZoneManager()
        self.preview_renderer = PreviewRenderer()
        self.picam2 = self.initialize_camera()

        self.root = self.create_tkinter_gui()

    # =========================================================
    # CREATE TKINTER GUI
    # =========================================================

    def create_tkinter_gui(self):

        self.root = tk.Tk()

        self.root.title("Railway Control Panel")

        self.root.geometry("1250x950")

        self.root.lift()

        self.root.attributes("-topmost", True)

        self.root.after(
            100,
            lambda: self.root.attributes("-topmost", False)
        )

        (
            cameraFrame,
            maskFrame,
            leftFrame,
            centerFrame,
            rightFrame
        ) = self.create_main_frames(self.root)

        self.create_camera_area(
            cameraFrame,
            maskFrame
        )

        self.create_buttons_area(leftFrame)

        self.create_zone_editor(centerFrame)

        self.create_log_area(rightFrame)

        return self.root
    
    # =========================================================
    # MAIN UPDATE LOOP
    # =========================================================

    def update_loop(self):

        if not self.running:
            return

        # Capture frame
        try:
            frame = self.picam2.capture_array()
        except Exception as error:
            self.add_log(f"Camera error: {error}")
            self.root.after(1000, self.update_loop)
            return

        self.detector.process_frame(frame)

        if self.detector.background_frame is not None:
            self.process_all_zones()

        # GUI preview update
        if self.windows_enabled:
            try:
                self.update_preview_windows(frame)
            except Exception as error:
                self.add_log(f"GUI error: {error}")

        # Frame rate. Number increase, lowers refresh rate and CPU usage, also lowers detection accuracy
        self.root.after(30, self.update_loop)

    # =========================================================
    # PROCESS ALL ZONES
    # =========================================================

    def process_all_zones(self):
        for zone in self.zone_manager.zones:
            # Extract only detection zone area
            zoneMask = self.zone_manager.extract_zone(
                self.detector.mask,
                zone
            )
            polygonMask = zone.polygon_mask

            # While detection is disarmed continue to next loop
            if not self.detection_enabled:
                zone.occupied = False
                continue

            # When detection is armed
            # Check occupance status
            (occupied, occupancyPixels, occupancyRatio) = self.detector.check_occupancy(zoneMask, polygonMask)
            # Store previous state of Zone's occupied status to check if changed
            previousOccupied = zone.occupied
            # Current state of Zone's occupied status
            zone.occupied = occupied

            # Check if occupancy changed
            # ENTER event
            if not previousOccupied and occupied:
                self.add_log("Train ENTERED " + zone.name)

            # EXIT event
            if previousOccupied and not occupied:
                self.add_log("Train EXITED " + zone.name)
            
    # =========================================================
    # DRAW GUI
    # =========================================================

    def update_preview_windows(self, frame):

        if (
            not self.windows_enabled
            or self.camera_label is None
        ):
            return

        displayFrame = frame.copy()

        for zone in self.zone_manager.zones:

            if zone.occupied:
                status = "OCCUPIED"
            else:
                status = "FREE"

            self.preview_renderer.draw_zone_overlays(
                displayFrame,
                zone.points_np,
                zone.name,
                zone.occupied,
                status
            )

        self.preview_renderer.update_camera_preview(
            self.camera_label,
            displayFrame
        )

        self.preview_renderer.update_mask_preview(
            self.mask_label,
            self.detector,
            self.zone_manager
        )


    # =========================================================
    # GUI LOG SYSTEM
    # =========================================================

    def add_log(self, message):

        self.log_text.insert(
            tk.END,
            message + "\n"
        )

        # Limit log size
        maxLines = 300

        currentLines = int(
            self.log_text.index('end-1c').split('.')[0]
        )

        if currentLines > maxLines:
            linesToDelete = currentLines - maxLines
            self.log_text.delete("1.0", f"{linesToDelete}.0")

        # Auto-scroll to latest message
        self.log_text.see(tk.END)
    
    # =========================================================
    # GUI BUTTON FUNCTIONS
    # =========================================================

    def capture_background(self):
        self.detector.background_frame = self.detector.gray_frame.copy()
        self.arm_button.config(state=tk.NORMAL)
        self.add_log("Background reference captured")


    def arm_detection(self):
        self.detection_enabled = True
        self.arm_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.add_log("Occupancy detection armed")


    def stop_preview(self):
        self.windows_enabled = False

        self.stop_button.config(
            state=tk.DISABLED
        )

        # Destroy camera preview widget
        if self.camera_label is not None:
            self.camera_label.destroy()
            self.camera_label = None

        # Destroy mask preview widget
        if self.mask_label is not None:
            self.mask_label.destroy()
            self.mask_label = None

        self.add_log("Preview destroyed")


    def quit_program(self):
        self.running = False
        self.add_log("Exiting")
        try:
            self.picam2.stop()
        except Exception as error:
            print(error)
        self.root.quit()
        self.root.destroy()

    # =========================================================
    # CREATE MAIN FRAMES
    # =========================================================

    def create_main_frames(self, root):

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

    def create_camera_area(self, cameraFrame, maskFrame):

        # Prevent frames from resizing larger than content
        cameraFrame.pack_propagate(False)
        maskFrame.pack_propagate(False)

        # Match image sizes
        cameraFrame.config(
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT
        )

        maskFrame.config(
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT
        )

        # ============================================
        # Camera preview
        # ============================================

        self.camera_label = tk.Label(
            cameraFrame,
            bd=0
        )

        self.camera_label.pack(
            anchor="w"
        )

        self.camera_label.bind(
            "<Button-1>",
            self.on_camera_click
        )

        # ============================================
        # Mask preview
        # ============================================

        self.mask_label = tk.Label(
            maskFrame,
            bd=0
        )
        self.mask_label.pack(
            anchor="w"
        )

    # =========================================================
    # CREATE BUTTON AREA
    # =========================================================

    def create_buttons_area(self, leftFrame):

        captureButton = tk.Button(
            leftFrame,
            text="Capture Background",
            command=self.capture_background,
            height=2,
            width=25
        )

        captureButton.pack(pady=10)

        ToolTip(
            captureButton,
            "Capture empty railway image as background reference"
        )

        self.arm_button = tk.Button(
            leftFrame,
            text="Arm Detection",
            command=self.arm_detection,
            height=2,
            width=25,
            state=tk.DISABLED
        )

        self.arm_button.pack(pady=10)

        ToolTip(
            self.arm_button,
            "Enable train occupancy detection"
        )

        self.stop_button = tk.Button(
            leftFrame,
            text="Stop Preview",
            command=self.stop_preview,
            height=2,
            width=25,
            state=tk.DISABLED
        )

        self.stop_button.pack(pady=10)

        ToolTip(
            self.stop_button,
            "Close preview updates"
        )

        loadButton = tk.Button(
            leftFrame,
            text="Load Layout",
            command=self.load_layout_gui,
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
            command=self.save_layout_gui,
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
            command=self.quit_program,
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

    def create_zone_editor(self, centerFrame):

        tk.Label(
            centerFrame,
            text="Select Zone"
        ).pack()

        zoneNames = [ADD_NEW_ZONE_OPTION]

        self.zone_selector = ttk.Combobox(
            centerFrame,
            values=zoneNames,
            state="readonly",
            width=25
        )

        self.zone_selector.pack(pady=5)

        tk.Label(
            centerFrame,
            text="Zone Name"
        ).pack()

        zoneNameFrame = tk.Frame(centerFrame)

        zoneNameFrame.pack(pady=5)

        self.zone_name_entry = tk.Entry(
            zoneNameFrame,
            width=18
        )

        self.zone_name_entry.pack(
            side=tk.LEFT,
            padx=5
        )

        self.delete_button = tk.Button(
            zoneNameFrame,
            text="X",
            width=3,
            fg="red",
            command=self.delete_selected_zone,
            state=tk.DISABLED
        )

        self.delete_button.pack(side=tk.LEFT)

        ToolTip(
            self.delete_button,
            "Delete selected zone"
        )

        self.zone_selector.current(0)

        self.zone_selector.bind(
            "<<ComboboxSelected>>",
            self.load_zone_into_editor
        )

        self.point_entries.clear()

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
                lambda event, index=i: self.select_point_entry(index)
            )

            entry.pack(pady=2)

            self.point_entries.append(entry)

        applyButton = tk.Button(
            centerFrame,
            text="Apply Zone Changes",
            command=self.apply_zone_changes,
            height=2,
            width=25
        )

        applyButton.pack(pady=20)

        ToolTip(
            applyButton,
            "Apply changes to layout"
        )
    
    def save_layout_gui(self):
        filePath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if not filePath:
            return

        self.zone_manager.save_layout(filePath)
        self.add_log("Layout saved")
    
    def load_layout_gui(self):
        filePath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )

        if not filePath:
            return

        self.zone_manager.load_layout(filePath)
        self.refresh_zone_selector()
        self.add_log("Layout loaded")

    # =========================================================
    # CREATE LOG AREA
    # =========================================================

    def create_log_area(self, rightFrame):

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

        self.log_text = tk.Text(
            logFrame,
            width=45,
            yscrollcommand=scrollbar.set
        )

        self.log_text.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        # Connect scrollbar to text widget

        scrollbar.config(
            command=self.log_text.yview
        )


    # =========================================================
    # SELECTED TEXTBOX
    # =========================================================

    def select_point_entry(self, index):
        self.active_point_index = index


    # =========================================================
    # GET CAMERA CLICK
    # =========================================================

    def on_camera_click(self, event):

        x = int(event.x * FRAME_WIDTH / DISPLAY_WIDTH)
        y = int(event.y * FRAME_HEIGHT / DISPLAY_HEIGHT)

        self.add_log(f"Mouse click: x={x}, y={y}")

        activeIndex = self.active_point_index

        if activeIndex is not None:

            self.point_entries[activeIndex].delete(0, tk.END)

            self.point_entries[activeIndex].insert(
                0,
                f"{x},{y}"
            )


    # =========================================================
    # LOAD ZONE COODINATES IN GUI
    # =========================================================

    def load_zone_into_editor(self, event=None):

        selectedIndex = self.zone_selector.current()

        # ADD NEW ZONE selected
        if selectedIndex >= len(self.zone_manager.zones):

            self.delete_button.config(
                state=tk.DISABLED
            )

            self.zone_name_entry.delete(0, tk.END)

            for entry in self.point_entries:
                entry.delete(0, tk.END)

            return

        zone = self.zone_manager.zones[selectedIndex]

        self.delete_button.config(
            state=tk.NORMAL
        )

        self.zone_name_entry.delete(0, tk.END)

        self.zone_name_entry.insert(0,zone.name)

        for i in range(4):
            x = zone.points[i][0]
            y = zone.points[i][1]
            self.point_entries[i].delete(0, tk.END)
            self.point_entries[i].insert(0,f"{x},{y}")


    # =========================================================
    # DELETE SELECTED ZONE
    # =========================================================

    def delete_selected_zone(self):
        selectedIndex = self.zone_selector.current()

        # Prevent deleting ADD NEW ZONE option
        if selectedIndex >= len(self.zone_manager.zones):
            return

        deletedName = self.zone_manager.zones[selectedIndex].name

        del self.zone_manager.zones[selectedIndex]

        # Rebuild combobox
        self.refresh_zone_selector()

        self.add_log(f"Deleted zone: {deletedName}")


    # =========================================================
    # REFRESH ZONE SELECTOR
    # =========================================================

    def refresh_zone_selector(self, selectIndex=0):

        zoneNames = []

        for zone in self.zone_manager.zones:
            zoneNames.append(zone.name)

        zoneNames.append(ADD_NEW_ZONE_OPTION)

        self.zone_selector["values"] = zoneNames

        self.zone_selector.current(selectIndex)

        self.load_zone_into_editor()


    # =========================================================
    # APPLY ZONE COORDINATES CHANGES
    # =========================================================

    def apply_zone_changes(self):
        selectedIndex = self.zone_selector.current()
        newZoneName = self.zone_name_entry.get()
        newPoints = []

        for i in range(4):
            textValue = self.point_entries[i].get()
            splitValues = textValue.split(",")
            try:
                x = int(splitValues[0])
                y = int(splitValues[1])

                if x < 0 or x >= FRAME_WIDTH:
                    self.add_log("X coordinate out of bounds")
                    return
                if y < 0 or y >= FRAME_HEIGHT:
                    self.add_log("Y coordinate out of bounds")
                    return

                newPoints.append([x, y])
            except ValueError:
                self.add_log("Invalid coordinate format")
                return

        # Require exactly 4 valid points
        if len(newPoints) != 4:
            self.add_log("Zone requires 4 valid points")
            return

        # Prevent empty zone names
        if newZoneName.strip() == "":
            self.add_log("Zone name cannot be empty")
            return

        # ============================================
        # ADD NEW ZONE
        # ============================================

        if selectedIndex >= len(self.zone_manager.zones):
            newZone = Zone(
                newZoneName,
                newPoints
            )
            self.zone_manager.zones.append(newZone)
            self.add_log("New zone added")

        # ============================================
        # UPDATE EXISTING ZONE
        # ============================================

        else:
            self.zone_manager.zones[selectedIndex].name = newZoneName
            self.zone_manager.zones[selectedIndex].points = newPoints
            self.zone_manager.zones[selectedIndex].rebuild_polygon()
            self.add_log("Zone updated")

        self.refresh_zone_selector(len(self.zone_manager.zones) - 1)


    # =========================================================
    # Camera Initialazation
    # =========================================================

    def initialize_camera(self):
        picam2 = Picamera2()

        picam2.configure(
            picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
            )
        )
        picam2.start()
        return picam2


# =========================================================
# ZONE CLASS
# =========================================================

class Zone:

    def __init__(self, name, points):

        self.name = name

        self.points = points

        self.rebuild_polygon()

        self.occupied = False
    
    def rebuild_polygon(self):

        self.points_np = np.int32(self.points)

        self.polygon_mask = np.zeros(
            (FRAME_HEIGHT, FRAME_WIDTH),
            dtype=np.uint8
        )

        cv2.fillPoly(
            self.polygon_mask,
            [self.points_np],
            255
        )


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
# MAIN PROGRAM
# =========================================================

app = RailwayDetectorApp()

app.update_loop()

app.root.mainloop()
