# Significant re-factoring in classes
from picamera2 import Picamera2
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
import time
from tkinter import filedialog
from PIL import Image, ImageTk
import traceback
from dataclasses import dataclass

class AppConfig:
    # Variables Configuration

    # Camera resolution
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    # Camera window size
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 450

    # Application window size
    WINDOW_WIDTH = 1250
    WINDOW_HEIGHT = 950

    # OpenCV font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    MASK_FONT_SCALE = 0.6

    # Frame rate. Number increase, lowers refresh rate and CPU usage, also lowers detection accuracy
    GUI_UPDATE_DELAY = 30

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

    MAX_LOG_LINES = 300

    # Pre-created morphology kernel matrix
    MORPH_KERNEL = np.ones(
        (KERNEL_SIZE, KERNEL_SIZE),
        np.uint8
    )

    # Colors (BGR format for OpenCV)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (30, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (255, 255, 0)


class Messages:
    LAYOUT_LOADED = "Layout loaded"
    LAYOUT_SAVED = "Layout saved"

    PREVIEW_DESTROYED = "Preview stopped"

    ZONE_UPDATED = "Zone updated"
    ZONE_ADDED = "New zone added"

    BACKGROUND_CAPTURED = "Background reference captured"

    DETECTION_ARMED = "Occupancy detection armed"

    INVALID_ZONE_POINTS = "Zone requires 4 valid points"

    INVALID_JSON_FILE = "Selected JSON file is invalid"

    EMPTY_ZONE_NAME = "Zone name cannot be empty"

    NO_CAMERA_FRAME = "No camera frame available yet"

    PREVIEW_UPDATE_FAILED = "Preview update failed"

    INVALID_COORDINATE_FORMAT = "Invalid coordinate format"

    ZONE_DELETED_PREFIX = "Deleted zone:"

    EXITING = "Exiting"


class EventType:
    ENTERED = "ENTERED"
    EXITED = "EXITED"

ADD_NEW_ZONE_OPTION = "ADD NEW ZONE"

@dataclass
class OccupancyResult:
    occupied: bool
    occupancy_pixels: int
    occupancy_ratio: float


class OccupancyAnalyzer:

    def analyze_zone(
        self,
        zone_mask: np.ndarray,
        total_zone_pixels: int) -> OccupancyResult:

        occupancy_pixels = cv2.countNonZero(zone_mask)

        if total_zone_pixels == 0:
            return OccupancyResult(
                occupied=False,
                occupancy_pixels=occupancy_pixels,
                occupancy_ratio=0
            )

        occupancy_ratio = (
            occupancy_pixels / total_zone_pixels
        )

        occupied = (
            occupancy_ratio > AppConfig.OCCUPANCY_PERCENT
        )

        return OccupancyResult(
            occupied=occupied,
            occupancy_pixels=occupancy_pixels,
            occupancy_ratio=occupancy_ratio
        )


@dataclass(frozen=True)
class FrameContext:
    raw_frame: np.ndarray
    gray_frame: np.ndarray | None
    motion_mask: np.ndarray | None


class OccupancyDetector:
    def __init__(self):

        self.background_frame: np.ndarray | None = None

        self.gray_frame: np.ndarray | None = None

        self.mask: np.ndarray | None = None

    def create_difference_mask(self, gray_frame: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(
            self.background_frame,
            gray_frame
        )

        _, binary_mask = cv2.threshold(
            diff,
            AppConfig.DIFF_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )

        return binary_mask
  
    def remove_noise(self, motion_mask: np.ndarray) -> np.ndarray:
        clean_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            AppConfig.MORPH_KERNEL
        )

        return clean_mask
    

    def process_frame(self, frame: np.ndarray) -> FrameContext:
        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if self.background_frame is None:
            return FrameContext(
                raw_frame=frame,
                gray_frame=gray_frame,
                motion_mask=None
            )

        motion_mask = self.create_difference_mask(
            gray_frame
        )

        if AppConfig.USE_CLEANUP:
            motion_mask = self.remove_noise(
                motion_mask
            )
        
        self.gray_frame = gray_frame
        self.mask = motion_mask

        return FrameContext(
            raw_frame=frame,
            gray_frame=gray_frame,
            motion_mask=motion_mask
        )


class LayoutPersistence:
    def save_layout(self, file_path: str, zones: list["Zone"]):
        zones_to_save = []

        for zone in zones:

            zones_to_save.append({
                "name": zone.name,
                "points": zone.points
            })

        layout_data = {
            "zones": zones_to_save
        }

        with open(file_path, "w") as file:
            json.dump(layout_data, file, indent=4)

    def load_layout(self, file_path: str) -> list["Zone"]:
        with open(file_path, "r") as file:
            layout_data = json.load(file)
        if "zones" not in layout_data:
            raise ValueError("Invalid layout file")

        if not isinstance(layout_data["zones"], list):
            raise ValueError("Zones must be a list")

        zones = []

        for zone_data in layout_data["zones"]:

            if "name" not in zone_data or "points" not in zone_data:
                raise ValueError("Invalid zone format")

            if len(zone_data["points"]) != 4:
                raise ValueError("Zone must contain exactly 4 points")

            zone = Zone(
                zone_data["name"],
                zone_data["points"]
            )

            zones.append(zone)

        return zones


class ZoneRepository:
    def __init__(self):
        self.zones: list["Zone"] = []

    def add_zone(self, name: str, points: list[list[int]]) -> None:
        zone = Zone(name, points)
        self.zones.append(zone)

    def update_zone(self, index: int, name: str, points: list[list[int]]) -> None:
        zone = self.zones[index]
        zone.name = name
        zone.update_points(points)
    
    def delete_zone(self, index: int) -> None:
        del self.zones[index]

    def set_zones(self, zones: list["Zone"]) -> None:
        self.zones = zones


class ZoneProcessor:
    def __init__(
        self,
        repository: ZoneRepository,
        occupancy_analyzer: OccupancyAnalyzer):

        self.repository = repository
        self.occupancy_analyzer = occupancy_analyzer

    def extract_zone(self, mask: np.ndarray, zone: "Zone") -> np.ndarray:
        zone_mask = cv2.bitwise_and(
            mask,
            zone.geometry.polygon_mask
        )
        return zone_mask

    def process_zone(
        self,
        detector: OccupancyDetector,
        zone: "Zone") -> tuple[str, str] | None:

        zone_mask = self.extract_zone(
            detector.mask,
            zone
        )

        result = self.occupancy_analyzer.analyze_zone(
            zone_mask,
            zone.geometry.total_pixels
        )

        previous_occupied = zone.occupied

        zone.apply_analysis(result)

        if not previous_occupied and result.occupied:
            return (EventType.ENTERED, zone.name)

        if previous_occupied and not result.occupied:
            return (EventType.EXITED, zone.name)

        return None
 
    # PROCESS ALL ZONES - DETECTION LOGIC
    def process_all_zones(
        self,
        detector: OccupancyDetector,
        detection_enabled: bool) -> list[tuple[str, str]]:

        if detector.mask is None:
            return []

        events = []

        for zone in self.repository.zones:

            if not detection_enabled:
                zone.update_occupancy(False, 0, 0)
                continue

            event = self.process_zone(
                detector,
                zone
            )

            if event is not None:
                events.append(event)

        return events


class ZoneManager:

    def __init__(
        self,
        repository: ZoneRepository,
        processor: ZoneProcessor):

        self.repository = repository
        self.processor = processor

    @property
    def zones(self):
        return self.repository.zones

    def add_zone(
        self,
        name: str,
        points: list[list[int]]):

        self.repository.add_zone(
            name,
            points
        )

    def update_zone(
        self,
        index: int,
        name: str,
        points: list[list[int]]):

        self.repository.update_zone(
            index,
            name,
            points
        )

    def delete_zone(
        self,
        index: int):

        self.repository.delete_zone(index)

    def set_zones(
        self,
        zones: list["Zone"]):

        self.repository.set_zones(zones)

    def process_all_zones(
        self,
        detector: OccupancyDetector,
        detection_enabled: bool):

        return self.processor.process_all_zones(
            detector,
            detection_enabled
        )


class TkImagePresenter:
    def update_camera_preview(
        self,
        camera_label: tk.Label,
        overlay_frame: np.ndarray):

        overlay_frame = cv2.resize(
            overlay_frame,
            (
                AppConfig.DISPLAY_WIDTH,
                AppConfig.DISPLAY_HEIGHT
            )
        )

        rgb_frame = cv2.cvtColor(
            overlay_frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb_frame)

        photo = ImageTk.PhotoImage(image=image)

        camera_label.config(image=photo)

        camera_label.image = photo


    def update_mask_preview(
        self,
        mask_label: tk.Label,
        mask: np.ndarray,
        detection_enabled: bool):

        if mask is None or not detection_enabled:
            return

        mask_frame = cv2.resize(
            mask,
            (
                AppConfig.DISPLAY_WIDTH,
                AppConfig.DISPLAY_HEIGHT
            )
        )

        rgb_mask = cv2.cvtColor(
            mask_frame,
            cv2.COLOR_BGR2RGB
        )

        mask_image = Image.fromarray(rgb_mask)

        mask_photo = ImageTk.PhotoImage(image=mask_image)

        mask_label.config(image=mask_photo)

        mask_label.image = mask_photo


class CameraOverlayRenderer:

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        font_scale: float) -> None:

        cv2.putText(
            frame,
            text,
            (x, y),
            AppConfig.FONT,
            font_scale,
            color,
            AppConfig.FONT_THICKNESS
        )


    def draw_zone_overlays(self,frame: np.ndarray, zone: "Zone", status: str) -> None:
        color = zone.color

        cv2.polylines(
            frame,
            [zone.geometry.points_np],
            True,
            color,
            2
        )

        first_point = zone.geometry.points_np[0]
        text_x = first_point[0]
        text_y = first_point[1]

        overlay_text = f"{zone.name} {status}"

        self.draw_text(
            frame,
            overlay_text,
            text_x,
            text_y - 10,
            color,
            AppConfig.FONT_SCALE
        )

    def get_zone_status_text(self, occupied: bool) -> str:
        if occupied:
            return "OCCUPIED"
        else:
            return "FREE"


class MaskRenderer:

    def render(
        self,
        mask: np.ndarray,
        zones: list["Zone"]) -> np.ndarray:

        mask_color = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )

        mask_color[mask > 0] = AppConfig.COLOR_WHITE

        for zone in zones:

            color = zone.color

            occupancy_percent = (
                zone.occupancy_ratio * 100
            )

            cv2.polylines(
                mask_color,
                [zone.geometry.points_np],
                True,
                color,
                2
            )

            first_point = zone.geometry.points_np[0]

            cv2.putText(
                mask_color,
                f"{occupancy_percent:.1f}%",
                (first_point[0], first_point[1] - 10),
                AppConfig.FONT,
                AppConfig.MASK_FONT_SCALE,
                color,
                AppConfig.FONT_THICKNESS
            )

        return mask_color


class CameraPanel:
    def __init__(self):
        self.camera_label: tk.Label | None = None
        self.mask_label: tk.Label | None = None

    # CREATE CAMERA AREA
    def create_camera_area(self, camera_frame, mask_frame):

        # Prevent frames from resizing larger than content
        camera_frame.pack_propagate(False)
        mask_frame.pack_propagate(False)

        # Match image sizes
        camera_frame.config(
            width=AppConfig.DISPLAY_WIDTH,
            height=AppConfig.DISPLAY_HEIGHT
        )

        mask_frame.config(
            width=AppConfig.DISPLAY_WIDTH,
            height=AppConfig.DISPLAY_HEIGHT
        )

        # ============================================
        # Camera preview
        # ============================================

        self.camera_label = tk.Label(
            camera_frame,
            bd=0
        )

        self.camera_label.pack(
            anchor="w"
        )

        # ============================================
        # Mask preview
        # ============================================

        self.mask_label = tk.Label(
            mask_frame,
            bd=0
        )
        self.mask_label.pack(
            anchor="w"
        )


class LogPanel:
    def __init__(self):
        self.log_text = None


    # CREATE LOG AREA
    def create_log_area(self, right_frame):

        tk.Label(
            right_frame,
            text="System Log"
        ).pack()

        # ============================================
        # Container frame
        # ============================================

        logFrame = tk.Frame(right_frame)

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
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD
        )

        self.log_text.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        self.log_text.config(
            state=tk.DISABLED
        )

        # Connect scrollbar to text widget

        scrollbar.config(
            command=self.log_text.yview
        )


    # LOGGING SYSTEM
    def add_log(self, message: str) -> None:
        self.log_text.config(
            state=tk.NORMAL
        )

        self.log_text.insert(
            tk.END,
            message + "\n"
        )

        # Limit log size
        max_lines = AppConfig.MAX_LOG_LINES

        current_lines = int(
            self.log_text.index('end-1c').split('.')[0]
        )

        if current_lines > max_lines:
            lines_to_delete = current_lines - max_lines
            self.log_text.delete("1.0", f"{lines_to_delete}.0")

        # Auto-scroll to latest message
        self.log_text.see(tk.END)

        self.log_text.config(
            state=tk.DISABLED
        )


class ControlPanel:
    def __init__(self):
        self.capture_button: tk.Button | None = None
        self.arm_button: tk.Button | None = None
        self.stop_button: tk.Button | None = None
        self.load_button: tk.Button | None = None
        self.save_button: tk.Button | None = None
        self.quit_button: tk.Button | None = None

    # CREATE BUTTON AREA
    def create_buttons_area(self, left_frame):

        self.capture_button = tk.Button(
            left_frame,
            text="Capture Background",
            command=None,
            height=2,
            width=25
        )

        self.capture_button.pack(pady=10)

        ToolTip(
            self.capture_button,
            "Capture empty railway image as background reference"
        )

        self.arm_button = tk.Button(
            left_frame,
            text="Arm Detection",
            command=None,
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
            left_frame,
            text="Stop Preview",
            command=None,
            height=2,
            width=25,
            state=tk.DISABLED
        )

        self.stop_button.pack(pady=10)

        ToolTip(
            self.stop_button,
            "Close preview updates"
        )

        self.load_button = tk.Button(
            left_frame,
            text="Load Layout",
            command=None,
            height=2,
            width=25
        )

        self.load_button.pack(pady=10)

        ToolTip(
            self.load_button,
            "Load layout from a JSON file"
        )

        self.save_button = tk.Button(
            left_frame,
            text="Save Layout",
            command=None,
            height=2,
            width=25
        )

        self.save_button.pack(pady=10)

        ToolTip(
            self.save_button,
            "Save layout to JSON file"
        )

        self.quit_button = tk.Button(
            left_frame,
            text="Quit",
            command=None,
            height=2,
            width=25
        )

        self.quit_button.pack(pady=10)

        ToolTip(
            self.quit_button,
            "Quit program"
        )
    
    def enable_arm_button(self):
        self.arm_button.config(state=tk.NORMAL)

    def disable_arm_button(self):
        self.arm_button.config(state=tk.DISABLED)

    def enable_stop_button(self):
        self.stop_button.config(state=tk.NORMAL)

    def disable_stop_button(self):
        self.stop_button.config(state=tk.DISABLED)


class ZoneEditorPanel:
    def __init__(self):
        self.zone_selector: ttk.Combobox | None = None
        self.zone_name_entry: tk.Entry | None = None
        self.point_entries: list[tk.Entry] = []
        self.delete_button: tk.Button | None = None
        self.apply_button: tk.Button | None = None

    # CREATE ZONE EDITOR
    def create_zone_editor(self, center_frame):

        tk.Label(
            center_frame,
            text="Select Zone"
        ).pack()

        zone_names = [ADD_NEW_ZONE_OPTION]

        self.zone_selector = ttk.Combobox(
            center_frame,
            values=zone_names,
            state="readonly",
            width=25
        )

        self.zone_selector.pack(pady=5)

        tk.Label(
            center_frame,
            text="Zone Name"
        ).pack()

        zone_name_frame = tk.Frame(center_frame)

        zone_name_frame.pack(pady=5)

        self.zone_name_entry = tk.Entry(
            zone_name_frame,
            width=18
        )

        self.zone_name_entry.pack(
            side=tk.LEFT,
            padx=5
        )

        self.delete_button = tk.Button(
            zone_name_frame,
            text="X",
            width=3,
            fg="red",
            command=None,
            state=tk.DISABLED
        )

        self.delete_button.pack(side=tk.LEFT)

        ToolTip(
            self.delete_button,
            "Delete selected zone"
        )

        self.zone_selector.current(0)

        self.point_entries.clear()

        for i in range(4):

            label = tk.Label(
                center_frame,
                text=f"Point {i+1} (x,y)"
            )

            label.pack()

            entry = tk.Entry(
                center_frame,
                width=20
            )

            entry.pack(pady=2)

            self.point_entries.append(entry)

        self.apply_button = tk.Button(
            center_frame,
            text="Apply Zone Changes",
            command=None,
            height=2,
            width=25
        )

        self.apply_button.pack(pady=20)

        ToolTip(
            self.apply_button,
            "Apply changes to layout"
        )
    
    def get_zone_name(self) -> str:
        return self.zone_name_entry.get()

    def get_point_texts(self):
        values = []

        for entry in self.point_entries:
            values.append(entry.get())

        return values
    
    def clear_editor(self):
        self.zone_name_entry.delete(0, tk.END)

        for entry in self.point_entries:
            entry.delete(0, tk.END)

        self.delete_button.config(
            state=tk.DISABLED
        )

    def load_zone(self, zone):
        self.delete_button.config(
            state=tk.NORMAL
        )

        self.zone_name_entry.delete(0, tk.END)

        self.zone_name_entry.insert(0, zone.name)

        for i in range(4):

            x = zone.points[i][0]
            y = zone.points[i][1]

            self.point_entries[i].delete(0, tk.END)

            self.point_entries[i].insert(
                0,
                f"{x},{y}"
            )

    def set_zone_name(self, text: str) -> None:
        self.zone_name_entry.delete(0, tk.END)
        self.zone_name_entry.insert(0, text)

    def set_point_text(self, index: int, text: str) -> None:
        entry = self.point_entries[index]
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def clear_point(self, index: int) -> None:
        self.point_entries[index].delete(0, tk.END)

    def enable_delete_button(self) -> None:
        self.delete_button.config(
            state=tk.NORMAL
        )

    def disable_delete_button(self) -> None:

        self.delete_button.config(
            state=tk.DISABLED
        )


class ZoneEditorController:

    def __init__(
        self,
        zone_manager: ZoneManager,
        editor_panel: ZoneEditorPanel,
        log_panel: LogPanel,
        camera_panel: CameraPanel,
        state: "UIState"):

        self.zone_manager = zone_manager
        self.editor_panel = editor_panel
        self.log_panel = log_panel
        self.camera_panel = camera_panel
        self.state = state
    
    # SELECTED TEXTBOX
    def select_point_entry(self, index: int) -> None:
        self.state.active_point_index = index

    # GET CAMERA CLICK
    def on_camera_click(self, event: tk.Event) -> None:
        if self.camera_panel.camera_label is None:
            return

        x = int(event.x * AppConfig.FRAME_WIDTH / AppConfig.DISPLAY_WIDTH)
        y = int(event.y * AppConfig.FRAME_HEIGHT / AppConfig.DISPLAY_HEIGHT)

        self.log_panel.add_log(f"Mouse click: x={x}, y={y}")

        active_index = self.state.active_point_index

        if active_index is not None:

            self.editor_panel.point_entries[active_index].delete(0, tk.END)

            self.editor_panel.point_entries[active_index].insert(
                0,
                f"{x},{y}"
            )

    # LOAD ZONE COORDINATES IN GUI
    def load_zone_into_editor(self, event: tk.Event | None = None):
        selected_index = self.editor_panel.zone_selector.current()

        # ADD NEW ZONE selected
        if selected_index >= len(self.zone_manager.zones):

            self.state.active_point_index = None

            self.editor_panel.clear_editor()

            return

        zone = self.zone_manager.zones[selected_index]

        if len(zone.points) != 4:
            self.log_panel.add_log(
                f"Zone '{zone.name}' has invalid point count"
            )
            return

        self.editor_panel.load_zone(zone)

    # DELETE SELECTED ZONE
    def delete_selected_zone(self):
        selected_index = self.editor_panel.zone_selector.current()

        # Prevent deleting ADD NEW ZONE option
        if selected_index >= len(self.zone_manager.zones):
            return

        deletedName = (self.zone_manager.zones[selected_index].name)

        self.zone_manager.delete_zone(selected_index)

        # Rebuild combobox
        self.refresh_zone_selector()

        self.log_panel.add_log(f"{Messages.ZONE_DELETED_PREFIX} {deletedName}")

    # REFRESH ZONE SELECTOR
    def refresh_zone_selector(self, selectIndex=0):

        zone_names = []

        for zone in self.zone_manager.zones:
            zone_names.append(zone.name)

        zone_names.append(ADD_NEW_ZONE_OPTION)

        self.editor_panel.zone_selector["values"] = zone_names

        self.editor_panel.zone_selector.current(selectIndex)

        self.load_zone_into_editor()

    def get_points_from_editor(self) -> list[list[int]] | None:
        new_points = []
        for text_value in self.editor_panel.get_point_texts():
            try:
                point = CoordinateParser.parse_point(text_value)
                new_points.append(point)
            except ValueError as error:
                self.log_panel.add_log(str(error))
                return None
        return new_points

    def validate_zone_name(self, zone_name):
        if zone_name.strip() == "":
            self.log_panel.add_log(Messages.EMPTY_ZONE_NAME)
            return False

        return True

    # APPLY ZONE COORDINATES CHANGES
    def apply_zone_changes(self) -> None:
        selected_index = self.editor_panel.zone_selector.current()
        new_zone_name = self.editor_panel.zone_name_entry.get()
        new_points = self.get_points_from_editor()

        if new_points is None:
            return

        # Require exactly 4 valid points
        if len(new_points) != 4:
            self.log_panel.add_log(Messages.INVALID_ZONE_POINTS)
            return

        # Prevent empty zone names
        if not self.validate_zone_name(new_zone_name):
            return

        # ============================================
        # ADD NEW ZONE
        # ============================================

        if selected_index >= len(self.zone_manager.zones):
            self.zone_manager.add_zone(
                new_zone_name,
                new_points
            )
            self.log_panel.add_log(Messages.ZONE_ADDED)

        # ============================================
        # UPDATE EXISTING ZONE
        # ============================================

        else:
            self.zone_manager.update_zone(
                selected_index,
                new_zone_name,
                new_points
            )
            self.log_panel.add_log(Messages.ZONE_UPDATED)

        if selected_index >= len(self.zone_manager.zones) - 1:
            self.refresh_zone_selector(
                len(self.zone_manager.zones) - 1
            )
        else:
            self.refresh_zone_selector(selected_index)

    def log_zone_result(self, created: bool) -> None:
        if created:
            self.log_panel.add_log(
                Messages.ZONE_ADDED
            )
        else:
            self.log_panel.add_log(
                Messages.ZONE_UPDATED
            )


class UIState:

    def __init__(self):
        self.preview_enabled = True
        self.detection_enabled = False
        self.active_point_index = None
        self.latest_frame = None


class DetectionController:

    def __init__(
        self,
        engine: "RailwayDetectionEngine",
        control_panel: ControlPanel,
        log_panel: LogPanel,
        state: UIState):

        self.engine = engine
        self.control_panel = control_panel
        self.log_panel = log_panel
        self.state = state


    def capture_background(self):

        success = self.engine.capture_background_reference(self.state.latest_frame)

        if not success:
            self.log_panel.add_log(
                Messages.NO_CAMERA_FRAME
            )
            return

        self.control_panel.enable_arm_button()

        self.log_panel.add_log(
            Messages.BACKGROUND_CAPTURED
        )


    def arm_detection(self):
        self.state.detection_enabled = True
        self.control_panel.disable_arm_button()
        self.control_panel.enable_stop_button()
        self.log_panel.add_log(
            Messages.DETECTION_ARMED
        )


class CoordinateParser:

    @staticmethod
    def parse_point(text_value: str) -> list[int]:

        split_values = text_value.split(",")

        if len(split_values) != 2:
            raise ValueError(Messages.INVALID_COORDINATE_FORMAT)

        x = int(split_values[0])
        y = int(split_values[1])

        if x < 0 or x >= AppConfig.FRAME_WIDTH:
            raise ValueError("X coordinate out of bounds")

        if y < 0 or y >= AppConfig.FRAME_HEIGHT:
            raise ValueError("Y coordinate out of bounds")

        return [x, y]


class CameraService:

    def __init__(self):
        self.picam2 = Picamera2()

    def start(self) -> None:

        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={
                    "size": (
                        AppConfig.FRAME_WIDTH,
                        AppConfig.FRAME_HEIGHT
                    )
                }
            )
        )

        self.picam2.start()

    def capture_frame(self) -> np.ndarray:
        return self.picam2.capture_array()

    def stop(self) -> None:
        self.picam2.stop()


class RailwayDetectionEngine:
    def __init__(self):

        self.detector = OccupancyDetector()

        self.occupancy_analyzer = OccupancyAnalyzer()

        self.zone_repository = ZoneRepository()

        self.zone_processor = ZoneProcessor(
            self.zone_repository,
            self.occupancy_analyzer
        )

        self.zone_manager = ZoneManager(
            self.zone_repository,
            self.zone_processor
        )

    def process_detection(
        self,
        frame: np.ndarray,
        detection_enabled: bool
    ) -> list[tuple[str, str]]:

        self.detector.process_frame(frame)

        if self.detector.background_frame is None:
            return []

        events = self.zone_manager.process_all_zones(
            self.detector,
            detection_enabled
        )

        return events

    def capture_background_reference(self, frame: np.ndarray | None) -> bool:

        if frame is None:
            return False

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.detector.background_frame = gray_frame.copy()

        return True


class PreviewController:

    def __init__(
        self,
        renderer: CameraOverlayRenderer,
        camera_panel: CameraPanel,
        zone_manager: ZoneManager,
        detector: OccupancyDetector,
        mask_renderer: MaskRenderer,
        state: UIState):

        self.renderer = renderer
        self.mask_renderer = mask_renderer
        self.camera_panel = camera_panel
        self.zone_manager = zone_manager
        self.detector = detector
        self.state = state
        self.presenter = TkImagePresenter()
    
    # DRAW GUI
    def update_previews(self, frame: np.ndarray) -> None:

        if (
            not self.state.preview_enabled
            or self.camera_panel.camera_label is None
            or self.camera_panel.mask_label is None):
            return

        overlay_frame = frame.copy()

        for zone in self.zone_manager.zones:
            status = self.renderer.get_zone_status_text(
                zone.occupied
            )

            self.renderer.draw_zone_overlays(overlay_frame, zone, status)

        self.presenter.update_camera_preview(
            self.camera_panel.camera_label,
            overlay_frame
        )

        if self.detector.mask is not None:
            mask_frame = self.mask_renderer.render(
                self.detector.mask,
                self.zone_manager.zones
            )

            self.presenter.update_mask_preview(
                self.camera_panel.mask_label,
                mask_frame,
                self.state.detection_enabled
            )


class FrameProcessor:

    def __init__(
        self,
        engine: RailwayDetectionEngine,
        preview_controller: PreviewController,
        log_panel: LogPanel,
        state: UIState):

        self.engine = engine
        self.preview_controller = preview_controller
        self.log_panel = log_panel
        self.state = state

    def process_frame(
        self,
        frame: np.ndarray) -> None:

        self.run_detection_cycle(frame)

        if not self.state.preview_enabled:
            return

        try:
            self.preview_controller.update_previews(frame)

        except Exception:
            self.log_panel.add_log(
                traceback.format_exc()
            )

    def run_detection_cycle(
        self,
        frame: np.ndarray) -> None:

        events = self.engine.process_detection(
            frame,
            self.state.detection_enabled
        )

        for event_type, zone_name in events:

            if event_type == EventType.ENTERED:
                message = f"Object ENTERED {zone_name}"
            else:
                message = f"Object EXITED {zone_name}"

            self.log_panel.add_log(message)


@dataclass
class ApplicationServices:
    state: UIState
    engine: RailwayDetectionEngine
    layout_persistence: LayoutPersistence
    log_panel: LogPanel
    control_panel: ControlPanel
    camera_panel: CameraPanel
    zone_editor_panel: ZoneEditorPanel
    zone_editor_controller: ZoneEditorController
    detection_controller: DetectionController
    preview_controller: PreviewController
    frame_processor: FrameProcessor
    camera_service: CameraService


class ApplicationLoop:
    def __init__(
        self,
        root: tk.Tk,
        services: ApplicationServices):

        self.root = root
        self.services = services
        self.running = True

        # MAIN APPLICATION LOOP
    def run_frame_loop(self) -> None:
        if not self.running:
            return
        frame = self.capture_frame()

        if frame is not None:
            self.services.frame_processor.process_frame(frame)
            self.services.state.latest_frame = frame

        self.schedule_next_update()

    def capture_frame(self) -> np.ndarray | None:

        try:
            return self.services.camera_service.capture_frame()
        except Exception:
            self.services.log_panel.add_log(
                Messages.PREVIEW_UPDATE_FAILED
            )

            self.services.log_panel.add_log(
                traceback.format_exc()
            )

            return None

    def schedule_next_update(self) -> None:

        self.root.after(
            AppConfig.GUI_UPDATE_DELAY,
            self.run_frame_loop
        )

    def quit_program(self):
        self.running = False
        self.services.log_panel.add_log(Messages.EXITING)
        try:
            self.services.camera_service.stop()
        except Exception:
            self.log_panel.add_log(traceback.format_exc())
        finally:
            self.root.quit()
            self.root.destroy()


class RailwayDetectorApp:
    def __init__(self):

        self.state = UIState()

        self.engine = RailwayDetectionEngine()

        self.layout_persistence = LayoutPersistence()

        self.log_panel = LogPanel()

        self.control_panel = ControlPanel()

        self.camera_panel = CameraPanel()

        self.zone_editor_panel = ZoneEditorPanel()

        self.zone_editor_controller = ZoneEditorController(
            self.engine.zone_manager,
            self.zone_editor_panel,
            self.log_panel,
            self.camera_panel,
            self.state
        )

        self.detection_controller = DetectionController(
            self.engine,
            self.control_panel,
            self.log_panel,
            self.state
        )

        self.camera_overlay_renderer = CameraOverlayRenderer()

        self.mask_renderer = MaskRenderer()

        self.preview_controller = PreviewController(
            self.camera_overlay_renderer,
            self.camera_panel,
            self.engine.zone_manager,
            self.engine.detector,
            self.mask_renderer,
            self.state
        )

        self.frame_processor = FrameProcessor(
            self.engine,
            self.preview_controller,
            self.log_panel,
            self.state
        )

        self.camera_service = CameraService()

        self.camera_service.start()

        self.root = self.create_tkinter_gui()

        self.application_loop = ApplicationLoop(
            self.root,
            ApplicationServices(
                state=self.state,
                engine=self.engine,
                layout_persistence=self.layout_persistence,
                log_panel=self.log_panel,
                control_panel=self.control_panel,
                camera_panel=self.camera_panel,
                zone_editor_panel=self.zone_editor_panel,
                zone_editor_controller=self.zone_editor_controller,
                detection_controller=self.detection_controller,
                preview_controller=self.preview_controller,
                frame_processor=self.frame_processor,
                camera_service=self.camera_service
            )
        )

        self.bind_events()

    def create_tkinter_gui(self) -> tk.Tk:

        root = tk.Tk()

        root.title("Railway Detector")

        root.geometry(
            f"{AppConfig.WINDOW_WIDTH}x{AppConfig.WINDOW_HEIGHT}"
        )

        (
            camera_frame,
            mask_frame,
            left_frame,
            center_frame,
            right_frame
        ) = MainLayoutBuilder.create_main_frames(root)

        # Build UI sections
        self.camera_panel.create_camera_area(
            camera_frame,
            mask_frame
        )

        self.control_panel.create_buttons_area(
            left_frame
        )

        self.zone_editor_panel.create_zone_editor(
            center_frame
        )

        self.log_panel.create_log_area(
            right_frame
        )

        return root
    
    def bind_events(self):
        # Buttons
        self.control_panel.capture_button.config(
            command=self.detection_controller.capture_background
        )

        self.control_panel.arm_button.config(
            command=self.detection_controller.arm_detection
        )

        self.control_panel.save_button.config(
            command=self.save_layout_gui
        )

        self.control_panel.load_button.config(
            command=self.load_layout_gui
        )

        self.control_panel.quit_button.config(
            command=self.application_loop.quit_program
        )

        self.zone_editor_panel.apply_button.config(
            command=self.zone_editor_controller.apply_zone_changes
        )

        self.zone_editor_panel.delete_button.config(
            command=self.zone_editor_controller.delete_selected_zone
        )

        self.zone_editor_panel.zone_selector.bind(
            "<<ComboboxSelected>>",
            self.zone_editor_controller.load_zone_into_editor
        )

        # Point entry focus tracking
        for index, entry in enumerate(self.zone_editor_panel.point_entries):

            entry.bind(
                "<FocusIn>",
                lambda event, i=index:
                self.zone_editor_controller.select_point_entry(i)
            )

        # Camera click
        self.camera_panel.camera_label.bind(
            "<Button-1>",
            self.zone_editor_controller.on_camera_click
        )

    def save_layout_gui(self) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        self.layout_persistence.save_layout(file_path, self.engine.zone_manager.zones)
        self.log_panel.add_log(Messages.LAYOUT_SAVED)
    
    def load_layout_gui(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        try:
            zones = self.layout_persistence.load_layout(file_path)
            self.engine.zone_manager.set_zones(zones)
        except json.JSONDecodeError:
            self.log_panel.add_log(
                Messages.INVALID_JSON_FILE
            )
            return
        except ValueError as error:
            self.log_panel.add_log(str(error))
            return
        except Exception:
            self.log_panel.add_log(
                traceback.format_exc()
            )
            return

        self.zone_editor_controller.refresh_zone_selector()

        self.log_panel.add_log(
            Messages.LAYOUT_LOADED
        )


@dataclass
class ZoneGeometry:
    points_np: np.ndarray
    polygon_mask: np.ndarray
    total_pixels: int


class ZoneGeometryBuilder:
    def build(
        self,
        points: list[list[int]]) -> ZoneGeometry:

        points_np = np.array(
            points,
            dtype=np.int32
        )

        polygon_mask = np.zeros(
            (
                AppConfig.FRAME_HEIGHT,
                AppConfig.FRAME_WIDTH
            ),
            dtype=np.uint8
        )

        cv2.fillPoly(
            polygon_mask,
            [points_np],
            255
        )

        total_pixels = cv2.countNonZero(
            polygon_mask
        )

        return ZoneGeometry(
            points_np=points_np,
            polygon_mask=polygon_mask,
            total_pixels=total_pixels
        )


@dataclass
class Zone:
    name: str
    points: list[list[int]]

    occupied: bool = False
    occupancy_pixels: int = 0
    occupancy_ratio: float = 0

    def __post_init__(self):
        self.rebuild_polygon()

    def update_points(
        self,
        new_points: list[list[int]]) -> None:

        self.points = new_points

        # Rebuild cached geometry
        self.rebuild_polygon()

    def update_occupancy(
        self,
        occupied: bool,
        occupancy_pixels: int,
        occupancy_ratio: float) -> None:

        self.occupied = occupied
        self.occupancy_pixels = occupancy_pixels
        self.occupancy_ratio = occupancy_ratio

    def apply_analysis(
        self,
        result: OccupancyResult) -> None:

        self.occupied = result.occupied
        self.occupancy_pixels = result.occupancy_pixels
        self.occupancy_ratio = result.occupancy_ratio

    @property
    def color(self) -> tuple[int, int, int]:

        if self.occupied:
            return AppConfig.COLOR_RED

        return AppConfig.COLOR_GREEN
    
    def rebuild_polygon(self) -> None:

        builder = ZoneGeometryBuilder()

        self.geometry = builder.build(
            self.points
        )


class MainLayoutBuilder:
    # GUI LAYOUT BUILDER
    @staticmethod
    def create_main_frames(root: tk.Tk):
        # MAIN HORIZONTAL SPLIT
        mainFrame = tk.Frame(root)

        mainFrame.pack(
            anchor="nw",
            padx=10,
            pady=10
        )

        # LEFT COLUMN
        leftColumn = tk.Frame(mainFrame)

        leftColumn.pack(
            side=tk.LEFT,
            anchor="n",
            padx=(0, 10)
        )

        # RIGHT COLUMN
        rightColumn = tk.Frame(mainFrame)

        rightColumn.pack(
            side=tk.LEFT,
            anchor="n"
        )

        # LEFT COLUMN CONTENT
        camera_frame = tk.Frame(leftColumn)

        camera_frame.pack(
            fill=tk.BOTH,
            expand=True,
            pady=(0, 10)
        )

        mask_frame = tk.Frame(leftColumn)

        mask_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        # RIGHT TOP AREA
        rightTopFrame = tk.Frame(rightColumn)

        rightTopFrame.pack(
            fill=tk.X,
            pady=(0, 10)
        )

        # Buttons
        left_frame = tk.Frame(rightTopFrame)

        left_frame.pack(
            side=tk.LEFT,
            anchor="n",
            padx=(0, 15)
        )

        # Zone editor
        center_frame = tk.Frame(rightTopFrame)

        center_frame.pack(
            side=tk.LEFT,
            anchor="n"
        )

        # RIGHT BOTTOM AREA = LOG
        right_frame = tk.Frame(rightColumn)

        right_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        return (
            camera_frame,
            mask_frame,
            left_frame,
            center_frame,
            right_frame
        )


class ToolTip:

    def __init__(self, widget: tk.Widget, text: str):

        self.widget = widget
        self.text = text

        widget.bind("<Enter>", self.schedule_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

        self._tooltip = None
        self._after_id = None

    def show_tooltip(self, event):

        # Already visible
        if self._tooltip is not None:
            return

        x = event.x_root + 10
        y = event.y_root + 10

        self._tooltip = tk.Toplevel(self.widget)

        self._tooltip.wm_overrideredirect(True)
        self._tooltip.geometry(f"+{x}+{y}")

        label = tk.Label(
            self._tooltip,
            text=self.text,
            background="lightyellow",
            relief="solid",
            borderwidth=1
        )

        label.pack()

    def hide_tooltip(self, event):

        # Cancel scheduled tooltip
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

        # Destroy visible tooltip
        if self._tooltip is not None:
            try:
                self._tooltip.destroy()
            except tk.TclError:
                pass

            self._tooltip = None

    def schedule_tooltip(self, event):

        self._after_id = self.widget.after(
            500,
            lambda: self.show_tooltip(event)
        )


# MAIN PROGRAM

app = RailwayDetectorApp()

app.application_loop.run_frame_loop()

app.root.mainloop()
