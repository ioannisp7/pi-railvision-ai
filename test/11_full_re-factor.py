# Full re-factoring of the code
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

@dataclass(frozen=True)
class FrameGeometry:
    width: int
    height: int


@dataclass(frozen=True)
class DetectionConfig:
    occupancy_percent: float
    diff_threshold: int
    use_cleanup: bool
    kernel_size: int


@dataclass(frozen=True)
class DisplayConfig:
    width: int
    height: int


@dataclass(frozen=True)
class TimingConfig:
    gui_update_delay: int

class AppConfig:
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

    ZONE_ADDED_SUFFIX = "zone created"
    ZONE_UPDATED_SUFFIX = "zone updated"
    ZONE_DELETED_SUFFIX = "zone deleted"

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


class ValidationErrorCode:
    INVALID_COORDINATE_FORMAT = "INVALID_COORDINATE_FORMAT"
    COORDINATE_OUT_OF_BOUNDS = "COORDINATE_OUT_OF_BOUNDS"
    EMPTY_ZONE_NAME = "EMPTY_ZONE_NAME"
    INVALID_ZONE_POINTS = "INVALID_ZONE_POINTS"
    INVALID_LAYOUT_FILE = "Invalid layout file"
    ZONES_AS_LIST = "Zones must be a list"
    INVALID_ZONE_FORMAT = "Invalid zone format"
    ZONE_CONTAIN_4_POINTS = "Zone must contain exactly 4 points"
    POINT_AS_LIST = "Point must be a list"
    ZONE_CONTAIN_XY = "Point must contain x,y"
    POINTS_AS_INTEGERS = "Point coordinates must be integers"


class ValidationError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class EventType:
    ENTERED = "ENTERED"
    EXITED = "EXITED"


class EventFormatter:
    @staticmethod
    def format(event_type: str, zone_name: str) -> str:
        if event_type == EventType.ENTERED:
            return f"Object ENTERED {zone_name}"
        return f"Object EXITED {zone_name}"


class EventDetector:
    @staticmethod
    def detect_transition(
        previous_occupied: bool,
        current_occupied: bool,
        zone_name: str) -> tuple[str, str] | None:
        if (
            not previous_occupied
            and current_occupied):
            return (EventType.ENTERED, zone_name)

        if (previous_occupied and not current_occupied):
            return (EventType.EXITED, zone_name)

        return None


ADD_NEW_ZONE_OPTION = "ADD NEW ZONE"

@dataclass
class OccupancyResult:
    occupied: bool
    occupancy_pixels: int
    occupancy_ratio: float


@dataclass
class ZoneRuntimeState:
    occupied: bool = False
    occupancy_pixels: int = 0
    occupancy_ratio: float = 0


@dataclass(frozen=True)
class ZoneDetectionResult:
    zone_name: str
    occupied: bool
    occupancy_pixels: int
    occupancy_ratio: float


@dataclass(frozen=True)
class DetectionSnapshot:
    raw_frame: np.ndarray
    motion_mask: np.ndarray | None
    zone_results: list["ZoneDetectionResult"]


@dataclass(frozen=True)
class ZonePresentation:
    name: str
    points_np: np.ndarray
    occupied: bool
    occupancy_ratio: float


@dataclass(frozen=True)
class PresentationFrame:
    raw_frame: np.ndarray
    motion_mask: np.ndarray | None
    zones: list[ZonePresentation]


@dataclass(frozen=True)
class RawFrame:
    frame: np.ndarray


@dataclass(frozen=True)
class PreprocessedFrame:
    raw_frame: np.ndarray
    gray_frame: np.ndarray


@dataclass(frozen=True)
class MotionFrame:
    raw_frame: np.ndarray
    gray_frame: np.ndarray
    motion_mask: np.ndarray


class OccupancyAnalyzer:

    def __init__(
        self,
        detection_config: DetectionConfig
    ):

        self.detection_config = detection_config

    def analyze_zone(
        self,
        zone_mask: np.ndarray,
        total_zone_pixels: int
    ) -> OccupancyResult:

        occupancy_pixels = cv2.countNonZero(
            zone_mask
        )

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
            occupancy_ratio >
            self.detection_config.occupancy_percent
        )

        return OccupancyResult(
            occupied=occupied,
            occupancy_pixels=occupancy_pixels,
            occupancy_ratio=occupancy_ratio
        )


@dataclass(frozen=True)
class EngineDetectionResult:
    snapshot: DetectionSnapshot
    events: list[tuple[str, str]]


class MotionDetector:

    def __init__(
        self,
        detection_config: DetectionConfig
    ):

        self.detection_config = detection_config

        self.morph_kernel = np.ones(
            (
                detection_config.kernel_size,
                detection_config.kernel_size
            ),
            np.uint8
        )

    def create_difference_mask(
        self,
        background_frame: np.ndarray,
        gray_frame: np.ndarray
    ) -> np.ndarray:

        diff = cv2.absdiff(
            background_frame,
            gray_frame
        )

        _, binary_mask = cv2.threshold(
            diff,
            self.detection_config.diff_threshold,
            255,
            cv2.THRESH_BINARY
        )

        return binary_mask

    def remove_noise(
        self,
        motion_mask: np.ndarray
    ) -> np.ndarray:

        clean_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            self.morph_kernel
        )

        return clean_mask

    def detect_motion(
        self,
        frame: PreprocessedFrame,
        background_frame: np.ndarray | None
    ) -> MotionFrame | None:

        if background_frame is None:
            return None

        motion_mask = self.create_difference_mask(
            background_frame,
            frame.gray_frame
        )

        if self.detection_config.use_cleanup:

            motion_mask = self.remove_noise(
                motion_mask
            )

        return MotionFrame(
            raw_frame=frame.raw_frame,
            gray_frame=frame.gray_frame,
            motion_mask=motion_mask
        )


class LayoutPersistence:

    def __init__(
        self,
        geometry_builder: "ZoneGeometryBuilder"
    ):

        self.geometry_builder = geometry_builder

    def save_layout(
        self,
        file_path: str,
        zones: list["Zone"]
    ):

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

            json.dump(
                layout_data,
                file,
                indent=4
            )

    def load_layout(
        self,
        file_path: str
    ) -> list["Zone"]:

        with open(file_path, "r") as file:

            layout_data = json.load(file)

        if "zones" not in layout_data:

            raise LayoutValidationError(
                LayoutErrorCode.INVALID_LAYOUT_FILE
            )

        if not isinstance(
            layout_data["zones"],
            list
        ):

            raise LayoutValidationError(
                LayoutErrorCode.ZONES_AS_LIST
            )

        zones = []

        for zone_data in layout_data["zones"]:

            if (
                "name" not in zone_data
                or "points" not in zone_data
            ):

                raise LayoutValidationError(
                    LayoutErrorCode.INVALID_ZONE_FORMAT
                )

            zone = Zone(
                zone_data["name"],
                zone_data["points"],
                self.geometry_builder
            )

            zones.append(zone)

        return zones


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


class ZoneRepository:

    def __init__(
        self,
        geometry_builder: "ZoneGeometryBuilder"
    ):

        self.geometry_builder = geometry_builder

        self.zones: list["Zone"] = []

    def add_zone(
        self,
        name: str,
        points: list[list[int]]
    ) -> None:

        zone = Zone(
            name,
            points,
            self.geometry_builder
        )

        self.zones.append(zone)

    def update_zone(
        self,
        index: int,
        name: str,
        points: list[list[int]]
    ) -> tuple[str, str]:

        zone = self.zones[index]

        old_name = zone.name

        zone.name = name

        zone.update_points(points)

        return old_name, name

    def delete_zone(
        self,
        index: int
    ) -> str:

        zone = self.zones[index]

        deleted_name = zone.name

        del self.zones[index]

        return deleted_name

    def set_zones(
        self,
        zones: list["Zone"]
    ) -> None:

        self.zones = zones


class ZoneProcessor:

    def __init__(
        self,
        repository: ZoneRepository,
        occupancy_analyzer: OccupancyAnalyzer,
        runtime_state_manager: "RuntimeStateManager"
    ):

        self.repository = repository

        self.occupancy_analyzer = occupancy_analyzer

        self.runtime_state_manager = runtime_state_manager

    def extract_zone(
        self,
        motion_mask: np.ndarray,
        zone: "Zone"
    ) -> np.ndarray:

        return cv2.bitwise_and(
            motion_mask,
            zone.geometry.polygon_mask
        )

    def process_zone(
        self,
        motion_frame: MotionFrame,
        zone: "Zone"
    ) -> tuple[
        ZoneDetectionResult,
        tuple[str, str] | None
    ]:

        zone_mask = self.extract_zone(
            motion_frame.motion_mask,
            zone
        )

        occupancy_result = (
            self.occupancy_analyzer.analyze_zone(
                zone_mask,
                zone.geometry.total_pixels
            )
        )

        previous_occupied, _ = (
            self.runtime_state_manager
            .update_zone_state(
                zone.name,
                occupancy_result
            )
        )

        event = (
            EventDetector.detect_transition(
                previous_occupied,
                occupancy_result.occupied,
                zone.name
            )
        )

        detection_result = ZoneDetectionResult(
            zone_name=zone.name,
            occupied=occupancy_result.occupied,
            occupancy_pixels=occupancy_result.occupancy_pixels,
            occupancy_ratio=occupancy_result.occupancy_ratio
        )

        return detection_result, event

    def process_all_zones(
        self,
        motion_frame: MotionFrame
    ) -> tuple[
        list[ZoneDetectionResult],
        list[tuple[str, str]]
    ]:

        detection_results = []

        events = []

        for zone in self.repository.zones:

            if not self.runtime_state_manager.detection_enabled:

                detection_result = ZoneDetectionResult(
                    zone_name=zone.name,
                    occupied=False,
                    occupancy_pixels=0,
                    occupancy_ratio=0
                )

                detection_results.append(
                    detection_result
                )

                continue

            detection_result, event = (
                self.process_zone(
                    motion_frame,
                    zone
                )
            )

            detection_results.append(
                detection_result
            )

            if event is not None:

                events.append(event)

        return detection_results, events


class TkImageConverter:

    @staticmethod
    def to_photo_image(
        frame: np.ndarray
    ) -> ImageTk.PhotoImage:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb_frame
        )

        return ImageTk.PhotoImage(
            image=image
        )


class MaskRenderer:

    def render(
        self,
        mask: np.ndarray,
        zones: list["ZonePresentation"]
    ) -> np.ndarray:

        mask_color = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )

        mask_color[mask > 0] = (
            AppConfig.COLOR_WHITE
        )

        for zone in zones:

            if zone.occupied:
                color = AppConfig.COLOR_RED
            else:
                color = AppConfig.COLOR_GREEN

            cv2.polylines(
                mask_color,
                [zone.points_np],
                True,
                color,
                2
            )

            x = zone.points_np[0][0]
            y = zone.points_np[0][1] - 10

            label = (
                f"{zone.name} "
                f"{zone.occupancy_ratio * 100:.1f}%"
            )

            cv2.putText(
                mask_color,
                label,
                (x, y),
                AppConfig.FONT,
                AppConfig.MASK_FONT_SCALE,
                color,
                AppConfig.FONT_THICKNESS
            )

        return mask_color


class CameraOverlayRenderer:

    def render(
        self,
        frame: np.ndarray,
        zones: list["ZonePresentation"]
    ) -> np.ndarray:

        overlay_frame = frame.copy()

        for zone in zones:

            if zone.occupied:

                color = AppConfig.COLOR_RED

            else:

                color = AppConfig.COLOR_GREEN

            cv2.polylines(
                overlay_frame,
                [zone.points_np],
                True,
                color,
                2
            )

            x = zone.points_np[0][0]
            y = zone.points_np[0][1] - 10

            label = (
                f"{zone.name} "
                f"{zone.occupancy_ratio * 100:.1f}%"
            )

            cv2.putText(
                overlay_frame,
                label,
                (x, y),
                AppConfig.FONT,
                AppConfig.FONT_SCALE,
                color,
                AppConfig.FONT_THICKNESS
            )

        return overlay_frame


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


class FrameStore:
    def __init__(self):
        self.latest_frame: np.ndarray | None = None


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
        self.set_zone_name("")
        for i in range(4):
            self.clear_point(i)
        self.disable_delete_button()

    def load_zone(self, zone):
        self.enable_delete_button()
        self.set_zone_name(zone.name)
        for i in range(4):
            x = zone.points[i][0]
            y = zone.points[i][1]
            self.set_point_text(
                i,
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


class EditorState:

    def __init__(self):
        self.active_point_index = None


class RuntimeStateManager:

    def __init__(self):

        self.detection_enabled = False

        self.zone_runtime_states: dict[str, ZoneRuntimeState] = {}

        self.background_frame: np.ndarray | None = None

        self.zone_runtime_states: dict[str, ZoneRuntimeState] = {}

        self.background_frame: np.ndarray | None = None

    def update_zone_state(
        self,
        zone_name: str,
        result: OccupancyResult
    ) -> tuple[bool, ZoneRuntimeState]:

        self.ensure_zone_state_exists(
            zone_name
        )

        state = self.zone_runtime_states[
            zone_name
        ]

        previous_occupied = state.occupied

        state.occupied = result.occupied
        state.occupancy_pixels = (
            result.occupancy_pixels
        )
        state.occupancy_ratio = (
            result.occupancy_ratio
        )

        return previous_occupied, state

    def ensure_zone_state_exists(
        self,
        zone_name: str
    ) -> None:

        if zone_name not in self.zone_runtime_states:

            self.zone_runtime_states[zone_name] = (
                ZoneRuntimeState()
            )

    def remove_zone_state(
        self,
        zone_name: str
    ) -> None:

        if zone_name in self.zone_runtime_states:

            del self.zone_runtime_states[zone_name]

    def rename_zone_state(
        self,
        old_name: str,
        new_name: str
    ) -> None:

        runtime_state = (
            self.zone_runtime_states.pop(
                old_name,
                ZoneRuntimeState()
            )
        )

        self.zone_runtime_states[new_name] = runtime_state

    def rebuild_runtime_states(
        self,
        zones: list["Zone"]
    ) -> None:

        self.zone_runtime_states.clear()

        for zone in zones:

            self.zone_runtime_states[zone.name] = (
                ZoneRuntimeState()
            )

    def reset_all_zone_states(self) -> None:

        for runtime_state in (
            self.zone_runtime_states.values()
        ):

            runtime_state.occupied = False
            runtime_state.occupancy_pixels = 0
            runtime_state.occupancy_ratio = 0

    def enable_detection(self) -> None:

        self.detection_enabled = True

    def disable_detection(self) -> None:

        self.detection_enabled = False

        self.reset_all_zone_states()


class ZoneEditorController:

    def __init__(
        self,
        zone_repository: ZoneRepository,
        editor_panel: ZoneEditorPanel,
        log_panel: LogPanel,
        camera_panel: CameraPanel,
        editor_state: EditorState,
        coordinate_parser: "CoordinateParser",
        runtime_state_manager: RuntimeStateManager):

        self.zone_repository = zone_repository
        self.editor_panel = editor_panel
        self.log_panel = log_panel
        self.camera_panel = camera_panel
        self.editor_state = editor_state
        self.coordinate_parser = coordinate_parser
        self.runtime_state_manager = (
            runtime_state_manager
        )

    # SELECTED TEXTBOX
    def select_point_entry(self, index: int) -> None:
        self.editor_state.active_point_index = index

    # GET CAMERA CLICK
    def on_camera_click(self, event: tk.Event) -> None:

        if self.camera_panel.camera_label is None:
            return

        x = int(
            event.x
            * AppConfig.FRAME_WIDTH
            / AppConfig.DISPLAY_WIDTH
        )

        y = int(
            event.y
            * AppConfig.FRAME_HEIGHT
            / AppConfig.DISPLAY_HEIGHT
        )

        self.log_panel.add_log(
            f"Mouse click: x={x}, y={y}"
        )

        active_index = (
            self.editor_state.active_point_index
        )

        if active_index is not None:

            self.editor_panel.set_point_text(
                active_index,
                f"{x},{y}"
            )

    # LOAD ZONE COORDINATES IN GUI
    def load_zone_into_editor(
        self,
        event: tk.Event | None = None):

        selected_index = (
            self.editor_panel.zone_selector.current()
        )

        # ADD NEW ZONE selected
        if selected_index >= len(
            self.zone_repository.zones
        ):

            self.editor_state.active_point_index = None

            self.editor_panel.clear_editor()

            return

        zone = (
            self.zone_repository.zones[selected_index]
        )

        if len(zone.points) != 4:

            self.log_panel.add_log(
                f"Zone '{zone.name}' has invalid point count"
            )

            return

        self.editor_panel.load_zone(zone)

    # DELETE SELECTED ZONE
    def delete_selected_zone(self):

        selected_index = (
            self.editor_panel.zone_selector.current()
        )

        # Prevent deleting ADD NEW ZONE option
        if selected_index >= len(
            self.zone_repository.zones
        ):
            return

        deleted_name = (
            self.zone_repository
            .zones[selected_index]
            .name
        )

        self.zone_repository.delete_zone(
            selected_index
        )

        self.runtime_state_manager.remove_zone_state(
            deleted_name
        )

        # Rebuild combobox
        self.refresh_zone_selector()

        self.log_panel.add_log(
            f"{deleted_name} {Messages.ZONE_DELETED_SUFFIX}"
        )

    # REFRESH ZONE SELECTOR
    def refresh_zone_selector(
        self,
        selectIndex=0):

        zone_names = []

        for zone in self.zone_repository.zones:

            zone_names.append(zone.name)

        zone_names.append(
            ADD_NEW_ZONE_OPTION
        )

        self.editor_panel.zone_selector["values"] = (
            zone_names
        )

        self.editor_panel.zone_selector.current(
            selectIndex
        )

        self.load_zone_into_editor()

    def get_points_from_editor(self) -> list[list[int]] | None:

        new_points = []

        for text_value in (
            self.editor_panel.get_point_texts()
        ):

            try:

                point = (
                    self.coordinate_parser.parse_point(
                        text_value
                    )
                )

                new_points.append(point)

            except ValidationError as error:

                self.log_panel.add_log(
                    self.get_validation_message(error)
                )

                return None

        return new_points

    def validate_zone_name(
        self,
        zone_name) -> None:

        if zone_name.strip() == "":

            raise ValidationError(
                ValidationErrorCode.EMPTY_ZONE_NAME
            )
    
    # APPLY ZONE COORDINATES CHANGES
    def apply_zone_changes(self) -> None:

        selected_index = (
            self.editor_panel.zone_selector.current()
        )

        new_zone_name = (
            self.editor_panel.get_zone_name()
        )

        new_points = (
            self.get_points_from_editor()
        )

        if new_points is None:
            return

        # Require exactly 4 valid points
        if len(new_points) != 4:

            self.log_panel.add_log(
                Messages.INVALID_ZONE_POINTS
            )

            return

        # Prevent empty zone names
        try:

            self.validate_zone_name(
                new_zone_name
            )

        except ValidationError as error:

            self.log_panel.add_log(
                self.get_validation_message(error)
            )

            return

        # ============================================
        # ADD NEW ZONE
        # ============================================

        if selected_index >= len(
            self.zone_repository.zones
        ):

            self.zone_repository.add_zone(
                new_zone_name,
                new_points
            )

            self.runtime_state_manager.ensure_zone_state_exists(
                new_zone_name
            )

            self.log_panel.add_log(
                f"{new_zone_name} {Messages.ZONE_ADDED_SUFFIX}"
            )

        # ============================================
        # UPDATE EXISTING ZONE
        # ============================================

        else:

            old_name = (
                self.zone_repository
                .zones[selected_index]
                .name
            )

            self.zone_repository.update_zone(
                selected_index,
                new_zone_name,
                new_points
            )

            # Handle rename
            if old_name != new_zone_name:

                self.runtime_state_manager.rename_zone_state(
                    old_name,
                    new_zone_name
                )

            self.log_panel.add_log(
                f"{new_zone_name} {Messages.ZONE_UPDATED_SUFFIX}"
            )

        if selected_index >= len(
            self.zone_repository.zones
        ) - 1:

            self.refresh_zone_selector(
                len(self.zone_repository.zones) - 1
            )

        else:

            self.refresh_zone_selector(
                selected_index
            )

    def get_validation_message(
        self,
        error: ValidationError) -> str:

        if (
            error.code
            == ValidationErrorCode.INVALID_COORDINATE_FORMAT
        ):
            return Messages.INVALID_COORDINATE_FORMAT

        if (
            error.code
            == ValidationErrorCode.COORDINATE_OUT_OF_BOUNDS
        ):
            return "Coordinate out of bounds"

        if (
            error.code
            == ValidationErrorCode.EMPTY_ZONE_NAME
        ):
            return Messages.EMPTY_ZONE_NAME

        if (
            error.code
            == ValidationErrorCode.INVALID_ZONE_POINTS
        ):
            return Messages.INVALID_ZONE_POINTS

        return "Validation error"


class EventBus:

    def __init__(self):

        self.subscribers = {}

    def subscribe(
        self,
        event_type: str,
        callback
    ) -> None:

        self.subscribers.setdefault(
            event_type,
            []
        ).append(callback)

    def publish(
        self,
        event_type: str,
        data=None
    ) -> None:

        for callback in self.subscribers.get(event_type, []):

            callback(data)


class DetectionController:

    def __init__(
        self,
        engine: "RailwayDetectionEngine",
        control_panel: "ControlPanel",
        log_panel: LogPanel,
        frame_store: FrameStore):

        self.engine = engine

        self.control_panel = control_panel

        self.log_panel = log_panel

        self.frame_store = frame_store

    def capture_background(self):

        success = (
            self.engine
            .capture_background_reference(
                self.frame_store.latest_frame
            )
        )

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

        self.engine.enable_detection()

        self.control_panel.disable_arm_button()

        self.control_panel.enable_stop_button()

        self.log_panel.add_log(
            Messages.DETECTION_ARMED
        )


class PreviewStateController:

    def __init__(
        self,
        runtime_state_manager: RuntimeStateManager,
        preview_controller: "PreviewPresenter",
        control_panel: "ControlPanel",
        log_panel: LogPanel):

        self.runtime_state_manager = (
            runtime_state_manager
        )

        self.preview_controller = (
            preview_controller
        )

        self.control_panel = control_panel

        self.log_panel = log_panel

        self.preview_enabled = True

    def toggle_preview(self):

        if self.preview_enabled:

            self.preview_enabled = False

            self.preview_controller.stop_preview()

            self.control_panel.stop_button.config(
                text="Resume Preview"
            )

            self.log_panel.add_log(
                Messages.PREVIEW_DESTROYED
            )

        else:

            self.preview_enabled = True

            self.control_panel.stop_button.config(
                text="Stop Preview"
            )

            self.log_panel.add_log(
                "Preview resumed"
            )

    def is_preview_enabled(self) -> bool:
        return self.preview_enabled


class CoordinateParser:

    def __init__(
        self,
        frame_geometry: FrameGeometry
    ):

        self.frame_geometry = frame_geometry

    def parse_point(
        self,
        text_value: str
    ) -> list[int]:

        split_values = text_value.split(",")

        if len(split_values) != 2:

            raise ValidationError(
                ValidationErrorCode.INVALID_COORDINATE_FORMAT
            )

        x = int(split_values[0])
        y = int(split_values[1])

        if (
            x < 0
            or x >= self.frame_geometry.width
        ):

            raise ValidationError(
                ValidationErrorCode.COORDINATE_OUT_OF_BOUNDS
            )

        if (
            y < 0
            or y >= self.frame_geometry.height
        ):

            raise ValidationError(
                ValidationErrorCode.COORDINATE_OUT_OF_BOUNDS
            )

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


class FramePreprocessor:

    def preprocess(
        self,
        frame: RawFrame
    ) -> PreprocessedFrame:

        gray_frame = cv2.cvtColor(
            frame.frame,
            cv2.COLOR_BGR2GRAY
        )

        return PreprocessedFrame(
            raw_frame=frame.frame,
            gray_frame=gray_frame
        )


class RailwayDetectionEngine:

    def __init__(
        self,
        frame_geometry: FrameGeometry,
        detection_config: DetectionConfig
    ):

        self.preprocessor = (
            FramePreprocessor()
        )

        self.motion_detector = (
            MotionDetector(
                detection_config
            )
        )

        self.occupancy_analyzer = (
            OccupancyAnalyzer(
                detection_config
            )
        )

        self.runtime_state_manager = (
            RuntimeStateManager()
        )

        self.geometry_builder = (
            ZoneGeometryBuilder(
                frame_geometry
            )
        )

        self.zone_repository = (
            ZoneRepository(
                self.geometry_builder
            )
        )

        self.zone_processor = (
            ZoneProcessor(
                self.zone_repository,
                self.occupancy_analyzer,
                self.runtime_state_manager
            )
        )

    def process_detection(
        self,
        frame: np.ndarray
    ) -> EngineDetectionResult:

        raw_frame = RawFrame(frame)

        preprocessed_frame = (
            self.preprocessor.preprocess(
                raw_frame
            )
        )

        motion_frame = (
            self.motion_detector.detect_motion(
                preprocessed_frame,
                self.runtime_state_manager.background_frame
            )
        )

        if motion_frame is None:

            empty_results = []

            for zone in self.zone_repository.zones:

                empty_results.append(
                    ZoneDetectionResult(
                        zone_name=zone.name,
                        occupied=False,
                        occupancy_pixels=0,
                        occupancy_ratio=0
                    )
                )

            snapshot = DetectionSnapshot(
                raw_frame=frame,
                motion_mask=None,
                zone_results=empty_results
            )

            return EngineDetectionResult(
                snapshot=snapshot,
                events=[]
            )

        zone_results, events = (
            self.zone_processor.process_all_zones(
                motion_frame
            )
        )

        snapshot = DetectionSnapshot(
            raw_frame=motion_frame.raw_frame,
            motion_mask=motion_frame.motion_mask,
            zone_results=zone_results
        )

        return EngineDetectionResult(
            snapshot=snapshot,
            events=events
        )

    def capture_background_reference(
        self,
        frame: np.ndarray | None
    ) -> bool:

        if frame is None:
            return False

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.runtime_state_manager.background_frame = (
            gray_frame.copy()
        )

        return True

    def enable_detection(self) -> None:

        self.runtime_state_manager.enable_detection()

    def disable_detection(self) -> None:

        self.runtime_state_manager.disable_detection()


class PreviewPresenter:

    def __init__(
        self,
        renderer: CameraOverlayRenderer,
        camera_panel: CameraPanel,
        mask_renderer: MaskRenderer,
        runtime_state_manager: RuntimeStateManager,
        preview_state_controller: PreviewStateController):

        self.renderer = renderer

        self.mask_renderer = mask_renderer

        self.camera_panel = camera_panel

        self.runtime_state_manager = (
            runtime_state_manager
        )

        self.preview_state_controller = (
            preview_state_controller
        )

    def update_previews(
        self,
        presentation_frame: PresentationFrame
    ) -> None:

        if (
            not self.preview_state_controller
            .is_preview_enabled()
        ):
            return

        overlay_frame = self.renderer.render(
            presentation_frame.raw_frame,
            presentation_frame.zones
        )

        overlay_frame = cv2.resize(
            overlay_frame,
            (
                AppConfig.DISPLAY_WIDTH,
                AppConfig.DISPLAY_HEIGHT
            )
        )

        photo = (
            TkImageConverter.to_photo_image(
                overlay_frame
            )
        )

        self.camera_panel.camera_label.config(
            image=photo
        )

        self.camera_panel.camera_label.image = (
            photo
        )

        if (
            self.runtime_state_manager
            .detection_enabled
            and presentation_frame.motion_mask
            is not None
        ):

            mask_frame = self.mask_renderer.render(
                presentation_frame.motion_mask,
                presentation_frame.zones
            )

            mask_frame = cv2.resize(
                mask_frame,
                (
                    AppConfig.DISPLAY_WIDTH,
                    AppConfig.DISPLAY_HEIGHT
                )
            )

            mask_photo = (
                TkImageConverter.to_photo_image(
                    mask_frame
                )
            )

            self.camera_panel.mask_label.config(
                image=mask_photo
            )

            self.camera_panel.mask_label.image = (
                mask_photo
            )

    def stop_preview(self) -> None:

        if self.camera_panel.camera_label is not None:

            self.camera_panel.camera_label.config(
                image=""
            )

            self.camera_panel.camera_label.image = None

            self.camera_panel.camera_label.update_idletasks()

        if self.camera_panel.mask_label is not None:

            self.camera_panel.mask_label.config(
                image=""
            )

            self.camera_panel.mask_label.image = None

            self.camera_panel.mask_label.update_idletasks()


class PresentationBuilder:

    def __init__(
        self,
        zone_repository: ZoneRepository
    ):

        self.zone_repository = zone_repository

    def build(
        self,
        snapshot: DetectionSnapshot
    ) -> PresentationFrame:

        presentation_zones = []

        detection_map = {}

        for result in snapshot.zone_results:

            detection_map[
                result.zone_name
            ] = result

        for zone in self.zone_repository.zones:

            detection_result = detection_map.get(
                zone.name
            )

            occupied = False
            occupancy_ratio = 0

            if detection_result is not None:

                occupied = (
                    detection_result.occupied
                )

                occupancy_ratio = (
                    detection_result.occupancy_ratio
                )

            presentation_zone = ZonePresentation(
                name=zone.name,
                points_np=zone.geometry.points_np,
                occupied=occupied,
                occupancy_ratio=occupancy_ratio
            )

            presentation_zones.append(
                presentation_zone
            )

        return PresentationFrame(
            raw_frame=snapshot.raw_frame,
            motion_mask=snapshot.motion_mask,
            zones=presentation_zones
        )


class PipelineCoordinator:

    def __init__(
        self,
        engine: RailwayDetectionEngine,
        presentation_builder: PresentationBuilder,
        event_bus: EventBus
    ):

        self.engine = engine

        self.presentation_builder = (
            presentation_builder
        )

        self.event_bus = event_bus

    def process_frame(
        self,
        frame: np.ndarray
    ) -> None:

        engine_result = (
            self.engine.process_detection(
                frame
            )
        )

        presentation_frame = (
            self.presentation_builder.build(
                engine_result.snapshot
            )
        )

        self.event_bus.publish(
            "presentation_frame",
            presentation_frame
        )

        for event_type, zone_name in (
            engine_result.events
        ):

            message = (
                EventFormatter.format(
                    event_type,
                    zone_name
                )
            )

            self.event_bus.publish(
                "log",
                message
            )


class ApplicationLoop:

    def __init__(
        self,
        root: tk.Tk,
        camera_service: CameraService,
        pipeline_coordinator: PipelineCoordinator,
        frame_store: FrameStore,
        event_bus: EventBus):

        self.root = root

        self.camera_service = camera_service

        self.pipeline_coordinator = (
            pipeline_coordinator
        )

        self.frame_store = frame_store

        self.event_bus = event_bus

        self.running = True

        self.camera_failure_logged = False

    def run_frame_loop(self) -> None:

        if not self.running:
            return

        frame = self.capture_frame()

        if frame is not None:

            self.pipeline_coordinator.process_frame(
                frame
            )

            self.frame_store.latest_frame = frame

        self.schedule_next_update()

    def capture_frame(self) -> np.ndarray | None:

        try:

            return (
                self.camera_service.capture_frame()
            )

        except Exception:

            if not self.camera_failure_logged:

                self.event_bus.publish(
                    "log",
                    Messages.PREVIEW_UPDATE_FAILED
                )

                self.event_bus.publish(
                    "log",
                    traceback.format_exc()
                )

                self.camera_failure_logged = True

            return None

    def schedule_next_update(self) -> None:

        self.root.after(
            AppConfig.GUI_UPDATE_DELAY,
            self.run_frame_loop
        )

    def quit_program(self):

        self.running = False

        self.event_bus.publish(
            "log",
            Messages.EXITING
        )

        try:

            self.camera_service.stop()

        except Exception:

            self.event_bus.publish(
                "log",
                traceback.format_exc()
            )

        finally:

            self.root.quit()

            self.root.destroy()


class RailwayDetectorApp:

    def __init__(self):
        # CONFIGURATION
        self.frame_geometry = FrameGeometry(
            width=AppConfig.FRAME_WIDTH,
            height=AppConfig.FRAME_HEIGHT
        )

        self.detection_config = DetectionConfig(
            occupancy_percent=AppConfig.OCCUPANCY_PERCENT,
            diff_threshold=AppConfig.DIFF_THRESHOLD,
            use_cleanup=AppConfig.USE_CLEANUP,
            kernel_size=AppConfig.KERNEL_SIZE
        )

        # BASIC STATE
        self.frame_store = FrameStore()

        self.editor_state = EditorState()

        self.event_bus = EventBus()

        # ENGINE
        self.engine = RailwayDetectionEngine(
            self.frame_geometry,
            self.detection_config
        )

        # PANELS
        self.log_panel = LogPanel()

        self.control_panel = ControlPanel()

        self.camera_panel = CameraPanel()

        self.zone_editor_panel = ZoneEditorPanel()

        # LOG EVENTS
        self.event_bus.subscribe(
            "log",
            self.log_panel.add_log
        )

        # HELPERS
        self.layout_persistence = LayoutPersistence(
            self.engine.geometry_builder
        )

        self.coordinate_parser = CoordinateParser(
            self.frame_geometry
        )

        # CONTROLLERS
        self.zone_editor_controller = ZoneEditorController(
            self.engine.zone_repository,
            self.zone_editor_panel,
            self.log_panel,
            self.camera_panel,
            self.editor_state,
            self.coordinate_parser,
            self.engine.runtime_state_manager
        )

        self.detection_controller = DetectionController(
            self.engine,
            self.control_panel,
            self.log_panel,
            self.frame_store
        )

        # RENDERERS
        self.camera_overlay_renderer = (
            CameraOverlayRenderer()
        )

        self.mask_renderer = (
            MaskRenderer()
        )

        # PREVIEW STATE CONTROLLER
        self.preview_state_controller = (
            PreviewStateController(
                self.engine.runtime_state_manager,
                None,
                self.control_panel,
                self.log_panel
            )
        )

        # PREVIEW PRESENTER
        self.preview_controller = PreviewPresenter(
            self.camera_overlay_renderer,
            self.camera_panel,
            self.mask_renderer,
            self.engine.runtime_state_manager,
            self.preview_state_controller
        )

        # Inject presenter back into state controller
        self.preview_state_controller.preview_controller = (
            self.preview_controller
        )

        # PRESENTATION PIPELINE
        self.presentation_builder = (
            PresentationBuilder(
                self.engine.zone_repository
            )
        )

        self.pipeline_coordinator = (
            PipelineCoordinator(
                self.engine,
                self.presentation_builder,
                self.event_bus
            )
        )

        # PREVIEW EVENTS
        self.event_bus.subscribe(
            "presentation_frame",
            self.preview_controller.update_previews
        )

        # CAMERA
        self.camera_service = CameraService()

        self.camera_service.start()

        # GUI
        self.root = self.create_tkinter_gui()

        # APPLICATION LOOP
        self.application_loop = ApplicationLoop(
            self.root,
            self.camera_service,
            self.pipeline_coordinator,
            self.frame_store,
            self.event_bus
        )

        # EVENT BINDINGS
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

        # BUILD UI
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

        # CONTROL BUTTONS
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

        self.control_panel.stop_button.config(
            command=self.preview_state_controller.toggle_preview
        )

        # ZONE EDITOR
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

        # POINT ENTRY FOCUS
        for index, entry in enumerate(
            self.zone_editor_panel.point_entries
        ):

            entry.bind(
                "<FocusIn>",
                lambda event, i=index:
                self.zone_editor_controller.select_point_entry(i)
            )

        # CAMERA CLICK
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

        self.layout_persistence.save_layout(
            file_path,
            self.engine.zone_repository.zones
        )

        self.log_panel.add_log(
            Messages.LAYOUT_SAVED
        )

    def load_layout_gui(self) -> None:

        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        try:

            zones = self.layout_persistence.load_layout(
                file_path
            )

            self.engine.zone_repository.set_zones(
                zones
            )

            self.engine.runtime_state_manager.rebuild_runtime_states(
                zones
            )

        except json.JSONDecodeError:

            self.log_panel.add_log(
                Messages.INVALID_JSON_FILE
            )

            return

        except ValueError as error:

            self.log_panel.add_log(
                str(error)
            )

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

    def __init__(
        self,
        frame_geometry: FrameGeometry
    ):

        self.frame_geometry = frame_geometry

    def build(
        self,
        points: list[list[int]]
    ) -> ZoneGeometry:

        points_np = np.array(
            points,
            dtype=np.int32
        )

        polygon_mask = np.zeros(
            (
                self.frame_geometry.height,
                self.frame_geometry.width
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
    geometry_builder: "ZoneGeometryBuilder"

    def __post_init__(self):

        self.rebuild_polygon()

    def update_points(
        self,
        new_points: list[list[int]]
    ) -> None:

        self.points = new_points

        self.rebuild_polygon()

    def rebuild_polygon(self) -> None:

        self.geometry = (
            self.geometry_builder.build(
                self.points
            )
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

        widget.bind("<Enter>", self.schedule_tooltip, add="+")
        widget.bind("<Leave>", self.hide_tooltip, add="+")

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
