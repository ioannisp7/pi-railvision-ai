# Full re-factoring of the code limiting the number of classes
from picamera2 import Picamera2
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import json
from PIL import Image, ImageTk
import traceback
from dataclasses import dataclass

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
    MORPH_KERNEL = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    
    # Colors (BGR format for OpenCV)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (30, 255, 0)
    COLOR_RED = (0, 0, 255)
        
class Messages:
    ADD_NEW_ZONE_OPTION = "Add new zone"
    LAYOUT_LOADED = "Layout loaded"
    LAYOUT_SAVED = "Layout saved"
    PREVIEW_STOPPED = "Preview stopped"
    PREVIEW_RESUMED = "Preview resumed"
    ZONE_ADDED_SUFFIX = "zone created"
    ZONE_UPDATED_SUFFIX = "zone updated"
    ZONE_DELETED_SUFFIX = "zone deleted"
    BACKGROUND_CAPTURED = "Background reference captured"
    DETECTION_ARMED = "Occupancy detection armed"
    INVALID_JSON_FILE = "Selected JSON file is invalid"
    EXITING = "Exiting"

    #Error messages
    POINTS_AS_LIST = "POINTS_AS_LIST"
    DUPLICATE_ZONE_NAME = "Zone name already exists"
    INVALID_COORDINATE_FORMAT = "Invalid coordinate format"
    COORDINATE_OUT_OF_BOUNDS = "COORDINATE_OUT_OF_BOUNDS"
    EMPTY_ZONE_NAME = "Zone name cannot be empty"
    INVALID_ZONE_POINTS = "Zone requires 4 valid points"
    INVALID_LAYOUT_FILE = "INVALID_LAYOUT_FILE"
    ZONES_AS_LIST = "ZONES_AS_LIST"
    INVALID_ZONE_FORMAT = "INVALID_ZONE_FORMAT"
    ZONE_CONTAIN_4_POINTS = "ZONE_CONTAIN_4_POINTS"
    POINT_AS_LIST = "POINT_AS_LIST"
    ZONE_CONTAIN_XY = "ZONE_CONTAIN_XY"
    POINTS_AS_INTEGERS = "POINTS_AS_INTEGERS"
    CAMERA_CAPTURE_FAILED = "Camera frame capture failed"


class MotionDetectionPipeline:
    def process(
        self,
        frame: np.ndarray,
        background_frame: np.ndarray | None):

        if background_frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(background_frame, gray)

        _, motion_mask = cv2.threshold(
            diff,
            AppConfig.DIFF_THRESHOLD,
            255,
            cv2.THRESH_BINARY)

        if AppConfig.USE_CLEANUP:

            motion_mask = cv2.morphologyEx(
                motion_mask,
                cv2.MORPH_OPEN,
                AppConfig.MORPH_KERNEL)

        return motion_mask


class ZoneRepository:
    def __init__(
        self,
        frame_width,
        frame_height):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.zones: list["Zone"] = []

    def add_zone(
        self,
        name: str,
        points: list[list[int]]) -> None:

        zone = Zone(
            name,
            points,
            self.frame_width,
            self.frame_height)
        self.zones.append(zone)

    def update_zone(
        self,
        index: int,
        name: str,
        points: list[list[int]]) -> None:

        zone = self.zones[index]

        zone.name = name

        zone.update_points(points)

    def delete_zone(self, index: int) -> None:

        zone = self.zones[index]
        deleted_name = zone.name
        del self.zones[index]

    def save_layout(self, file_path: str) -> None:

        zones_to_save = []

        for zone in self.zones:
            zones_to_save.append({
                "name": zone.name,
                "points": zone.points})

        layout_data = {"zones": zones_to_save}

        with open(file_path, "w") as file:
            json.dump(
                layout_data,
                file,
                indent=4)

    def load_layout(self, file_path: str) -> None:
        with open(file_path, "r") as file:
            layout_data = json.load(file)

        if "zones" not in layout_data:
            raise ValueError(Messages.INVALID_LAYOUT_FILE)

        if not isinstance(layout_data["zones"], list):

            raise ValueError(Messages.ZONES_AS_LIST)

        zones = []

        for zone_data in layout_data["zones"]:
            if not isinstance(zone_data, dict):

                raise ValueError(Messages.INVALID_ZONE_FORMAT)

            if ("name" not in zone_data or "points" not in zone_data):
                raise ValueError(Messages.INVALID_ZONE_FORMAT)

            points = zone_data["points"]

            if not isinstance(points, list):
                raise ValueError(Messages.POINTS_AS_LIST)

            if len(points) != 4:
                raise ValueError(Messages.ZONE_CONTAIN_4_POINTS)

            for point in points:
                if not isinstance(point, list):
                    raise ValueError(Messages.POINT_AS_LIST)

                if len(point) != 2:
                    raise ValueError(Messages.ZONE_CONTAIN_XY)

                if (not isinstance(point[0], int) or not isinstance(point[1], int)):
                    raise ValueError(Messages.POINTS_AS_INTEGERS)

            zone = Zone(
                zone_data["name"],
                points,
                self.frame_width,
                self.frame_height
            )

            zones.append(zone)

        self.zones = zones


class LogPanel:
    def __init__(self):
        self.log_text = None

    # CREATE LOG AREA
    def create_log_area(self, right_frame):

        tk.Label(
            right_frame,
            text="System Log"
        ).pack()

        # Container frame
        logFrame = tk.Frame(right_frame)

        logFrame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
        )

        # Scrollbar
        scrollbar = tk.Scrollbar(logFrame)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget
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

        self.log_text.config(state=tk.DISABLED)

        # Connect scrollbar to text widget
        scrollbar.config(command=self.log_text.yview)

    # LOGGING SYSTEM
    def add_log(self, message: str) -> None:
        self.log_text.config(state=tk.NORMAL)

        self.log_text.insert(tk.END,message + "\n")

        # Limit log size
        max_lines = AppConfig.MAX_LOG_LINES

        current_lines = int(self.log_text.index('end-1c').split('.')[0])

        if current_lines > max_lines:
            lines_to_delete = current_lines - max_lines
            self.log_text.delete("1.0", f"{lines_to_delete}.0")

        # Auto-scroll to latest message
        self.log_text.see(tk.END)

        self.log_text.config(state=tk.DISABLED)


class ZoneEditorController:
    def __init__(
        self,
        engine,
        ui,
        log_panel,
        app,
        frame_width,
        frame_height):

        self.engine = engine
        self.zone_repository = engine.zone_repository
        self.ui = ui
        self.log_panel = log_panel
        self.app = app

        self.frame_width = frame_width
        self.frame_height = frame_height

    def parse_point(self, text_value: str) -> list[int]:

        split_values = text_value.split(",")

        if len(split_values) != 2:
            raise ValueError( Messages.INVALID_COORDINATE_FORMAT)

        try:
            x = int(split_values[0])
            y = int(split_values[1])

        except ValueError:
            raise ValueError(Messages.INVALID_COORDINATE_FORMAT)

        if (x < 0 or x >= self.frame_width):
            raise ValueError(Messages.COORDINATE_OUT_OF_BOUNDS)

        if (y < 0 or y >= self.frame_height):
            raise ValueError(Messages.COORDINATE_OUT_OF_BOUNDS)

        return [x, y]

    def select_point_entry(self, index: int) -> None:
        self.app.active_point_index = index

    def on_camera_click(self, event: tk.Event) -> None:
        if self.ui.camera_label is None:
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

        self.log_panel.add_log(f"Mouse click: x={x}, y={y}")

        active_index = (self.app.active_point_index)

        if active_index is not None:
            self.ui.set_point_text(
                active_index,
                f"{x},{y}")

    def load_zone_into_editor(
        self,
        event: tk.Event | None = None):

        selected_index = (self.ui.zone_selector.current())

        if selected_index >= len(self.zone_repository.zones):
            self.app.active_point_index = None
            self.ui.clear_editor()
            return

        zone = (self.zone_repository.zones[selected_index])
        self.ui.load_zone(zone)
        self.app.active_point_index = None

    def delete_selected_zone(self):
        selected_index = (self.ui.zone_selector.current())

        if selected_index >= len(self.zone_repository.zones):
            return

        deleted_name = self.zone_repository.zones[selected_index].name
        self.engine.delete_zone(selected_index)
        self.refresh_zone_selector()
        self.log_panel.add_log(f"{deleted_name} {Messages.ZONE_DELETED_SUFFIX}")

    def refresh_zone_selector(self, selectIndex=0):
        zone_names = [
            zone.name
            for zone in self.zone_repository.zones
        ] + [Messages.ADD_NEW_ZONE_OPTION]

        self.ui.zone_selector["values"] = (zone_names)

        max_index = len(zone_names) - 1

        selectIndex = max(
            0,
            min(selectIndex, max_index))

        self.ui.zone_selector.current(selectIndex)

        self.load_zone_into_editor()

    def get_points_from_editor(self):
        point_texts = (self.ui.get_point_texts())

        if len(point_texts) != 4:
            raise ValueError(Messages.INVALID_ZONE_POINTS)

        new_points = []

        for text_value in point_texts:
            if text_value.strip() == "":
                raise ValueError(Messages.INVALID_ZONE_POINTS)

            point = self.parse_point(text_value)
            new_points.append(point)

        return new_points

    def validate_zone_name(
        self,
        zone_name: str) -> None:

        if zone_name.strip() == "":
            raise ValueError(Messages.EMPTY_ZONE_NAME)

    def validate_zone_name_unique(
        self,
        zone_name: str,
        selected_index: int) -> None:

        for i, zone in enumerate(self.zone_repository.zones):
            if zone.name == zone_name and i != selected_index:
                raise ValueError(Messages.DUPLICATE_ZONE_NAME)

    def apply_zone_changes(self):
        selected_index = (
            self.ui.zone_selector.current())

        new_zone_name = (
            self.ui.get_zone_name())

        is_new_zone = (
            selected_index >= len(
                self.zone_repository.zones
            )
        )

        try:
            new_points = (self.get_points_from_editor())

            self.validate_zone_name(new_zone_name)

            self.validate_zone_name_unique(new_zone_name, selected_index)

        except ValueError as error:
            self.log_panel.add_log(self.get_validation_message(error))
            return

        try:
            if is_new_zone:
                self.engine.add_zone(new_zone_name, new_points)

                self.log_panel.add_log(
                    f"{new_zone_name} {Messages.ZONE_ADDED_SUFFIX}"
                )

            else:
                self.engine.update_zone(
                    selected_index,
                    new_zone_name,
                    new_points
                )

                self.log_panel.add_log(f"{new_zone_name} {Messages.ZONE_UPDATED_SUFFIX}")

        except Exception:
            self.log_panel.add_log(traceback.format_exc())
            return

        new_selection = (
            len(self.zone_repository.zones) - 1
            if is_new_zone
            else selected_index
        )

        self.refresh_zone_selector(new_selection)

    def get_validation_message(self,error: ValueError):

        messages = {
            Messages.INVALID_COORDINATE_FORMAT:
                Messages.INVALID_COORDINATE_FORMAT,

            Messages.COORDINATE_OUT_OF_BOUNDS:
                "Coordinate out of bounds",

            Messages.EMPTY_ZONE_NAME:
                Messages.EMPTY_ZONE_NAME,

            Messages.INVALID_ZONE_POINTS:
                Messages.INVALID_ZONE_POINTS,

            Messages.DUPLICATE_ZONE_NAME:
                Messages.DUPLICATE_ZONE_NAME
        }

        return messages.get(str(error), "Validation error")


class DetectionEngine:

    def __init__(
        self,
        frame_width,
        frame_height):

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zone_repository = ZoneRepository(frame_width,frame_height)
        self.pipeline = MotionDetectionPipeline()
        self.zone_states = {}
        self.background_frame = None
        self.detection_enabled = False

    def create_empty_state(self):
        return {
            "occupied": False,
            "occupancy_pixels": 0,
            "occupancy_ratio": 0
        }

    def ensure_zone_exists(self, zone_name):
        self.zone_states.setdefault(zone_name,self.create_empty_state())

    def remove_zone(self, zone_name):
        if zone_name in self.zone_states:
            del self.zone_states[zone_name]

    def rename_zone(
        self,
        old_name,
        new_name):

        state = self.zone_states.pop(
            old_name,
            self.create_empty_state()
        )

        self.zone_states[new_name] = state

    def rebuild_zone_states(self):
        self.zone_states.clear()
        for zone in self.zone_repository.zones:
            self.ensure_zone_exists(zone.name)

    def update_zone_state(
        self,
        zone_name,
        occupancy_result):

        self.ensure_zone_exists(zone_name)
        state = self.zone_states[zone_name]
        previous = state["occupied"]

        state["occupied"] = occupancy_result["occupied"]
        state["occupancy_pixels"] = occupancy_result["occupancy_pixels"]
        state["occupancy_ratio"] = occupancy_result["occupancy_ratio"]

        return previous


    def enable_detection(self):
        self.detection_enabled = True

    def capture_background_reference(self, frame) -> bool:
        if frame is None:
            return False

        self.background_frame = (
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )
        )

        return True

    def process_detection(self, frame):
        if not self.detection_enabled:
            empty_results = self.build_empty_results()
            return {
                "raw_frame": frame,
                "motion_mask": None,
                "zone_results": empty_results,
                "events": []
            }

        motion_mask = self.pipeline.process(
            frame,
            self.background_frame
        )

        zone_results, events = self.process_all_zones(motion_mask)

        return {
            "raw_frame": frame,
            "motion_mask": motion_mask,
            "zone_results": zone_results,
            "events": events
        }

    def analyze_zone(
        self,
        zone_mask: np.ndarray,
        total_zone_pixels: int):

        occupancy_pixels = cv2.countNonZero(zone_mask)

        if total_zone_pixels == 0:
            return {
                "occupied": False,
                "occupancy_pixels": occupancy_pixels,
                "occupancy_ratio": 0
            }

        occupancy_ratio = (occupancy_pixels / total_zone_pixels)

        occupied = (
            occupancy_ratio >
            AppConfig.OCCUPANCY_PERCENT
        )

        return {
            "occupied": occupied,
            "occupancy_pixels": occupancy_pixels,
            "occupancy_ratio": occupancy_ratio
        }

    def process_all_zones(self,motion_frame):
        results = []
        events = []

        for zone in self.zone_repository.zones:
            zone_mask = cv2.bitwise_and(
                motion_frame,
                zone.polygon_mask
            )

            occupancy_result = self.analyze_zone(
                zone_mask,
                zone.total_pixels
            )

            previous = self.update_zone_state(
                zone.name,
                occupancy_result
            )

            current = occupancy_result["occupied"]

            if not previous and current:
                events.append(f"Object ENTERED {zone.name}")

            elif previous and not current:
                events.append(f"Object EXITED {zone.name}")

            results.append({
                "zone_name": zone.name,
                "occupied": current,
                "occupancy_pixels": occupancy_result["occupancy_pixels"],
                "occupancy_ratio": occupancy_result["occupancy_ratio"]
            })

        return results, events

    def add_zone(
        self,
        name,
        points):

        self.zone_repository.add_zone(name, points)
        self.ensure_zone_exists(name)

    def build_empty_results(self):
        results = []

        for zone in self.zone_repository.zones:
            result = self.create_empty_state()
            result["zone_name"] = zone.name
            results.append(result)

        return results

    def update_zone(
        self,
        index,
        name,
        points):

        old_name = (self.zone_repository.zones[index].name)

        self.zone_repository.update_zone(
            index,
            name,
            points
        )

        if old_name != name:
            self.rename_zone(old_name, name)

    def delete_zone(self, index):
        zone_name = (self.zone_repository.zones[index].name)

        self.zone_repository.delete_zone(index)
        self.remove_zone(zone_name)

    def load_layout(self, file_path):
        self.zone_repository.load_layout(file_path)
        self.rebuild_zone_states()


class PreviewSystem:

    def __init__(
        self,
        camera_panel,
        engine,
        app,
        zone_repository):

        self.ui = camera_panel
        self.engine = engine
        self.app = app
        self.zone_repository = zone_repository

    def build_zone_display_data(self, zone_results):

        detection_map = {
            result["zone_name"]: result
            for result in zone_results
        }

        zones = []

        for zone in self.zone_repository.zones:
            result = detection_map.get(zone.name,
                {
                    "occupied": False,
                    "occupancy_ratio": 0
                }
            )

            occupied = result["occupied"]
            ratio = result["occupancy_ratio"]

            zones.append({
                "name": zone.name,
                "points_np": zone.points_np,
                "occupied": occupied,
                "occupancy_ratio": ratio
            })

        return zones

    def update(
        self,
        raw_frame,
        motion_mask,
        zone_results):

        if not self.app.preview_enabled:
            return

        zones = self.build_zone_display_data(zone_results)

        overlay = self.render_camera_overlay(raw_frame, zones)

        overlay = cv2.resize(
            overlay,
            (
                AppConfig.DISPLAY_WIDTH,
                AppConfig.DISPLAY_HEIGHT
            )
        )

        photo = self.to_photo(overlay)

        if self.ui.camera_label is not None:
            self.ui.camera_label.config(image=photo)

            self.ui.camera_label.image = photo

        if (self.engine.detection_enabled
            and motion_mask is not None):

            mask_frame = self.render_motion_overlay(
                motion_mask,
                zones
            )

            mask_frame = cv2.resize(mask_frame,
                (
                    AppConfig.DISPLAY_WIDTH,
                    AppConfig.DISPLAY_HEIGHT
                )
            )

            mask_photo = self.to_photo(mask_frame)
            self.ui.mask_label.config(image=mask_photo)
            self.ui.mask_label.image = mask_photo

    def stop_preview(self):
        if self.ui.camera_label is not None:

            self.ui.camera_label.config(image="")
            self.ui.camera_label.image = None

        if self.ui.mask_label is not None:
            self.ui.mask_label.config(image="")
            self.ui.mask_label.image = None

    def render_camera_overlay(self, frame, zones):
        overlay_frame = frame.copy()

        for zone in zones:
            color = (
                AppConfig.COLOR_RED
                if zone["occupied"]
                else AppConfig.COLOR_GREEN
            )

            cv2.polylines(
                overlay_frame,
                [zone["points_np"]],
                True,
                color,
                2
            )

            x = zone["points_np"][0][0]
            y = zone["points_np"][0][1] - 10

            label = (
                f"{zone['name']} "
                f"{zone['occupancy_ratio'] * 100:.1f}%"
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

    def render_motion_overlay(self, mask, zones):
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        mask_color[mask > 0] = AppConfig.COLOR_WHITE

        for zone in zones:
            color = (
                AppConfig.COLOR_RED
                if zone["occupied"]
                else AppConfig.COLOR_GREEN
            )

            cv2.polylines(
                mask_color,
                [zone["points_np"]],
                True,
                color,
                2
            )

            x = zone["points_np"][0][0]
            y = zone["points_np"][0][1] - 10

            label = (
                f"{zone['name']} "
                f"{zone['occupancy_ratio'] * 100:.1f}%"
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

    def to_photo(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=image)


class DetectionApp:

    def __init__(self):
        self.running = True
        self.preview_enabled = True
        self.camera_failure_logged = False
        self.latest_frame = None
        self.active_point_index = None
        self.engine = DetectionEngine(
            AppConfig.FRAME_WIDTH,
            AppConfig.FRAME_HEIGHT
        )
        self.log_panel = LogPanel()
        self.ui = None
        self.root = self.create_tkinter_gui()
        self.zone_editor_controller = ZoneEditorController(
            self.engine,
            self.ui,
            self.log_panel,
            self,
            AppConfig.FRAME_WIDTH,
            AppConfig.FRAME_HEIGHT
        )
        self.preview_controller = PreviewSystem(
            self.ui,
            self.engine,
            self,
            self.engine.zone_repository
        )
        self.start_camera()
        self.bind_events()

    def create_tkinter_gui(self):
        root = tk.Tk()
        root.title("Zone Detector")

        root.geometry(f"{AppConfig.WINDOW_WIDTH}x{AppConfig.WINDOW_HEIGHT}")

        self.ui = DetectionUI(root)
        right_frame = self.ui.build_ui()

        self.log_panel.create_log_area(right_frame)

        return root

    def bind_events(self):
        # CONTROL BUTTONS
        self.ui.capture_button.config(command=self.capture_background)
        self.ui.arm_button.config(command=self.arm_detection)
        self.ui.save_button.config(command=self.save_layout_gui)
        self.ui.load_button.config(command=self.load_layout_gui)
        self.ui.quit_button.config(command=self.quit_program)
        self.ui.stop_button.config(command=self.toggle_preview)

        # ZONE EDITOR
        self.ui.apply_button.config(command=self.zone_editor_controller.apply_zone_changes)
        self.ui.delete_button.config(command=self.zone_editor_controller.delete_selected_zone)

        self.ui.zone_selector.bind("<<ComboboxSelected>>", self.zone_editor_controller.load_zone_into_editor)

        # POINT ENTRY FOCUS
        for index, entry in enumerate(self.ui.point_entries):
            entry.bind(
                "<FocusIn>",
                lambda event, i=index:
                self.zone_editor_controller.select_point_entry(i)
            )

        # CAMERA CLICK
        self.ui.camera_label.bind("<Button-1>",self.zone_editor_controller.on_camera_click)

    def save_layout_gui(self) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        self.engine.zone_repository.save_layout(file_path)

        self.log_panel.add_log(Messages.LAYOUT_SAVED)

    def load_layout_gui(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        try:
            self.engine.load_layout(file_path)

        except json.JSONDecodeError:
            self.log_panel.add_log(Messages.INVALID_JSON_FILE)

            return

        except ValueError as error:
            self.log_panel.add_log(str(error))

            return

        except Exception:
            self.log_panel.add_log(traceback.format_exc())

            return

        self.zone_editor_controller.refresh_zone_selector()
        self.log_panel.add_log(Messages.LAYOUT_LOADED)

    def run_frame_loop(self):
        if not self.running:
            return

        frame = self.capture_frame()

        if frame is not None:
            result = self.engine.process_detection(frame)

            self.preview_controller.update(
                result["raw_frame"],
                result["motion_mask"],
                result["zone_results"]
            )

            for event in result["events"]:
                self.log_panel.add_log(event)

            self.latest_frame = frame

        self.root.after(AppConfig.GUI_UPDATE_DELAY, self.run_frame_loop)

    def quit_program(self):
        self.running = False
        self.log_panel.add_log(Messages.EXITING)

        try:
            self.stop_camera()
        except Exception:
            self.log_panel.add_log(traceback.format_exc())

        self.root.quit()
        self.root.destroy()

    def capture_background(self):
        success = self.engine.capture_background_reference(self.latest_frame)

        if not success:
            return

        if not self.engine.detection_enabled:
            self.ui.enable_arm_button()

        self.log_panel.add_log(Messages.BACKGROUND_CAPTURED)

    def arm_detection(self):
        self.engine.enable_detection()
        self.ui.disable_arm_button()
        self.ui.enable_stop_button()
        self.log_panel.add_log(Messages.DETECTION_ARMED)

    def toggle_preview(self):
        if self.preview_enabled:
            self.preview_enabled = False
            self.preview_controller.stop_preview()
            self.ui.stop_button.config(text="Resume Preview")
            self.log_panel.add_log(Messages.PREVIEW_STOPPED)

        else:
            self.preview_enabled = True
            self.ui.stop_button.config(text="Stop Preview")
            self.log_panel.add_log(Messages.PREVIEW_RESUMED)

    def start_camera(self):
        try:
            self.picam2 = Picamera2()
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

        except Exception:
            self.picam2 = None
            self.log_panel.add_log(traceback.format_exc())

    def capture_frame(self):
        if self.picam2 is None:
            return None

        try:
            return self.picam2.capture_array()

        except Exception:
            if not self.camera_failure_logged:
                if hasattr(self, "log_panel"):
                    self.log_panel.add_log(Messages.CAMERA_CAPTURE_FAILED)

                self.camera_failure_logged = True

            return None

    def stop_camera(self):
        if self.picam2 is None:
            return

        try:
            self.picam2.stop()
        except Exception:
            self.log_panel.add_log(traceback.format_exc())


@dataclass
class Zone:
    name: str
    points: list[list[int]]
    frame_width: int
    frame_height: int

    def __post_init__(self):
        self.rebuild_geometry()

    def update_points(
        self,
        new_points: list[list[int]]) -> None:

        self.points = new_points
        self.rebuild_geometry()

    def rebuild_geometry(self) -> None:
        points_np = np.array(self.points, dtype=np.int32)

        polygon_mask = np.zeros(
            (
                self.frame_height,
                self.frame_width
            ),
            dtype=np.uint8
        )

        cv2.fillPoly(
            polygon_mask,
            [points_np],
            255
        )

        total_pixels = cv2.countNonZero(polygon_mask)

        self.points_np = points_np
        self.polygon_mask = polygon_mask
        self.total_pixels = total_pixels


class DetectionUI:
    def __init__(self, root):
        self.root = root

        # Camera widgets
        self.camera_label = None
        self.mask_label = None

        # Control buttons
        self.capture_button = None
        self.arm_button = None
        self.stop_button = None
        self.load_button = None
        self.save_button = None
        self.quit_button = None

        # Zone editor widgets
        self.zone_selector = None
        self.zone_name_entry = None
        self.point_entries = []

        self.delete_button = None
        self.apply_button = None

    def build_ui(self):
        (
            camera_frame,
            mask_frame,
            left_frame,
            center_frame,
            right_frame
        ) = self.create_main_frames()

        self.create_camera_area(camera_frame, mask_frame)
        self.create_buttons_area(left_frame)
        self.create_zone_editor(center_frame)

        return right_frame

    def create_main_frames(self):
        mainFrame = tk.Frame(self.root)

        mainFrame.pack(
            anchor="nw",
            padx=10,
            pady=10
        )

        leftColumn = tk.Frame(mainFrame)

        leftColumn.pack(
            side=tk.LEFT,
            anchor="n",
            padx=(0, 10)
        )

        rightColumn = tk.Frame(mainFrame)
        rightColumn.pack(side=tk.LEFT,anchor="n")
        camera_frame = tk.Frame(leftColumn)

        camera_frame.pack(
            fill=tk.BOTH,
            expand=True,
            pady=(0, 10)
        )

        mask_frame = tk.Frame(leftColumn)
        mask_frame.pack(fill=tk.BOTH,expand=True)
        rightTopFrame = tk.Frame(rightColumn)

        rightTopFrame.pack(fill=tk.X, pady=(0, 10))
        left_frame = tk.Frame(rightTopFrame)

        left_frame.pack(
            side=tk.LEFT,
            anchor="n",
            padx=(0, 15)
        )

        center_frame = tk.Frame(rightTopFrame)
        center_frame.pack(side=tk.LEFT, anchor="n")
        right_frame = tk.Frame(rightColumn)

        right_frame.pack(fill=tk.BOTH, expand=True)

        return (
            camera_frame,
            mask_frame,
            left_frame,
            center_frame,
            right_frame
        )

    def create_camera_area(
        self,
        camera_frame,
        mask_frame):

        camera_frame.pack_propagate(False)
        mask_frame.pack_propagate(False)

        camera_frame.config(
            width=AppConfig.DISPLAY_WIDTH,
            height=AppConfig.DISPLAY_HEIGHT
        )

        mask_frame.config(
            width=AppConfig.DISPLAY_WIDTH,
            height=AppConfig.DISPLAY_HEIGHT
        )

        self.camera_label = tk.Label(camera_frame, bd=0)
        self.camera_label.pack(anchor="w")
        self.mask_label = tk.Label(mask_frame,bd=0)
        self.mask_label.pack(anchor="w")

    def create_buttons_area(self, left_frame):
        self.capture_button = tk.Button(
            left_frame,
            text="Capture Background",
            height=2,
            width=25
        )

        self.capture_button.pack(pady=10)
        self.arm_button = tk.Button(
            left_frame,
            text="Arm Detection",
            height=2,
            width=25,
            state=tk.DISABLED
        )

        self.arm_button.pack(pady=10)
        self.stop_button = tk.Button(
            left_frame,
            text="Stop Preview",
            height=2,
            width=25,
            state=tk.DISABLED
        )

        self.stop_button.pack(pady=10)
        self.load_button = tk.Button(
            left_frame,
            text="Load Layout",
            height=2,
            width=25
        )

        self.load_button.pack(pady=10)
        self.save_button = tk.Button(
            left_frame,
            text="Save Layout",
            height=2,
            width=25
        )

        self.save_button.pack(pady=10)
        self.quit_button = tk.Button(
            left_frame,
            text="Quit",
            height=2,
            width=25
        )

        self.quit_button.pack(pady=10)

    def enable_arm_button(self):
        self.arm_button.config(state=tk.NORMAL)

    def disable_arm_button(self):
        self.arm_button.config(state=tk.DISABLED)

    def enable_stop_button(self):
        self.stop_button.config(state=tk.NORMAL)

    def disable_stop_button(self):
        self.stop_button.config(state=tk.DISABLED)

    def create_zone_editor(self, center_frame):
        tk.Label(
            center_frame,
            text="Select Zone"
        ).pack()

        self.zone_selector = ttk.Combobox(
            center_frame,
            values=[Messages.ADD_NEW_ZONE_OPTION],
            state="readonly",
            width=25
        )

        self.zone_selector.pack(pady=5)
        self.zone_selector.current(0)

        tk.Label(
            center_frame,
            text="Zone Name"
        ).pack()

        zone_name_frame = tk.Frame(center_frame)
        zone_name_frame.pack(pady=5)

        self.zone_name_entry = tk.Entry(zone_name_frame, width=18)

        self.zone_name_entry.pack(side=tk.LEFT, padx=5)

        self.delete_button = tk.Button(
            zone_name_frame,
            text="X",
            width=3,
            fg="red",
            state=tk.DISABLED
        )

        self.delete_button.pack(side=tk.LEFT)
        self.point_entries.clear()

        for i in range(4):
            tk.Label(
                center_frame,
                text=f"Point {i+1} (x,y)"
            ).pack()

            entry = tk.Entry(center_frame, width=20)

            entry.pack(pady=2)
            self.point_entries.append(entry)

        self.apply_button = tk.Button(
            center_frame,
            text="Apply Zone Changes",
            height=2,
            width=25
        )

        self.apply_button.pack(pady=20)

    def get_zone_name(self):
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

    def set_zone_name(self, text):
        self.zone_name_entry.delete(0, tk.END)
        self.zone_name_entry.insert(0, text)

    def set_point_text(self, index, text):
        entry = self.point_entries[index]
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def clear_point(self, index):
        self.point_entries[index].delete(0, tk.END)

    def enable_delete_button(self):
        self.delete_button.config(state=tk.NORMAL)

    def disable_delete_button(self):
        self.delete_button.config(state=tk.DISABLED)


# MAIN PROGRAM
app = DetectionApp()
app.run_frame_loop()
app.root.mainloop()
