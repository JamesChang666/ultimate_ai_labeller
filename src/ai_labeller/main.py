import ctypes
import ctypes.wintypes
import copy
import datetime
import gc
import glob
import json
import math
import os
import shutil
import tkinter as tk
from collections import deque
from importlib import resources
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ai_labeller.core import (
    AppConfig,
    AppState,
    SessionState,
    HistoryManager,
    atomic_write_json,
    atomic_write_text,
    setup_logging,
)

# Optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import torch
    from groundingdino.util.inference import Model as GroundingDINOModel
    from segment_anything import sam_model_registry, SamPredictor
    HAS_FOUNDATION_STACK = True
except ImportError:
    HAS_FOUNDATION_STACK = False

LOGGER = setup_logging()

# ==================== Theme Colors ====================
COLORS = {
    # Primary
    "primary": "#5551FF",           # Figma ????????
    "primary_hover": "#4845E4",
    "primary_light": "#7B79FF",
    "primary_bg": "#F0F0FF",
    
    # Status colors
    "success": "#0FA958",           # Draw a new box?
    "danger": "#F24822",            # ???????
    "warning": "#FFAA00",           # ??????
    "info": "#18A0FB",              # Draw a new box?
    
    # Neutral colors
    "bg_dark": "#1E1E1E",           # ???????
    "bg_medium": "#2C2C2C",         # ?????
    "bg_light": "#F5F5F5",          # ????
    "bg_white": "#FFFFFF",          # ???
    "bg_canvas": "#18191B",         # Canvas ?
    
    # Text
    "text_primary": "#000000",
    "text_secondary": "#8E8E93",
    "text_tertiary": "#C7C7CC",
    "text_white": "#FFFFFF",
    
    # Borders
    "border": "#E5E5EA",
    "divider": "#38383A",
    
    # Bounding boxes
    "box_1": "#FF3B30",  # ??
    "box_2": "#FF9500",  # ??
    "box_3": "#FFCC00",  # ??
    "box_4": "#34C759",  # ??
    "box_5": "#5AC8FA",  # ??
    "box_6": "#5856D6",  # ??
    "box_selected": "#00D4FF",  # ???????
    
    # Effects
    "shadow": "rgba(0, 0, 0, 0.1)",
}

THEMES = {
    "dark": {
        "bg_dark": "#1E1E1E",
        "bg_medium": "#2C2C2C",
        "bg_light": "#F5F5F5",
        "bg_white": "#FFFFFF",
        "bg_canvas": "#18191B",
        "text_primary": "#000000",
        "text_secondary": "#8E8E93",
        "text_tertiary": "#C7C7CC",
        "border": "#E5E5EA",
        "divider": "#38383A",
    },
    "light": {
        "bg_dark": "#F5F5F7",
        "bg_medium": "#E5E5EA",
        "bg_light": "#FFFFFF",
        "bg_white": "#FFFFFF",
        "bg_canvas": "#EFEFF4",
        "text_primary": "#000000",
        "text_secondary": "#5C5C5C",
        "text_tertiary": "#8E8E93",
        "border": "#D1D1D6",
        "divider": "#D1D1D6",
    },
}

# ==================== Language ====================
LANG_MAP = {
    "zh": {
        "title": "AI Labeller Pro",
        "load_proj": "[ZH] Load Project",
        "undo": "[ZH] Undo",
        "redo": "[ZH] Redo",
        "autolabel": "[ZH] Red Detection",
        "fuse": "[ZH] Fuse Boxes",
        "file_info": "[ZH] File Info",
        "no_img": "[ZH] No Image",
        "filename": "[ZH] File",
        "progress": "[ZH] Progress",
        "boxes": "[ZH] Boxes",
        "class_mgmt": "[ZH] Classes",
        "current_class": "[ZH] Current Class",
        "edit_classes": "[ZH] Edit Classes",
        "reassign_class": "[ZH] Reassign Selected Class",
        "clear_labels": "[ZH] Clear Labels (Current)",
        "add": "[ZH] Add",
        "rename": "[ZH] Rename",
        "apply": "[ZH] Apply",
        "class_name": "[ZH] Class Name",
        "rename_prompt": "[ZH] Modify '{name}':",
        "add_prompt": "[ZH] Class name:",
        "current": "[ZH] Current",
        "to": "[ZH] To",
        "no_label_selected": "[ZH] No label selected.",
        "no_classes_available": "[ZH] No classes available.",
        "theme_light": "[ZH] Light Mode",
        "theme_dark": "[ZH] Dark Mode",
        "export_format": "[ZH] Export All As",
        "ai_tools": "[ZH] AI Tools",
        "auto_detect": "[ZH] Auto Detect",
        "learning": "[ZH] Learning",
        "foundation_mode": "[ZH] Foundation Assist",
        "propagate": "[ZH] Propagate",
        "run_detection": "[ZH] Run Detection",
        "detection_model": "[ZH] Detection Model",
        "browse_model": "[ZH] Browse Model",
        "use_official_yolo26n": "[ZH] Use Official yolo26m.pt",
        "export": "[ZH] Export All",
        "prev": "[ZH] Previous",
        "next": "[ZH] Next",
        "shortcuts": "[ZH] Shortcuts",
        "shortcut_help": "[ZH] Shortcut Help",
        "dataset": "[ZH] Dataset",
        "lang_switch": "EN",
        "delete": "[ZH] Delete Selected",
        "remove_from_split": "[ZH] Remove From Split",
        "remove_confirm": "[ZH] Remove current image from {split}?",
        "remove_done": "[ZH] Removed: {name}",
        "remove_none": "[ZH] No image to remove.",
        "restore_from_split": "[ZH] Restore Deleted Frame",
        "restore_none": "[ZH] No removed frame found in this split.",
        "restore_title": "[ZH] Restore Deleted Frame",
        "restore_select": "[ZH] Select a frame to restore:",
        "restore_done": "[ZH] Restored: {name}",
        "select_image": "[ZH] Select Image",
        "startup_choose_source": "[ZH] Choose Startup Source",
        "startup_prompt": "[ZH] How do you want to start?",
        "startup_images": "[ZH] Open Images Folder",
        "startup_yolo": "[ZH] Open YOLO Dataset",
        "startup_rfdetr": "[ZH] Open RF-DETR Dataset",
        "startup_skip": "[ZH] Later",
        "back_to_source": "[ZH] Back to Source Select",
        "startup_model_cancel_title": "[ZH] Model Selection Cancelled",
        "startup_model_cancel_msg": "[ZH] No model selected. Continue with images folder only?",
        "pick_folder_title": "[ZH] Select Folder",
        "loaded_from": "[ZH] Loaded {count} images\nFrom: {path}\nSplit: {split}",
        "no_supported_images": "[ZH] No supported images found (png/jpg/jpeg)\nFolder: {path}",
        "select_export_folder": "[ZH] Select Export Folder",
        "export_no_project": "[ZH] No dataset loaded.",
        "export_done": "[ZH] Export completed: {count} images\nOutput: {path}",
        "export_failed": "[ZH] Export failed: {err}",
    },
    "en": {
        "title": "AI Labeller Pro",
        "load_proj": "Load Project",
        "undo": "Undo",
        "redo": "Redo",
        "autolabel": "Red Detection",
        "fuse": "Fuse Boxes",
        "file_info": "File Info",
        "no_img": "No Image",
        "filename": "File",
        "progress": "Progress",
        "boxes": "Boxes",
        "class_mgmt": "Classes",
        "current_class": "Current Class",
        "edit_classes": "Edit Classes",
        "reassign_class": "Reassign Selected Class",
        "clear_labels": "Clear Labels (Current)",
        "add": "Add",
        "rename": "Rename",
        "apply": "Apply",
        "class_name": "Class Name",
        "rename_prompt": "Modify '{name}':",
        "add_prompt": "Class name:",
        "current": "Current",
        "to": "To",
        "no_label_selected": "No label selected.",
        "no_classes_available": "No classes available.",
        "theme_light": "Light Mode",
        "theme_dark": "Dark Mode",
        "export_format": "Export All As",
        "ai_tools": "AI Tools",
        "auto_detect": "Auto Detect",
        "learning": "Learning",
        "foundation_mode": "Foundation Assist",
        "propagate": "Propagate",
        "run_detection": "Run Detection",
        "detection_model": "Detection Model",
        "browse_model": "Browse Model",
        "use_official_yolo26n": "Use Official yolo26m.pt",
        "export": "Export All",
        "prev": "Previous",
        "next": "Next",
        "shortcuts": "Shortcuts",
        "shortcut_help": "Shortcut Help",
        "dataset": "Dataset",
        "lang_switch": "ZH",
        "delete": "Delete Selected",
        "remove_from_split": "Remove From Split",
        "remove_confirm": "Remove current image from {split}?",
        "remove_done": "Removed: {name}",
        "remove_none": "No image to remove.",
        "restore_from_split": "Restore Deleted Frame",
        "restore_none": "No removed frame found in this split.",
        "restore_title": "Restore Deleted Frame",
        "restore_select": "Select a frame to restore:",
        "restore_done": "Restored: {name}",
        "select_image": "Select Image",
        "startup_choose_source": "Choose Startup Source",
        "startup_prompt": "How do you want to start?",
        "startup_images": "Open Images Folder",
        "startup_yolo": "Open YOLO Dataset",
        "startup_rfdetr": "Open RF-DETR Dataset",
        "startup_skip": "Later",
        "back_to_source": "Back to Source Select",
        "startup_model_cancel_title": "Model Selection Cancelled",
        "startup_model_cancel_msg": "No model selected. Continue with images folder only?",
        "pick_folder_title": "Select Folder",
        "loaded_from": "Loaded {count} images\nFrom: {path}\nSplit: {split}",
        "no_supported_images": "No supported images found (png/jpg/jpeg)\nFolder: {path}",
        "select_export_folder": "Select Export Folder",
        "export_no_project": "No dataset loaded.",
        "export_done": "Export completed: {count} images\nOutput: {path}",
        "export_failed": "Export failed: {err}",
    },
}
# ==================== Main App ====================
class UltimateLabeller:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.lang = "en"
        self.theme = "dark"
        self.config = AppConfig()
        self.state = AppState()
        self.history_manager = HistoryManager()
        self.logger = setup_logging(os.path.join(os.path.expanduser("~"), ".ai_labeller.log"))

        self.root.title(LANG_MAP[self.lang]["title"])
        self.root.geometry(self.config.default_window_size)
        self.root.minsize(self.config.min_window_width, self.config.min_window_height)
        self.window_icon_tk = None
        self.toolbar_logo_tk = None
        
        # ???????????????
        self.setup_fonts()
        self.apply_theme(self.theme, rebuild=False)
        self.setup_app_icon()
        self._tooltip_after_id = None
        self._tooltip_win = None
        
        # --- ????? ---
        self.project_root = self.state.project_root
        self.current_split = self.state.current_split
        self.image_files = self.state.image_files
        self.current_idx = self.state.current_idx
        self.rects = self.state.rects  # [x1, y1, x2, y2, cid]
        self.class_names = self.state.class_names
        self.learning_mem = deque(maxlen=self.config.max_learning_memory)
        
        # --- ??????---
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.img_pil = None
        self.img_tk = None
        self.selected_idx = None
        self.active_handle = None
        self.is_moving_box = False
        self.drag_start = None
        self.temp_rect_coords = None
        self.mouse_pos = (0, 0)
        self.HANDLE_SIZE = self.config.handle_size
        self.show_all_labels = True
        self._cursor_line_x: int | None = None
        self._cursor_line_y: int | None = None
        self._cursor_text_id: int | None = None
        self._cursor_bg_id: int | None = None
        
        # --- AI ? ---
        self.yolo_model = None
        self.yolo_path = tk.StringVar(value=self.config.yolo_model_path)
        self.det_model_mode = tk.StringVar(value="Official YOLO26m.pt (Bundled)")
        self._loaded_model_key: tuple[str, str] | None = None
        self.model_library: list[str] = [self.config.yolo_model_path]
        self.var_export_format = tk.StringVar(value="YOLO (.txt)")
        self.var_auto_yolo = tk.BooleanVar(value=False)
        self.var_propagate = tk.BooleanVar(value=False)
        self.var_yolo_conf = tk.DoubleVar(value=self.config.default_yolo_conf)
        self.session_path = os.path.join(os.path.expanduser("~"), self.config.session_file_name)
        self.foundation_dino = None
        self.foundation_sam_predictor = None
        self._uncertainty_cache: dict[str, float] = {}
        self._active_scan_offset = 0
        self._folder_dialog_open = False
        self._startup_dialog_shown = False
        self._startup_dialog_open = False
        
        self.setup_custom_style()
        self.setup_ui()
        self.bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.load_session_state()
        self.root.after(120, self.show_startup_source_dialog)
    
    def setup_fonts(self):
        """Configure platform-specific font families and sizes."""
        import platform
        system = platform.system()
        
        if system == "Windows":
            self.font_primary = ("Segoe UI", 10)
            self.font_bold = ("Segoe UI", 10, "bold")
            self.font_title = ("Segoe UI", 14, "bold")
            self.font_mono = ("Consolas", 9)
        elif system == "Darwin":  # macOS
            self.font_primary = ("SF Pro Text", 10)
            self.font_bold = ("SF Pro Text", 10, "bold")
            self.font_title = ("SF Pro Display", 14, "bold")
            self.font_mono = ("SF Mono", 9)
        else:  # Linux
            self.font_primary = ("Ubuntu", 10)
            self.font_bold = ("Ubuntu", 10, "bold")
            self.font_title = ("Ubuntu", 14, "bold")
            self.font_mono = ("Ubuntu Mono", 9)
    
    def setup_custom_style(self):
        """?????? ttk ???"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # ?????? Combobox
        style.configure("TCombobox",
            fieldbackground=COLORS["bg_white"],
            background=COLORS["bg_light"],
            foreground=COLORS["text_primary"],
            borderwidth=0,
            relief="flat",
            arrowcolor=COLORS["text_secondary"]
        )
        
        style.map("TCombobox",
            fieldbackground=[('readonly', COLORS["bg_white"])],
            selectbackground=[('readonly', COLORS["primary_bg"])],
            selectforeground=[('readonly', COLORS["primary"])]
        )

    def _resolve_asset_path(self, relative_path: str) -> str | None:
        try:
            packaged = resources.files("ai_labeller").joinpath(relative_path)
            if packaged.is_file():
                return str(packaged)
        except Exception:
            pass
        local = os.path.join(os.path.dirname(__file__), relative_path)
        if os.path.isfile(local):
            return local
        return None

    def setup_app_icon(self) -> None:
        icon_path = self._resolve_asset_path("assets/app_icon.png")
        if not icon_path:
            return
        try:
            icon_img = Image.open(icon_path).convert("RGBA")
            win_icon = icon_img.resize((32, 32), Image.Resampling.LANCZOS)
            toolbar_icon = icon_img.resize((20, 20), Image.Resampling.LANCZOS)
            self.window_icon_tk = ImageTk.PhotoImage(win_icon)
            self.toolbar_logo_tk = ImageTk.PhotoImage(toolbar_icon)
            self.root.iconphoto(True, self.window_icon_tk)
            icon_img.close()
        except Exception:
            self.logger.exception("Failed to load app icon")
    
    def delete_selected(self, e=None):
        """?????????????"""
        if self.selected_idx is not None:
            self.push_history()
            self.rects.pop(self.selected_idx)
            self.selected_idx = None
            self.render()
    
    def setup_ui(self):
        # ==================== ??????====================
        self.setup_toolbar()
        
        # ==================== ?????====================
        self.setup_sidebar()
        
        # ==================== Canvas ====================
        self.canvas = tk.Canvas(
            self.root,
            bg=COLORS["bg_canvas"],
            cursor="none",
            highlightthickness=0,
            relief="flat"
        )
        self.canvas.pack(fill="both", expand=True)
    
    def setup_toolbar(self):
        """????????????"""
        toolbar = tk.Frame(self.root, bg=COLORS["bg_dark"], height=56)
        toolbar.pack(side="top", fill="x")
        toolbar.pack_propagate(False)
        
        # ???????
        left_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        left_frame.pack(side="left", fill="y", padx=16)
        
        # Logo ????
        title_frame = tk.Frame(left_frame, bg=COLORS["bg_dark"])
        title_frame.pack(side="left", pady=12)
        
        if self.toolbar_logo_tk is not None:
            logo = tk.Label(
                title_frame,
                image=self.toolbar_logo_tk,
                bg=COLORS["bg_dark"]
            )
        else:
            logo = tk.Label(
                title_frame,
                text="AI",
                font=("Arial", 20),
                fg=COLORS["primary"],
                bg=COLORS["bg_dark"]
            )
        logo.pack(side="left", padx=(0, 8))
        logo.bind("<Button-1>", self.return_to_source_select)
        logo.bind("<Enter>", lambda _e: logo.config(cursor="hand2"))
        
        tk.Label(
            title_frame,
            text=LANG_MAP[self.lang]["title"],
            font=self.font_title,
            fg=self.toolbar_text_color(COLORS["bg_dark"]),
            bg=COLORS["bg_dark"]
        ).pack(side="left")
        
        # Draw a new box?
        tk.Frame(
            left_frame,
            width=1,
            bg=COLORS["divider"]
        ).pack(side="left", fill="y", padx=16)
        
        # ??????
        self.create_toolbar_button(
            left_frame,
            text=LANG_MAP[self.lang]["load_proj"],
            command=lambda: self.show_startup_source_dialog(force=True),
            bg=COLORS["primary"]
        ).pack(side="left", padx=4)
        
        # Dataset ?????
        dataset_frame = tk.Frame(left_frame, bg=COLORS["bg_dark"])
        dataset_frame.pack(side="left", padx=12)
        
        tk.Label(
            dataset_frame,
            text=LANG_MAP[self.lang]["dataset"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_dark"]
        ).pack(side="left", padx=(0, 8))
        
        self.combo_split = ttk.Combobox(
            dataset_frame,
            values=["train", "val", "test"],
            width=10,
            state="readonly",
            font=self.font_primary
        )
        self.combo_split.current(0)
        self.combo_split.pack(side="left")
        self.combo_split.bind("<<ComboboxSelected>>", self.on_split_change)
        
        # ????????- ??????????
        center_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        center_frame.pack(side="left", fill="y", padx=16)
        
        # Draw a new box??
        self.create_toolbar_icon_button(
            center_frame,
            text="U",
            command=self.undo,
            tooltip=LANG_MAP[self.lang]["undo"]
        ).pack(side="left", padx=2)
        
        self.create_toolbar_icon_button(
            center_frame,
            text="R",
            command=self.redo,
            tooltip=LANG_MAP[self.lang]["redo"]
        ).pack(side="left", padx=2)
        
        tk.Frame(center_frame, width=1, bg=COLORS["divider"]).pack(
            side="left", fill="y", padx=8
        )
        
        # AI ???
        self.create_toolbar_icon_button(
            center_frame,
            text="?",
            command=self.autolabel_red,
            tooltip=LANG_MAP[self.lang]["autolabel"],
            bg=COLORS["danger"]
        ).pack(side="left", padx=2)
        
        # Draw a new box???
        right_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        right_frame.pack(side="right", fill="y", padx=16)

        # ????????????????????? 1 ????????????
        self.create_help_icon(right_frame).pack(side="right", padx=4, pady=12)

        # ??????
        self.create_toolbar_button(
            right_frame,
            text=self.get_theme_switch_label(),
            command=self.toggle_theme,
            bg=COLORS["bg_medium"]
        ).pack(side="right", padx=4, pady=12)

        # Draw a new box?
        self.create_toolbar_button(
            right_frame,
            text=LANG_MAP[self.lang]["lang_switch"],
            command=self.toggle_language,
            bg=COLORS["bg_medium"]
        ).pack(side="right", padx=4, pady=12)
    
    def create_toolbar_button(self, parent, text, command, bg=None):
        """??????????????"""
        bg_val = bg or COLORS["bg_medium"]
        fg_val = self.toolbar_text_color(bg_val)
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_val,
            fg=fg_val,
            font=self.font_primary,
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0
        )
        
        # ?????
        def on_enter(e):
            btn.config(bg=COLORS["primary_hover"] if bg == COLORS["primary"] else COLORS["bg_medium"])
        
        def on_leave(e):
            btn.config(bg=bg_val)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn

    def create_help_icon(self, parent):
        """?????????????????"""
        btn = tk.Label(
            parent,
            text="?",
            font=self.font_bold,
            fg=self.toolbar_text_color(COLORS["bg_medium"]),
            bg=COLORS["bg_medium"],
            width=3,
            height=1,
            relief="flat"
        )
        btn.bind("<Enter>", lambda e: self.show_shortcut_tooltip(btn))
        btn.bind("<Leave>", lambda e: self.hide_shortcut_tooltip())
        return btn

    def build_shortcut_text(self):
        items = [
            ("F", LANG_MAP[self.lang]["next"]),
            ("D", LANG_MAP[self.lang]["prev"]),
            ("A", LANG_MAP[self.lang]["autolabel"]),
            ("Ctrl+Z", LANG_MAP[self.lang]["undo"]),
            ("Ctrl+Y", LANG_MAP[self.lang]["redo"]),
            ("Del", LANG_MAP[self.lang]["delete"]),
        ]
        lines = [LANG_MAP[self.lang]["shortcut_help"]]
        for key, desc in items:
            lines.append(f"{key} - {desc}")
        return "\n".join(lines)

    def show_shortcut_tooltip(self, widget):
        self.hide_shortcut_tooltip()
        self._tooltip_after_id = self.root.after(1000, lambda: self._show_tooltip_now(widget))

    def _show_tooltip_now(self, widget):
        if self._tooltip_win:
            return
        x = widget.winfo_rootx() + 10
        y = widget.winfo_rooty() + widget.winfo_height() + 6
        win = tk.Toplevel(self.root)
        win.wm_overrideredirect(True)
        win.configure(bg=COLORS["bg_white"])
        label = tk.Label(
            win,
            text=self.build_shortcut_text(),
            font=self.font_primary,
            fg=COLORS["text_primary"],
            bg=COLORS["bg_white"],
            justify="left",
            anchor="w",
            padx=10,
            pady=8
        )
        label.pack()
        win.update_idletasks()

        tooltip_w = win.winfo_reqwidth()
        tooltip_h = win.winfo_reqheight()
        left, top, right, bottom = self._get_widget_monitor_bounds(widget)
        margin = 8

        if x + tooltip_w > right - margin:
            x = right - tooltip_w - margin
        if x < left + margin:
            x = left + margin

        if y + tooltip_h > bottom - margin:
            y = widget.winfo_rooty() - tooltip_h - 6
        if y < top + margin:
            y = top + margin

        win.wm_geometry(f"+{int(x)}+{int(y)}")
        self._tooltip_win = win

    def _get_widget_monitor_bounds(self, widget):
        # Fallback for non-Windows or API failure: virtual desktop bounds.
        left = widget.winfo_vrootx()
        top = widget.winfo_vrooty()
        right = left + widget.winfo_vrootwidth()
        bottom = top + widget.winfo_vrootheight()

        try:
            user32 = ctypes.windll.user32
            monitor = user32.MonitorFromPoint(
                ctypes.wintypes.POINT(widget.winfo_rootx(), widget.winfo_rooty()),
                2
            )
            if not monitor:
                return left, top, right, bottom

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                ]

            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_ulong),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", ctypes.c_ulong),
                ]

            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            ok = user32.GetMonitorInfoW(monitor, ctypes.byref(mi))
            if ok:
                return (
                    mi.rcWork.left,
                    mi.rcWork.top,
                    mi.rcWork.right,
                    mi.rcWork.bottom,
                )
        except Exception:
            pass

        return left, top, right, bottom

    def hide_shortcut_tooltip(self):
        if self._tooltip_after_id:
            self.root.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        if self._tooltip_win:
            self._tooltip_win.destroy()
            self._tooltip_win = None
    
    def create_toolbar_icon_button(self, parent, text, command, tooltip="", bg=None):
        """????????????????"""
        bg_val = bg or COLORS["bg_medium"]
        fg_val = self.toolbar_text_color(bg_val)
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_val,
            fg=fg_val,
            font=("Arial", 14),
            relief="flat",
            width=3,
            height=1,
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0
        )
        
        # ?????
        def on_enter(e):
            if bg:
                btn.config(bg=self.lighten_color(bg))
            else:
                btn.config(bg=COLORS["divider"])
        
        def on_leave(e):
            btn.config(bg=bg_val)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def lighten_color(self, color):
        """??????????"""
        if color == COLORS["danger"]:
            return "#FF6B54"
        elif color == COLORS["warning"]:
            return "#FFBB33"
        return color
    
    def setup_sidebar(self):
        """???????????"""
        sidebar = tk.Frame(self.root, width=320, bg=COLORS["bg_light"])
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)
        
        # ?????????
        self.sidebar_canvas = tk.Canvas(
            sidebar,
            bg=COLORS["bg_light"],
            highlightthickness=0,
            relief="flat"
        )
        self.sidebar_scrollbar = ttk.Scrollbar(
            sidebar,
            orient="vertical",
            command=self.sidebar_canvas.yview
        )
        self.sidebar_scroll_frame = tk.Frame(self.sidebar_canvas, bg=COLORS["bg_light"])
        
        self.sidebar_window = self.sidebar_canvas.create_window(
            (0, 0),
            window=self.sidebar_scroll_frame,
            anchor="nw"
        )
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_scroll_frame.bind("<Configure>", self._on_sidebar_frame_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self.sidebar_canvas.bind("<MouseWheel>", self._on_sidebar_mousewheel)
        self.sidebar_canvas.bind("<Button-4>", lambda e: self.sidebar_canvas.yview_scroll(-1, "units"))
        self.sidebar_canvas.bind("<Button-5>", lambda e: self.sidebar_canvas.yview_scroll(1, "units"))
        
        # ===== ??????? =====
        self.create_info_card(self.sidebar_scroll_frame)
        
        # ===== ???????? =====
        self.create_class_card(self.sidebar_scroll_frame)
        
        # ===== AI ?????? =====
        self.create_ai_card(self.sidebar_scroll_frame)
        
        # ===== ????????=====
        self.create_shortcut_card(self.sidebar_scroll_frame)
        
        # ===== ??? =====
        self.create_navigation(sidebar)
        self._bind_sidebar_mousewheel(self.sidebar_scroll_frame)
        
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_scrollbar.pack(side="right", fill="y")

    def _on_sidebar_frame_configure(self, e=None):
        if hasattr(self, "sidebar_canvas"):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, e):
        if hasattr(self, "sidebar_canvas") and hasattr(self, "sidebar_window"):
            self.sidebar_canvas.itemconfigure(self.sidebar_window, width=e.width)

    def _on_sidebar_mousewheel(self, e):
        if e.delta:
            self.sidebar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        return "break"

    def _bind_sidebar_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_sidebar_mousewheel, add="+")
        widget.bind("<Button-4>", lambda e: self.sidebar_canvas.yview_scroll(-1, "units"), add="+")
        widget.bind("<Button-5>", lambda e: self.sidebar_canvas.yview_scroll(1, "units"), add="+")
        for child in widget.winfo_children():
            self._bind_sidebar_mousewheel(child)

    def get_theme_switch_label(self):
        key = "theme_light" if self.theme == "dark" else "theme_dark"
        return LANG_MAP[self.lang][key]

    def is_accent_bg(self, bg):
        return bg in {
            COLORS["primary"],
            COLORS["warning"],
            COLORS["danger"],
            COLORS["success"],
            COLORS["info"],
            COLORS["primary_light"],
        }

    def toolbar_text_color(self, bg):
        if self.is_accent_bg(bg):
            return COLORS["text_white"]
        return COLORS["text_white"] if self.theme == "dark" else COLORS["text_primary"]

    def apply_theme(self, theme, rebuild=True):
        self.theme = theme
        palette = THEMES.get(theme, THEMES["dark"])
        for k, v in palette.items():
            COLORS[k] = v
        self.root.configure(bg=COLORS["bg_dark"])
        if rebuild:
            self.rebuild_ui()

    def toggle_theme(self):
        new_theme = "light" if self.theme == "dark" else "dark"
        self.apply_theme(new_theme)

    def rebuild_ui(self):
        self.hide_shortcut_tooltip()
        for child in self.root.winfo_children():
            child.destroy()
        self.setup_custom_style()
        self.setup_ui()
        self.bind_events()
        self.root.title(LANG_MAP[self.lang]["title"])
        self.update_info_text()
        self.render()
    
    def create_card(self, parent, title=None):
        """Create a styled card container and return its content frame."""
        card = tk.Frame(
            parent,
            bg=COLORS["bg_white"],
            relief="flat",
            borderwidth=0
        )
        card.pack(fill="x", padx=16, pady=8)
        
        # ?????????????????????????????
        card_border = tk.Frame(card, bg=COLORS["border"], height=1)
        card_border.pack(fill="x", side="bottom")
        
        content = tk.Frame(card, bg=COLORS["bg_white"])
        content.pack(fill="both", expand=True, padx=16, pady=16)
        
        if title:
            title_label = tk.Label(
                content,
                text=title,
                font=self.font_bold,
                fg=COLORS["text_primary"],
                bg=COLORS["bg_white"],
                anchor="w"
            )
            title_label.pack(fill="x", pady=(0, 12))
        
        return content
    
    def create_info_card(self, parent):
        """??????????????"""
        content = self.create_card(parent, LANG_MAP[self.lang]["file_info"])
        
        # Draw a new box??
        self.lbl_filename = tk.Label(
            content,
            text=LANG_MAP[self.lang]["no_img"],
            font=self.font_mono,
            fg=COLORS["text_primary"],
            bg=COLORS["bg_white"],
            anchor="w",
            wraplength=260
        )
        self.lbl_filename.pack(fill="x")

        tk.Label(
            content,
            text=LANG_MAP[self.lang].get("select_image", "Select Image"),
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(10, 6))

        self.combo_image = ttk.Combobox(
            content,
            values=[],
            state="readonly",
            font=self.font_primary
        )
        self.combo_image.pack(fill="x")
        self.combo_image.bind("<<ComboboxSelected>>", self.on_image_selected)
        
        # Draw a new box??????
        progress_frame = tk.Frame(content, bg=COLORS["bg_white"])
        progress_frame.pack(fill="x", pady=(8, 0))
        
        # Draw a new box???
        self.lbl_progress = tk.Label(
            progress_frame,
            text="0 / 0",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"]
        )
        self.lbl_progress.pack(side="left")
        
        # ??????
        self.lbl_box_count = tk.Label(
            progress_frame,
            text=f"{LANG_MAP[self.lang]['boxes']}: 0",
            font=self.font_primary,
            fg=COLORS["primary"],
            bg=COLORS["bg_white"]
        )
        self.lbl_box_count.pack(side="right")
    
    def create_class_card(self, parent):
        """???????????????"""
        content = self.create_card(parent, LANG_MAP[self.lang]["class_mgmt"])
        
        # Draw a new box?????
        tk.Label(
            content,
            text=LANG_MAP[self.lang]["current_class"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        # ??????
        self.combo_cls = ttk.Combobox(
            content,
            values=self.class_names,
            state="readonly",
            font=self.font_primary
        )
        self.combo_cls.current(0)
        self.combo_cls.pack(fill="x", pady=(0, 12))
        self.combo_cls.bind("<<ComboboxSelected>>", self.on_class_change_request)
        
        # ????????????
        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["edit_classes"],
            command=self.edit_classes_table
        ).pack(fill="x", pady=(0, 12))

        # Draw a new box?????
        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang]["clear_labels"],
            command=self.clear_current_labels
        ).pack(fill="x", pady=(0, 12))

        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang].get("remove_from_split", "Remove From Split"),
            command=self.remove_current_from_split
        ).pack(fill="x", pady=(0, 12))

        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang].get("restore_from_split", "Restore Deleted Frame"),
            command=self.open_restore_removed_dialog
        ).pack(fill="x", pady=(0, 12))
        
        # Export format
        tk.Label(
            content,
            text=LANG_MAP[self.lang]["export_format"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        ttk.Combobox(
            content,
            textvariable=self.var_export_format,
            values=["YOLO (.txt)", "JSON"],
            state="readonly",
            font=self.font_primary
        ).pack(fill="x")

        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["export"],
            command=self.export_all_by_selected_format,
            bg=COLORS["info"]
        ).pack(fill="x", pady=(10, 0))
    
    def create_ai_card(self, parent):
        """??????? AI ??????"""
        content = self.create_card(parent, LANG_MAP[self.lang]["ai_tools"])
        
        # ?????????
        checkbox_style = {
            "bg": COLORS["bg_white"],
            "fg": COLORS["text_primary"],
            "font": self.font_primary,
            "activebackground": COLORS["bg_white"],
            "selectcolor": COLORS["bg_white"],
            "anchor": "w"
        }
        
        tk.Checkbutton(
            content,
            text=LANG_MAP[self.lang]["auto_detect"],
            variable=self.var_auto_yolo,
            **checkbox_style
        ).pack(fill="x", pady=4)
        
        tk.Checkbutton(
            content,
            text=LANG_MAP[self.lang]["propagate"],
            variable=self.var_propagate,
            **checkbox_style
        ).pack(fill="x", pady=4)

        tk.Label(
            content,
            text=LANG_MAP[self.lang].get("detection_model", "Detection Model"),
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(10, 4))

        self.combo_det_model = ttk.Combobox(
            content,
            textvariable=self.det_model_mode,
            values=[
                "Official YOLO26m.pt (Bundled)",
                "Custom YOLO (v5/v7/v8/v9/v11/v26)",
                "Custom RF-DETR",
            ],
            state="readonly",
            font=self.font_primary
        )
        self.combo_det_model.pack(fill="x", pady=(0, 6))
        self.combo_det_model.bind("<<ComboboxSelected>>", self.on_detection_model_mode_changed)

        self.combo_model_path = ttk.Combobox(
            content,
            textvariable=self.yolo_path,
            values=self.model_library,
            state="readonly",
            font=self.font_primary
        )
        self.combo_model_path.pack(fill="x", pady=(0, 6))
        self._refresh_model_dropdown()

        picker_row = tk.Frame(content, bg=COLORS["bg_white"])
        picker_row.pack(fill="x", pady=(0, 6))

        self.create_secondary_button(
            picker_row,
            text=LANG_MAP[self.lang].get("browse_model", "Browse Model"),
            command=self.browse_detection_model
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.create_secondary_button(
            picker_row,
            text=LANG_MAP[self.lang].get("use_official_yolo26n", "Use Official yolo26m.pt"),
            command=self.use_official_yolo26n
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        
        # Detection trigger
        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["run_detection"],
            command=self.run_yolo_detection,
            bg=COLORS["success"]
        ).pack(fill="x", pady=(12, 0))
    
    def create_shortcut_card(self, parent):
        """Render shortcut hint list."""
        content = self.create_card(parent, LANG_MAP[self.lang]["shortcuts"])
        
        shortcuts = [
            ("F", LANG_MAP[self.lang]["next"]),
            ("D", LANG_MAP[self.lang]["prev"]),
            ("A", LANG_MAP[self.lang]["autolabel"]),
            ("Ctrl+Z", LANG_MAP[self.lang]["undo"]),
            ("Ctrl+Y", LANG_MAP[self.lang]["redo"]),
            ("Del", LANG_MAP[self.lang]["delete"])
        ]
        
        for key, desc in shortcuts:
            row = tk.Frame(content, bg=COLORS["bg_white"])
            row.pack(fill="x", pady=2)
            
            tk.Label(
                row,
                text=key,
                font=self.font_mono,
                fg=COLORS["primary"],
                bg=COLORS["bg_white"],
                width=12,
                anchor="w"
            ).pack(side="left")
            
            tk.Label(
                row,
                text=desc,
                font=self.font_primary,
                fg=COLORS["text_secondary"],
                bg=COLORS["bg_white"],
                anchor="w"
            ).pack(side="left")
    
    def create_navigation(self, parent):
        """??????????"""
        nav_frame = tk.Frame(parent, bg=COLORS["bg_light"], height=80)
        nav_frame.pack(side="bottom", fill="x")
        nav_frame.pack_propagate(False)
        
        btn_container = tk.Frame(nav_frame, bg=COLORS["bg_light"])
        btn_container.pack(fill="both", expand=True, padx=16, pady=16)
        
        # Draw a new box?
        self.create_nav_button(
            btn_container,
            text=LANG_MAP[self.lang]["prev"],
            command=self.prev_img,
            side="left"
        )
        
        # Draw a new box?
        self.create_nav_button(
            btn_container,
            text=f"{LANG_MAP[self.lang]['next']} >",
            command=self.save_and_next,
            side="right",
            primary=True
        )
    
    def create_primary_button(self, parent, text, command, bg=None):
        """?????????????"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg or COLORS["primary"],
            fg=COLORS["text_white"],
            font=self.font_primary,
            relief="flat",
            pady=10,
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0
        )
        
        # ?????
        original_bg = bg or COLORS["primary"]
        hover_bg = COLORS["primary_hover"] if not bg else self.lighten_color(bg)
        
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda e: btn.config(bg=original_bg))
        
        return btn

    def create_secondary_button(self, parent, text, command):
        """Create a secondary sidebar button."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["bg_white"],
            fg=COLORS["text_primary"],
            font=self.font_primary,
            relief="flat",
            pady=10,
            cursor="hand2",
            borderwidth=1,
            highlightthickness=0
        )
        btn.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"])
        btn.bind("<Enter>", lambda e: btn.config(bg=COLORS["bg_light"]))
        btn.bind("<Leave>", lambda e: btn.config(bg=COLORS["bg_white"]))
        return btn
    
    def create_nav_button(self, parent, text, command, side, primary=False):
        """??????????"""
        bg = COLORS["primary"] if primary else COLORS["bg_white"]
        fg = COLORS["text_white"] if primary else COLORS["text_primary"]
        
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            font=self.font_bold,
            relief="flat",
            pady=12,
            cursor="hand2",
            borderwidth=0 if primary else 1,
            highlightthickness=0
        )
        
        if not primary:
            btn.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"])
        
        # ?????
        if primary:
            btn.bind("<Enter>", lambda e: btn.config(bg=COLORS["primary_hover"]))
            btn.bind("<Leave>", lambda e: btn.config(bg=COLORS["primary"]))
        else:
            btn.bind("<Enter>", lambda e: btn.config(bg=COLORS["bg_light"]))
            btn.bind("<Leave>", lambda e: btn.config(bg=COLORS["bg_white"]))
        
        if side == "left":
            btn.pack(side="left", fill="both", expand=True, padx=(0, 4))
        else:
            btn.pack(side="right", fill="both", expand=True, padx=(4, 0))
    
    def toggle_language(self):
        """????"""
        self.lang = "en" if self.lang == "zh" else "zh"
        self.rebuild_ui()
    
    def update_info_text(self):
        """????????"""
        if not self.image_files:
            self.lbl_filename.config(text=LANG_MAP[self.lang]["no_img"])
            self.lbl_progress.config(text="0 / 0")
            if hasattr(self, "combo_image"):
                self.combo_image.configure(values=[])
                self.combo_image.set("")
        else:
            filename = os.path.basename(self.image_files[self.current_idx])
            self.lbl_filename.config(text=filename)
            self.lbl_progress.config(
                text=f"{self.current_idx + 1} / {len(self.image_files)}"
            )
            self.refresh_image_dropdown()
        
        self.lbl_box_count.config(
            text=f"{LANG_MAP[self.lang]['boxes']}: {len(self.rects)}"
        )

    def refresh_image_dropdown(self):
        if not hasattr(self, "combo_image"):
            return
        names = [os.path.basename(p) for p in self.image_files]
        self.combo_image.configure(values=names)
        if not names:
            self.combo_image.set("")
            return
        if 0 <= self.current_idx < len(names):
            self.combo_image.set(names[self.current_idx])

    def on_image_selected(self, e: Any = None) -> None:
        if not self.image_files:
            return
        idx = self.combo_image.current()
        if idx < 0 or idx == self.current_idx:
            return
        self.save_current()
        self.current_idx = idx
        self.load_img()

    def _register_model_path(self, model_path: str) -> None:
        path = model_path.strip()
        if not path:
            return
        if path not in self.model_library:
            self.model_library.append(path)
        self._refresh_model_dropdown()

    def _refresh_model_dropdown(self) -> None:
        if hasattr(self, "combo_model_path"):
            self.combo_model_path.configure(values=self.model_library)

    def save_session_state(self) -> None:
        state = SessionState(
            project_root=self.project_root,
            split=self.current_split,
            image_name="",
            detection_model_mode=self.det_model_mode.get(),
            detection_model_path=self.yolo_path.get().strip(),
        )
        if self.image_files and 0 <= self.current_idx < len(self.image_files):
            state.image_name = os.path.basename(self.image_files[self.current_idx])
        try:
            atomic_write_json(self.session_path, state.__dict__)
        except Exception:
            self.logger.exception("Failed to save session state")

    def load_session_state(self) -> None:
        if not os.path.exists(self.session_path):
            return
        try:
            with open(self.session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.logger.exception("Failed to load session state")
            return

        project_root = data.get("project_root", "")
        split = data.get("split", "train")
        image_name = data.get("image_name", "")
        model_mode = data.get("detection_model_mode", "Official YOLO26m.pt (Bundled)")
        model_path = data.get("detection_model_path", self.config.yolo_model_path)

        if not project_root or not os.path.exists(project_root):
            return

        if split not in ["train", "val", "test"]:
            split = "train"
        if model_mode not in {
            "Official YOLO26m.pt (Bundled)",
            "Custom YOLO (v5/v7/v8/v9/v11/v26)",
            "Custom RF-DETR",
        }:
            model_mode = "Official YOLO26m.pt (Bundled)"
        self.det_model_mode.set(model_mode)
        if model_path:
            self.yolo_path.set(model_path)
            self._register_model_path(model_path)
        self.current_split = split
        self.combo_split.set(split)
        self.load_project_from_path(project_root, preferred_image=image_name, save_session=False)

    def on_detection_model_mode_changed(self, e: Any = None) -> None:
        if self.det_model_mode.get() == "Official YOLO26m.pt (Bundled)":
            self.yolo_path.set(self.config.yolo_model_path)
            self._register_model_path(self.config.yolo_model_path)
        self.yolo_model = None
        self._loaded_model_key = None

    def use_official_yolo26n(self) -> None:
        self.det_model_mode.set("Official YOLO26m.pt (Bundled)")
        self.yolo_path.set(self.config.yolo_model_path)
        self._register_model_path(self.config.yolo_model_path)
        self.yolo_model = None
        self._loaded_model_key = None

    def _resolve_official_model_path(self) -> str:
        candidates: list[str] = []
        try:
            packaged = resources.files("ai_labeller").joinpath("models", self.config.yolo_model_path)
            if packaged.is_file():
                return str(packaged)
        except Exception:
            self.logger.exception("Failed to resolve packaged official model path")

        candidates.append(self.config.yolo_model_path)
        candidates.append(os.path.join(os.path.dirname(__file__), "models", self.config.yolo_model_path))
        candidates.append(os.path.join(os.getcwd(), self.config.yolo_model_path))

        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            f"Official model not found: {self.config.yolo_model_path}. "
            "Please reinstall package or choose a custom model."
        )

    def browse_detection_model(self) -> None:
        self.pick_model_file()

    def pick_model_file(self, forced_mode: str | None = None) -> bool:
        self.logger.info("Opening model file dialog (mode=%s)", forced_mode or "auto")
        model_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select model",
            filetypes=[
                ("Model files", "*.pt *.onnx"),
                ("PyTorch", "*.pt"),
                ("ONNX", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if not model_path:
            self.logger.info("Model selection cancelled")
            return False
        model_path = os.path.abspath(model_path)
        if not os.path.isfile(model_path):
            self.logger.error("Selected model file not found: %s", model_path)
            messagebox.showerror("Model Error", f"Model file not found:\n{model_path}")
            return False
        if not model_path.lower().endswith((".pt", ".onnx")):
            proceed = messagebox.askyesno(
                "Model Warning",
                f"Selected file may not be a YOLO model:\n{os.path.basename(model_path)}\n\nContinue?",
            )
            if not proceed:
                self.logger.info("Model selection rejected by user due to extension warning")
                return False
        self.yolo_path.set(model_path)
        self._register_model_path(model_path)
        if forced_mode:
            self.det_model_mode.set(forced_mode)
        elif "rfdetr" in os.path.basename(model_path).lower():
            self.det_model_mode.set("Custom RF-DETR")
        else:
            self.det_model_mode.set("Custom YOLO (v5/v7/v8/v9/v11/v26)")
        self.yolo_model = None
        self._loaded_model_key = None
        self.save_session_state()
        self.logger.info("Model selected: %s (%s)", model_path, self.det_model_mode.get())
        return True

    def show_startup_source_dialog(self, force: bool = False, reason: str | None = None) -> None:
        if self._startup_dialog_open:
            return
        if not force and getattr(self, "_startup_dialog_shown", False):
            return
        self._startup_dialog_shown = True
        self._startup_dialog_open = True
        if reason:
            self.logger.info("Showing startup source dialog: %s", reason)

        win = tk.Toplevel(self.root)
        win.title(LANG_MAP[self.lang]["startup_choose_source"])
        win.geometry("420x240")
        win.resizable(False, False)
        win.configure(bg=COLORS["bg_light"])
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text=LANG_MAP[self.lang]["startup_prompt"],
            bg=COLORS["bg_light"],
            fg=COLORS["text_primary"],
            font=self.font_bold,
            anchor="w",
        ).pack(fill="x", padx=20, pady=(20, 14))

        def choose_images() -> None:
            self._close_startup_dialog(win)
            self.root.after(1, lambda: self.startup_choose_images_folder("images"))

        def choose_yolo() -> None:
            self._close_startup_dialog(win)
            self.det_model_mode.set("Custom YOLO (v5/v7/v8/v9/v11/v26)")
            self.root.after(1, lambda: self.startup_choose_images_folder("yolo"))

        def choose_rfdetr() -> None:
            self._close_startup_dialog(win)
            self.det_model_mode.set("Custom RF-DETR")
            self.root.after(1, lambda: self.startup_choose_images_folder("rfdetr"))

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang]["startup_images"],
            command=choose_images,
            bg=COLORS["primary"],
        ).pack(fill="x", padx=20, pady=(0, 10))

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang]["startup_yolo"],
            command=choose_yolo,
            bg=COLORS["success"],
        ).pack(fill="x", padx=20, pady=(0, 10))

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang]["startup_rfdetr"],
            command=choose_rfdetr,
            bg=COLORS["warning"],
        ).pack(fill="x", padx=20, pady=(0, 10))

        self.create_secondary_button(
            win,
            text=LANG_MAP[self.lang]["startup_skip"],
            command=lambda: self._close_startup_dialog(win),
        ).pack(fill="x", padx=20, pady=(0, 18))
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_startup_dialog(win))

    def _close_startup_dialog(self, win: tk.Toplevel) -> None:
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        self._startup_dialog_open = False
        self.logger.info("Startup source dialog closed")

    def _choose_model_then_images(self, mode: str) -> None:
        try:
            ok = self.pick_model_file(mode)
            if not ok:
                use_images_only = messagebox.askyesno(
                    LANG_MAP[self.lang].get("startup_model_cancel_title", "Model Selection Cancelled"),
                    LANG_MAP[self.lang].get(
                        "startup_model_cancel_msg",
                        "No model selected. Continue with images folder only?",
                    ),
                    parent=self.root,
                )
                if use_images_only:
                    self.root.after(120, self.startup_choose_images_folder)
                else:
                    self.root.after(120, lambda: self.show_startup_source_dialog(force=True, reason="model selection cancelled"))
                return
            model_path = self.yolo_path.get().strip()
            if not model_path or not os.path.isfile(model_path):
                messagebox.showerror("Model Error", "Invalid model file selected. Please try again.")
                self.root.after(120, lambda: self.show_startup_source_dialog(force=True, reason="invalid model path"))
                return
            # Open on next idle tick + short delay to avoid native-dialog focus races on Windows.
            self.root.after_idle(lambda: self.root.after(180, self.startup_choose_images_folder))
        except Exception:
            self.logger.exception("Error while selecting model and folder")
            messagebox.showerror("Error", "Failed during model selection.")
            self.root.after(120, lambda: self.show_startup_source_dialog(force=True, reason="selection error"))

    def return_to_source_select(self, e: Any = None) -> None:
        if self.project_root and self.image_files:
            try:
                self.save_current()
            except Exception:
                self.logger.exception("Failed to save before returning to source selector")
        self.show_startup_source_dialog(force=True)

    def startup_choose_images_folder(self, source_mode: str = "images") -> None:
        if self._folder_dialog_open:
            return
        self._folder_dialog_open = True
        self.logger.info("=== startup_choose_images_folder START (%s) ===", source_mode)
        try:
            directory = filedialog.askdirectory(
                parent=self.root,
                title=LANG_MAP[self.lang].get("pick_folder_title", "Select Folder")
            )
            if not directory:
                self.logger.info("Folder selection cancelled")
                return
            directory = os.path.abspath(directory)
            self.logger.info("Selected directory: %s", directory)

            diag = self.diagnose_folder_structure(directory)
            self.logger.info("Folder diagnosis: %s", diag)
            if not diag["is_yolo_project"] and diag["flat_images"] == 0:
                self.show_folder_diagnosis(directory)
                return

            root_dir = self.normalize_project_root(directory)
            yolo_root = self.find_yolo_project_root(root_dir)

            try:
                if yolo_root:
                    self.load_project_from_path(yolo_root)
                    if self.image_files:
                        self.root.lift()
                        self.root.focus_force()
                        messagebox.showinfo(
                            LANG_MAP[self.lang]["title"],
                            LANG_MAP[self.lang].get(
                                "loaded_from",
                                "Loaded {count} images\nFrom: {path}\nSplit: {split}",
                            ).format(
                                count=len(self.image_files),
                                path=yolo_root,
                                split=self.current_split,
                            ),
                            parent=self.root,
                        )
                    else:
                        self.show_folder_diagnosis(yolo_root)
                    return

                self.load_images_folder_only(root_dir)
                if self.image_files:
                    self.root.lift()
                    self.root.focus_force()
                    messagebox.showinfo(
                        LANG_MAP[self.lang]["title"],
                        LANG_MAP[self.lang].get(
                            "loaded_from",
                            "Loaded {count} images\nFrom: {path}\nSplit: {split}",
                        ).format(
                            count=len(self.image_files),
                            path=root_dir,
                            split=self.current_split,
                        ),
                        parent=self.root,
                    )
                else:
                    self.show_folder_diagnosis(root_dir)
            except Exception as exc:
                self.logger.exception("Failed to load selected folder: %s", directory)
                self.root.lift()
                self.root.focus_force()
                messagebox.showerror(
                    "Error",
                    f"Failed to load folder:\n{directory}\n\nError: {exc}",
                    parent=self.root,
                )
        finally:
            self._folder_dialog_open = False
            self.logger.info("=== startup_choose_images_folder END ===")

    def normalize_project_root(self, directory: str) -> str:
        root_dir = directory.replace("\\", "/").rstrip("/")
        base = os.path.basename(root_dir).lower()
        parent = os.path.dirname(root_dir).replace("\\", "/")
        parent_base = os.path.basename(parent).lower()
        grandparent = os.path.dirname(parent).replace("\\", "/")

        if base in {"train", "val", "test"} and parent_base == "images":
            if os.path.exists(f"{grandparent}/labels"):
                return grandparent
        if base == "images" and os.path.exists(f"{parent}/labels"):
            return parent
        if base == "labels" and os.path.exists(f"{parent}/images"):
            return parent
        return root_dir

    def find_yolo_project_root(self, directory: str) -> str | None:
        root = directory.replace("\\", "/").rstrip("/")
        if not root:
            return None

        def is_yolo_root(candidate: str) -> bool:
            return any(
                os.path.exists(f"{candidate}/images/{split}")
                for split in ("train", "val", "test")
            )

        # Try selected folder and a few parent levels first.
        candidates = [root]
        parent = os.path.dirname(root).replace("\\", "/")
        if parent and parent not in candidates:
            candidates.append(parent)
        grandparent = os.path.dirname(parent).replace("\\", "/") if parent else ""
        if grandparent and grandparent not in candidates:
            candidates.append(grandparent)

        for candidate in candidates:
            if is_yolo_root(candidate):
                return candidate

        # If user selected an upper folder, scan one level of child directories.
        try:
            for child in os.listdir(root):
                child_path = os.path.join(root, child).replace("\\", "/")
                if os.path.isdir(child_path) and is_yolo_root(child_path):
                    return child_path
        except Exception:
            pass

        return None

    def _list_split_images_for_root(self, project_root: str, split: str) -> list[str]:
        img_path = f"{project_root}/images/{split}"
        return sorted([
            f for f in glob.glob(f"{img_path}/*.*")
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def _existing_image_splits(self, project_root: str) -> list[str]:
        splits: list[str] = []
        for split in ("train", "val", "test"):
            if os.path.isdir(f"{project_root}/images/{split}"):
                splits.append(split)
        return splits

    def ensure_yolo_label_dirs(self, project_root: str) -> None:
        """Ensure labels/<split> exists for every existing images/<split>."""
        splits = self._existing_image_splits(project_root)
        if not splits:
            return
        for split in splits:
            os.makedirs(f"{project_root}/labels/{split}", exist_ok=True)

    def diagnose_folder_structure(self, directory: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "is_yolo_project": False,
            "has_images_folder": False,
            "has_labels_folder": False,
            "splits_found": [],
            "total_images": 0,
            "images_by_split": {},
            "flat_images": 0,
            "errors": [],
        }
        try:
            images_path = os.path.join(directory, "images")
            labels_path = os.path.join(directory, "labels")
            result["has_images_folder"] = os.path.isdir(images_path)
            result["has_labels_folder"] = os.path.isdir(labels_path)

            if result["has_images_folder"]:
                for split in ("train", "val", "test"):
                    split_path = os.path.join(images_path, split)
                    if not os.path.isdir(split_path):
                        continue
                    files = [
                        f for f in os.listdir(split_path)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    if files:
                        result["splits_found"].append(split)
                        result["images_by_split"][split] = len(files)
                        result["total_images"] += len(files)
                result["is_yolo_project"] = len(result["splits_found"]) > 0

            root_images = [
                f for f in os.listdir(directory)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            result["flat_images"] = len(root_images)
        except Exception as exc:
            result["errors"].append(str(exc))
        return result

    def show_folder_diagnosis(self, directory: str) -> None:
        diag = self.diagnose_folder_structure(directory)
        lines = [f"Folder: {directory}", ""]
        if diag["is_yolo_project"]:
            lines.append("YOLO project structure detected.")
            lines.append(f"Splits: {', '.join(diag['splits_found'])}")
            for split in ("train", "val", "test"):
                if split in diag["images_by_split"]:
                    lines.append(f"- {split}: {diag['images_by_split'][split]} images")
            lines.append(f"Total images: {diag['total_images']}")
        elif diag["flat_images"] > 0:
            lines.append("Flat image folder detected.")
            lines.append(f"Images found: {diag['flat_images']}")
        else:
            lines.append("No supported images found.")
            lines.append("Expected either:")
            lines.append("- folder/images/train|val|test/*.jpg")
            lines.append("- folder/*.jpg")
        if diag["errors"]:
            lines.append("")
            lines.append("Errors:")
            for err in diag["errors"]:
                lines.append(f"- {err}")
        self.root.lift()
        self.root.focus_force()
        messagebox.showwarning("Folder Diagnosis", "\n".join(lines), parent=self.root)

    def load_images_folder_only(self, directory: str) -> None:
        self.project_root = directory
        self.current_split = "train"
        self.combo_split.set(self.current_split)

        self.image_files = sorted([
            f for f in glob.glob(f"{self.project_root}/*.*")
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        os.makedirs(f"{self.project_root}/labels/{self.current_split}", exist_ok=True)

        if not self.image_files:
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang]["no_img"],
            )
            self.current_idx = 0
            self.img_pil = None
            self.img_tk = None
            self.rects = []
            self.update_info_text()
            self.render()
            self.save_session_state()
            return

        self.current_idx = 0
        self.load_img()

    def on_app_close(self) -> None:
        try:
            self.save_current()
        except Exception:
            self.logger.exception("Failed while saving on close")
        if self.img_pil is not None:
            self.img_pil.close()
            self.img_pil = None
        self.save_session_state()
        self.root.destroy()

    def render(self) -> None:
        """Redraw canvas image, boxes, guides, and overlays."""
        self.canvas.delete("all")
        self._cursor_line_x = None
        self._cursor_line_y = None
        self._cursor_text_id = None
        self._cursor_bg_id = None

        if not self.img_pil:
            return
        
        # 1. ???????????
        w = int(self.img_pil.width * self.scale)
        h = int(self.img_pil.height * self.scale)
        self.img_tk = ImageTk.PhotoImage(
            self.img_pil.resize((w, h), Image.Resampling.NEAREST)
        )
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            image=self.img_tk,
            anchor="nw"
        )
        
        # 2. ????????????
        box_colors = [
            COLORS["box_1"], COLORS["box_2"], COLORS["box_3"],
            COLORS["box_4"], COLORS["box_5"], COLORS["box_6"]
        ]
        
        for i, rect in enumerate(self.rects):
            x1, y1 = self.img_to_canvas(rect[0], rect[1])
            x2, y2 = self.img_to_canvas(rect[2], rect[3])
            
            is_selected = (i == self.selected_idx)
            color = COLORS["box_selected"] if is_selected else box_colors[rect[4] % len(box_colors)]
            width = 3 if is_selected else 2
            
            # ???????????????
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=width
            )
            
            # ????????????????????
            if is_selected:
                for hx, hy in self.get_handles(rect):
                    cx, cy = self.img_to_canvas(hx, hy)
                    self.canvas.create_oval(
                        cx - self.HANDLE_SIZE,
                        cy - self.HANDLE_SIZE,
                        cx + self.HANDLE_SIZE,
                        cy + self.HANDLE_SIZE,
                        fill=COLORS["bg_white"],
                        outline=color,
                        width=2
                    )
            
            if self.show_all_labels:
                # ?????????????????
                class_name = (
                    self.class_names[rect[4]]
                    if rect[4] < len(self.class_names)
                    else f"ID:{rect[4]}"
                )
                
                # Draw a new box???????????????????????
                label_y = max(y1 - 24, 8)  # Draw a new box???????
                
                # ???????????
                text_id = self.canvas.create_text(
                    x1 + 8,
                    label_y + 4,
                    text=class_name,
                    fill=COLORS["text_white"],
                    font=self.font_primary,
                    anchor="nw"
                )
                
                bbox = self.canvas.bbox(text_id)
                if bbox:
                    padding = 4
                    bg_id = self.canvas.create_rectangle(
                        bbox[0] - padding,
                        bbox[1] - padding,
                        bbox[2] + padding,
                        bbox[3] + padding,
                        fill=color,
                        outline=""
                    )
                    self.canvas.tag_lower(bg_id, text_id)
        
        # 3. Draw temporary rectangle while dragging
        if self.temp_rect_coords:
            cx, cy, ex, ey = self.temp_rect_coords
            self.canvas.create_rectangle(
                cx, cy, ex, ey,
                outline=COLORS["primary_light"],
                width=2,
                dash=(6, 4)
            )
        
        # 4. Update cursor overlay
        self.update_cursor_overlay()
        
        # Refresh side info labels
        self.update_info_text()
    
    # ==================== Mouse Interaction ====================
    
    def on_mouse_move(self, e: Any) -> None:
        """Track cursor and refresh overlay."""
        self.mouse_pos = (e.x, e.y)
        self.update_cursor_overlay()
    
    def update_cursor_overlay(self) -> None:
        if not self.canvas:
            return

        mx, my = self.mouse_pos
        canvas_h = self.canvas.winfo_height()
        canvas_w = self.canvas.winfo_width()

        if self._cursor_line_x is None:
            self._cursor_line_x = self.canvas.create_line(
                mx, 0, mx, canvas_h, fill=COLORS["primary"], width=1, dash=(2, 4), tags="cursor_overlay"
            )
        else:
            self.canvas.coords(self._cursor_line_x, mx, 0, mx, canvas_h)

        if self._cursor_line_y is None:
            self._cursor_line_y = self.canvas.create_line(
                0, my, canvas_w, my, fill=COLORS["primary"], width=1, dash=(2, 4), tags="cursor_overlay"
            )
        else:
            self.canvas.coords(self._cursor_line_y, 0, my, canvas_w, my)

        coord_text = f"{mx}, {my}"
        if self._cursor_text_id is None:
            self._cursor_text_id = self.canvas.create_text(
                mx + 12,
                my - 12,
                text=coord_text,
                fill=COLORS["text_primary"] if self.theme == "light" else COLORS["text_white"],
                font=self.font_mono,
                anchor="nw",
                tags="cursor_overlay",
            )
        else:
            self.canvas.coords(self._cursor_text_id, mx + 12, my - 12)
            self.canvas.itemconfig(self._cursor_text_id, text=coord_text)

        coord_bbox = self.canvas.bbox(self._cursor_text_id)
        if not coord_bbox:
            return

        padding = 4
        bx1 = coord_bbox[0] - padding
        by1 = coord_bbox[1] - padding
        bx2 = coord_bbox[2] + padding
        by2 = coord_bbox[3] + padding
        if self._cursor_bg_id is None:
            self._cursor_bg_id = self.canvas.create_rectangle(
                bx1,
                by1,
                bx2,
                by2,
                fill=COLORS["bg_dark"],
                outline=COLORS["primary"],
                width=1,
                tags="cursor_overlay",
            )
        else:
            self.canvas.coords(self._cursor_bg_id, bx1, by1, bx2, by2)
        self.canvas.tag_lower(self._cursor_bg_id, self._cursor_text_id)

    def on_mouse_down(self, e):
        """Handle mouse press for select, move, resize, or create box."""
        if not self.img_pil:
            return
        
        ix, iy = self.canvas_to_img(e.x, e.y)
        
        # Check if one of the resize handles is selected
        if self.selected_idx is not None:
            for i, (hx, hy) in enumerate(self.get_handles(self.rects[self.selected_idx])):
                dist = np.sqrt((ix - hx) ** 2 + (iy - hy) ** 2) * self.scale
                if dist < self.config.mouse_handle_hit_radius_px:
                    self.active_handle = i
                    self.drag_start = (ix, iy)
                    self.push_history()
                    return
        
        # Check if pointer is inside an existing box
        clicked_idx = None
        for i, rect in enumerate(self.rects):
            if (min(rect[0], rect[2]) < ix < max(rect[0], rect[2]) and
                min(rect[1], rect[3]) < iy < max(rect[1], rect[3])):
                clicked_idx = i
        
        if clicked_idx is not None:
            self.selected_idx = clicked_idx
            self.is_moving_box = True
            self.drag_start = (ix, iy)
            self.combo_cls.current(self.rects[clicked_idx][4])
            self.push_history()
        else:
            self.selected_idx = None
            self.drag_start = (ix, iy)
            self.temp_rect_coords = (e.x, e.y, e.x, e.y)
        
        self.render()
    
    def on_mouse_drag(self, e):
        """???"""
        self.mouse_pos = (e.x, e.y)
        
        if not self.img_pil or not self.drag_start:
            self.render()
            return
        
        ix, iy = self.canvas_to_img(e.x, e.y)
        W, H = self.img_pil.width, self.img_pil.height
        
        # Draw a new box????
        ix = max(0, min(W, ix))
        iy = max(0, min(H, iy))
        
        if self.selected_idx is not None and self.active_handle is not None:
            # ???????????
            rect = self.rects[self.selected_idx]
            if self.active_handle in [0, 6, 7]:
                rect[0] = ix
            if self.active_handle in [0, 1, 2]:
                rect[1] = iy
            if self.active_handle in [2, 3, 4]:
                rect[2] = ix
            if self.active_handle in [4, 5, 6]:
                rect[3] = iy
        
        elif self.is_moving_box:
            # Move selected box
            dx = ix - self.drag_start[0]
            dy = iy - self.drag_start[1]
            rect = self.rects[self.selected_idx]
            
            # Keep box inside image bounds
            if rect[0] + dx < 0:
                dx = -rect[0]
            if rect[2] + dx > W:
                dx = W - rect[2]
            if rect[1] + dy < 0:
                dy = -rect[1]
            if rect[3] + dy > H:
                dy = H - rect[3]
            
            rect[0] += dx
            rect[1] += dy
            rect[2] += dx
            rect[3] += dy
            self.drag_start = (ix, iy)
        
        else:
            # Draw a new box
            if self.temp_rect_coords:
                self.temp_rect_coords = (
                    self.temp_rect_coords[0],
                    self.temp_rect_coords[1],
                    e.x,
                    e.y
                )
        
        self.render()
    
    def on_mouse_up(self, e):
        """?????"""
        if self.temp_rect_coords:
            ix, iy = self.canvas_to_img(e.x, e.y)
            new_box = self.clamp_box([
                self.drag_start[0],
                self.drag_start[1],
                ix,
                iy,
                self.combo_cls.current()
            ])
            
            # Keep minimum box size threshold
            if (new_box[2] - new_box[0]) > 2 and (new_box[3] - new_box[1]) > 2:
                self.push_history()
                self.rects.append(new_box)
            
            self.temp_rect_coords = None
        
        if self.selected_idx is not None:
            self.rects[self.selected_idx] = self.clamp_box(
                self.rects[self.selected_idx]
            )
        
        self.is_moving_box = False
        self.active_handle = None
        self.render()
    
    def on_zoom(self, e):
        """Zoom image around mouse pointer."""
        factor = self.config.zoom_in_factor if e.delta > 0 else self.config.zoom_out_factor
        
        self.offset_x = e.x - (e.x - self.offset_x) * factor
        self.offset_y = e.y - (e.y - self.offset_y) * factor
        self.scale *= factor
        
        self.render()
    
    # ==================== ???? ====================
    
    def on_class_change_request(self, e=None):
        """???"""
        if self.selected_idx is not None:
            new_cid = self.combo_cls.current()
            if self.rects[self.selected_idx][4] != new_cid:
                self.push_history()
                self.rects[self.selected_idx][4] = new_cid
                self.render()
    
    def edit_classes_table(self):
        """??????????????"""
        L = LANG_MAP[self.lang]
        
        win = tk.Toplevel(self.root)
        win.title(L["edit_classes"])
        win.geometry("500x600")
        win.configure(bg=COLORS["bg_light"])
        
        # TreeView
        tree = ttk.Treeview(
            win,
            columns=("id", "name"),
            show="headings",
            height=15
        )
        tree.heading("id", text="ID")
        tree.column("id", width=80, anchor="center")
        tree.heading("name", text=L["class_name"])
        tree.column("name", width=300)
        tree.pack(fill="both", expand=True, padx=20, pady=20)
        
        def refresh():
            for item in tree.get_children():
                tree.delete(item)
            for i, name in enumerate(self.class_names):
                tree.insert("", "end", values=(i, name))
        
        def rename():
            sel = tree.selection()
            if not sel:
                return
            idx = int(tree.item(sel[0])['values'][0])
            new_name = simpledialog.askstring(
                L["rename"],
                L["rename_prompt"].format(name=self.class_names[idx]),
                initialvalue=self.class_names[idx]
            )
            if new_name:
                self.class_names[idx] = new_name
                refresh()
        
        def add():
            new_name = simpledialog.askstring(L["add"], L["add_prompt"])
            if new_name:
                self.class_names.append(new_name)
                refresh()

        def on_double_click(e):
            row = tree.identify_row(e.y)
            if row:
                tree.selection_set(row)
                rename()
        
        # ??
        btn_frame = tk.Frame(win, bg=COLORS["bg_light"])
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Button(
            btn_frame,
            text=L["add"],
            command=add,
            bg=COLORS["success"],
            fg=COLORS["text_white"],
            font=self.font_primary,
            relief="flat",
            padx=20,
            pady=10
        ).pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        tk.Button(
            btn_frame,
            text=L["rename"],
            command=rename,
            bg=COLORS["warning"],
            fg=COLORS["text_white"],
            font=self.font_primary,
            relief="flat",
            padx=20,
            pady=10
        ).pack(side="left", expand=True, fill="x", padx=(5, 0))
        
        tk.Button(
            win,
            text=L["apply"],
            command=lambda: [
                self.combo_cls.config(values=self.class_names),
                self.render(),
                win.destroy()
            ],
            bg=COLORS["primary"],
            fg=COLORS["text_white"],
            font=self.font_bold,
            relief="flat",
            pady=12
        ).pack(fill="x", padx=20, pady=(0, 20))
        
        tree.bind("<Double-1>", on_double_click)
        refresh()

    def reassign_labeled_class(self):
        """????????????????"""
        if self.selected_idx is None:
            messagebox.showinfo(LANG_MAP[self.lang]["class_mgmt"], LANG_MAP[self.lang]["no_label_selected"])
            return
        if not self.class_names:
            messagebox.showinfo(LANG_MAP[self.lang]["class_mgmt"], LANG_MAP[self.lang]["no_classes_available"])
            return

        win = tk.Toplevel(self.root)
        win.title(LANG_MAP[self.lang]["reassign_class"])
        win.geometry("420x220")
        win.configure(bg=COLORS["bg_light"])

        current_idx = self.rects[self.selected_idx][4]
        current_name = (
            self.class_names[current_idx]
            if current_idx < len(self.class_names)
            else str(current_idx)
        )

        tk.Label(
            win,
            text=f"{LANG_MAP[self.lang]['current']}: {current_name}",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_light"],
            anchor="w"
        ).pack(fill="x", padx=20, pady=(20, 6))

        tk.Label(
            win,
            text=LANG_MAP[self.lang]["to"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_light"],
            anchor="w"
        ).pack(fill="x", padx=20, pady=(10, 6))

        to_default = self.class_names[current_idx] if current_idx < len(self.class_names) else self.class_names[0]
        to_var = tk.StringVar(value=to_default)
        ttk.Combobox(
            win,
            values=self.class_names,
            textvariable=to_var,
            state="readonly",
            font=self.font_primary
        ).pack(fill="x", padx=20)

        def apply_change():
            to_name = to_var.get()
            try:
                to_idx = self.class_names.index(to_name)
            except ValueError:
                win.destroy()
                return

            if self.rects[self.selected_idx][4] == to_idx:
                win.destroy()
                return

            self.push_history()
            self.rects[self.selected_idx][4] = to_idx
            self.combo_cls.current(to_idx)
            self.render()
            win.destroy()

        tk.Button(
            win,
            text=LANG_MAP[self.lang]["apply"],
            command=apply_change,
            bg=COLORS["primary"],
            fg=COLORS["text_white"],
            font=self.font_bold,
            relief="flat",
            pady=10
        ).pack(fill="x", padx=20, pady=(20, 16))

    def clear_current_labels(self):
        """???????????"""
        if not self.rects:
            return
        self.push_history()
        self.rects = []
        self.selected_idx = None
        self.active_handle = None
        self.is_moving_box = False
        self.drag_start = None
        self.temp_rect_coords = None
        self.render()

    
    def _build_removed_path(self, kind, src_path):
        ext = os.path.splitext(src_path)[1]
        base = os.path.splitext(os.path.basename(src_path))[0]
        dst_dir = os.path.join(self.project_root, "removed", self.current_split, kind)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"{base}{ext}")
        if not os.path.exists(dst_path):
            return dst_path

        i = 1
        while True:
            candidate = os.path.join(dst_dir, f"{base}_{i}{ext}")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def _unique_target_path(self, target_path):
        if not os.path.exists(target_path):
            return target_path
        folder = os.path.dirname(target_path)
        base, ext = os.path.splitext(os.path.basename(target_path))
        i = 1
        while True:
            candidate = os.path.join(folder, f"{base}_{i}{ext}")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def remove_current_from_split(self):
        if not self.image_files:
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("remove_none", "No image to remove.")
            )
            return
        if not self.project_root:
            return

        confirm_msg = LANG_MAP[self.lang].get(
            "remove_confirm",
            "Remove current image from {split}?"
        ).format(split=self.current_split)
        if not messagebox.askyesno(LANG_MAP[self.lang]["title"], confirm_msg):
            return

        image_path = self.image_files[self.current_idx]
        image_name = os.path.basename(image_path)
        base = os.path.splitext(image_name)[0]

        try:
            moved_image_path = self._build_removed_path("images", image_path)
            shutil.move(image_path, moved_image_path)
            label_dir = os.path.join(self.project_root, "labels", self.current_split)
            for ext in (".txt", ".json"):
                label_path = os.path.join(label_dir, f"{base}{ext}")
                if os.path.exists(label_path):
                    moved_label_path = self._build_removed_path("labels", label_path)
                    shutil.move(label_path, moved_label_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        del self.image_files[self.current_idx]
        if self.current_idx >= len(self.image_files):
            self.current_idx = max(0, len(self.image_files) - 1)

        if self.image_files:
            self.load_img()
        else:
            self.img_pil = None
            self.img_tk = None
            self.rects = []
            self.history_manager.clear()
            self.selected_idx = None
            self.active_handle = None
            self.is_moving_box = False
            self.drag_start = None
            self.temp_rect_coords = None
            self.update_info_text()
            self.render()

        self.save_session_state()
        done_msg = LANG_MAP[self.lang].get("remove_done", "Removed: {name}").format(name=image_name)
        messagebox.showinfo(LANG_MAP[self.lang]["title"], done_msg)

    def open_restore_removed_dialog(self):
        if not self.project_root:
            return

        removed_img_dir = os.path.join(self.project_root, "removed", self.current_split, "images")
        if not os.path.isdir(removed_img_dir):
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("restore_none", "No removed frame found in this split.")
            )
            return

        removed_files = sorted([
            f for f in os.listdir(removed_img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not removed_files:
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("restore_none", "No removed frame found in this split.")
            )
            return

        win = tk.Toplevel(self.root)
        win.title(LANG_MAP[self.lang].get("restore_title", "Restore Deleted Frame"))
        win.geometry("520x420")
        win.configure(bg=COLORS["bg_light"])

        tk.Label(
            win,
            text=LANG_MAP[self.lang].get("restore_select", "Select a frame to restore:"),
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_light"],
            anchor="w"
        ).pack(fill="x", padx=16, pady=(16, 8))

        list_wrap = tk.Frame(win, bg=COLORS["bg_light"])
        list_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        lb = tk.Listbox(
            list_wrap,
            font=self.font_mono,
            activestyle="none",
            selectmode="browse"
        )
        for name in removed_files:
            lb.insert("end", name)
        lb.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        def do_restore():
            sel = lb.curselection()
            if not sel:
                return
            filename = lb.get(sel[0])
            self.restore_removed_file_by_name(filename)
            win.destroy()

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang].get("restore_from_split", "Restore Deleted Frame"),
            command=do_restore,
            bg=COLORS["success"]
        ).pack(fill="x", padx=16, pady=(0, 16))

    def restore_removed_file_by_name(self, filename):
        removed_img_path = os.path.join(self.project_root, "removed", self.current_split, "images", filename)
        if not os.path.exists(removed_img_path):
            return

        split_img_dir = os.path.join(self.project_root, "images", self.current_split)
        os.makedirs(split_img_dir, exist_ok=True)
        target_img_path = self._unique_target_path(os.path.join(split_img_dir, filename))

        base = os.path.splitext(filename)[0]
        removed_lbl_dir = os.path.join(self.project_root, "removed", self.current_split, "labels")
        split_lbl_dir = os.path.join(self.project_root, "labels", self.current_split)
        os.makedirs(split_lbl_dir, exist_ok=True)

        if self.image_files and self.img_pil:
            self.save_current()

        try:
            shutil.move(removed_img_path, target_img_path)
            for ext in (".txt", ".json"):
                removed_lbl_path = os.path.join(removed_lbl_dir, f"{base}{ext}")
                if os.path.exists(removed_lbl_path):
                    target_lbl_path = self._unique_target_path(
                        os.path.join(split_lbl_dir, f"{base}{ext}")
                    )
                    shutil.move(removed_lbl_path, target_lbl_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.load_split_data()
        if target_img_path in self.image_files:
            self.current_idx = self.image_files.index(target_img_path)
            self.load_img()

        msg = LANG_MAP[self.lang].get("restore_done", "Restored: {name}").format(name=os.path.basename(target_img_path))
        messagebox.showinfo(LANG_MAP[self.lang]["title"], msg)

    def load_img(self) -> None:
        """????????"""
        if not self.image_files:
            return
        
        path = self.image_files[self.current_idx]
        self.update_info_text()
        if self.img_pil is not None:
            self.img_pil.close()
            self.img_pil = None
        try:
            self.img_pil = Image.open(path)
        except Exception:
            self.logger.exception("Failed to load image: %s", path)
            messagebox.showerror("Error", f"Failed to open image:\n{path}")
            return
        
        prev_rects = copy.deepcopy(self.rects)
        self.rects = []
        self.history_manager.clear()
        self.selected_idx = None
        self.active_handle = None
        self.is_moving_box = False
        self.drag_start = None
        self.temp_rect_coords = None
        
        # ?????
        base = os.path.splitext(os.path.basename(path))[0]
        label_path = f"{self.project_root}/labels/{self.current_split}/{base}.txt"
        
        label_exists = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        if label_exists:
            W, H = self.img_pil.width, self.img_pil.height
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 5:
                            c, cx, cy, w, h = map(float, parts)
                            self.rects.append([
                                (cx - w / 2) * W,
                                (cy - h / 2) * H,
                                (cx + w / 2) * W,
                                (cy + h / 2) * H,
                                int(c)
                            ])
            except Exception:
                self.logger.exception("Failed to parse label file: %s", label_path)
                messagebox.showerror("Error", f"Failed to read label file: {label_path}")
                self.rects = []
        else:
            if self.var_propagate.get():
                self.rects = prev_rects
            if self.var_auto_yolo.get():
                self.run_yolo_detection()
        
        self.fit_image_to_canvas()
        self.save_session_state()
    
    def save_current(self) -> None:
        """?????????"""
        if not self.project_root or not self.img_pil:
            return
        
        path = self.image_files[self.current_idx]
        base = os.path.splitext(os.path.basename(path))[0]
        W, H = self.img_pil.width, self.img_pil.height
        
        label_path = f"{self.project_root}/labels/{self.current_split}/{base}.txt"
        
        lines = []
        for rect in self.rects:
            cx = (rect[0] + rect[2]) / 2 / W
            cy = (rect[1] + rect[3]) / 2 / H
            w = (rect[2] - rect[0]) / W
            h = (rect[3] - rect[1]) / H
            lines.append(f"{rect[4]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if not lines:
            if os.path.exists(label_path):
                try:
                    os.remove(label_path)
                except OSError:
                    self.logger.exception("Failed to remove empty label file: %s", label_path)
            return
        try:
            atomic_write_text(label_path, "".join(lines))
        except Exception:
            self.logger.exception("Failed to save label file: %s", label_path)
            messagebox.showerror("Error", f"Failed to save label file:\n{label_path}")
            return
    
    def load_project_from_path(self, directory, preferred_image=None, save_session=True):
        self.project_root = directory.replace('\\', '/')
        self.ensure_yolo_label_dirs(self.project_root)

        split_files = {
            split: self._list_split_images(split)
            for split in ("train", "val", "test")
            if os.path.exists(f"{self.project_root}/images/{split}")
        }
        non_empty_splits = [s for s, files in split_files.items() if files]
        if non_empty_splits:
            if self.current_split not in non_empty_splits:
                self.current_split = "train" if "train" in non_empty_splits else non_empty_splits[0]
        elif split_files:
            if self.current_split not in split_files:
                self.current_split = "train" if "train" in split_files else next(iter(split_files))
        img_path = f"{self.project_root}/images/{self.current_split}"

        if not os.path.exists(img_path):
            os.makedirs(f"{self.project_root}/labels/{self.current_split}", exist_ok=True)

        self.combo_split.set(self.current_split)
        self.load_split_data(preferred_image=preferred_image)
        if save_session:
            self.save_session_state()

    def load_project_root(self):
        """????"""
        directory = filedialog.askdirectory()
        if not directory:
            return
        normalized = self.normalize_project_root(directory)
        yolo_root = self.find_yolo_project_root(normalized)
        self.load_project_from_path(yolo_root or normalized)
    
    def on_split_change(self, e=None):
        """??????"""
        if self.project_root:
            self.save_current()
            self.current_split = self.combo_split.get()
            self.load_split_data()
            self.save_session_state()
    
    def load_split_data(self, preferred_image=None):
        """??????"""
        img_path = f"{self.project_root}/images/{self.current_split}"

        if self.project_root and not os.path.exists(img_path):
            fallback = next(
                (s for s in ("train", "val", "test") if os.path.exists(f"{self.project_root}/images/{s}")),
                None,
            )
            if fallback:
                self.current_split = fallback
                self.combo_split.set(self.current_split)
                img_path = f"{self.project_root}/images/{self.current_split}"

        if self.project_root and os.path.exists(img_path):
            current_files = self._list_split_images(self.current_split)
            if not current_files:
                fallback_non_empty = next(
                    (
                        s for s in ("train", "val", "test")
                        if os.path.exists(f"{self.project_root}/images/{s}") and self._list_split_images(s)
                    ),
                    None,
                )
                if fallback_non_empty and fallback_non_empty != self.current_split:
                    self.current_split = fallback_non_empty
                    self.combo_split.set(self.current_split)
                    img_path = f"{self.project_root}/images/{self.current_split}"
        
        if os.path.exists(img_path):
            self.image_files = self._list_split_images(self.current_split)
            if self.image_files:
                if preferred_image:
                    name_to_idx = {
                        os.path.basename(path): i
                        for i, path in enumerate(self.image_files)
                    }
                    self.current_idx = name_to_idx.get(preferred_image, 0)
                else:
                    self.current_idx = 0
                self.load_img()
            else:
                self.current_idx = 0
                self.img_pil = None
                self.img_tk = None
                self.rects = []
                self.update_info_text()
        else:
            self.image_files = []
            self.current_idx = 0
            self.img_pil = None
            self.img_tk = None
            self.rects = []
            self.update_info_text()
        
        self.render()
        self.save_session_state()

    def _list_split_images(self, split: str) -> list[str]:
        img_path = f"{self.project_root}/images/{split}"
        return sorted([
            f for f in glob.glob(f"{img_path}/*.*")
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    def autolabel_red(self) -> None:
        """Auto-label reddish regions using LAB + contour filtering."""
        if not HAS_CV2 or not self.img_pil:
            return
        try:
            self.push_history()

            img = cv2.cvtColor(np.array(self.img_pil), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            _, a, _ = cv2.split(lab)
            _, red = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            k = max(1, int(self.config.red_detection_kernel_size))
            kernel = np.ones((k, k), np.uint8)
            red = cv2.dilate(
                red,
                kernel,
                iterations=max(1, int(self.config.red_detection_dilate_iterations)),
            )

            contours, _ = cv2.findContours(
                red,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > self.config.auto_label_min_area:
                    self.rects.append(self.clamp_box([
                        x, y, x + w, y + h,
                        self.combo_cls.current()
                    ]))

            self.render()
        except Exception:
            self.logger.exception("Red auto-label failed")
            messagebox.showerror("Error", "Red auto-label failed. See logs for details.")
    
    def run_yolo_detection(self):
        """Run detection by the selected model and append boxes."""
        if not HAS_YOLO:
            messagebox.showwarning("YOLO Not Available", "Please install ultralytics first.")
            return
        if not self.img_pil:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            mode = self.det_model_mode.get()
            if mode == "Official YOLO26m.pt (Bundled)":
                model_path = self._resolve_official_model_path()
            else:
                model_path = self.yolo_path.get().strip()

            if not model_path:
                messagebox.showwarning("Model", "Please choose a model file first.")
                return
            model_path = os.path.abspath(model_path)
            if not os.path.exists(model_path):
                messagebox.showerror("Model Not Found", f"Model file not found:\n{model_path}")
                return
            if not os.path.isfile(model_path):
                messagebox.showerror("Model Error", f"Path is not a file:\n{model_path}")
                return

            loaded_key = (mode, model_path)
            if self.yolo_model is None or self._loaded_model_key != loaded_key:
                self.root.config(cursor="watch")
                self.root.update_idletasks()
                if self.yolo_model is not None:
                    del self.yolo_model
                    self.yolo_model = None
                    gc.collect()
                try:
                    self.logger.info("Loading YOLO model: %s", model_path)
                    self.yolo_model = YOLO(model_path)
                    self._loaded_model_key = loaded_key
                except Exception as exc:
                    self.yolo_model = None
                    self._loaded_model_key = None
                    raise RuntimeError(f"Failed to load model: {exc}") from exc
                finally:
                    self.root.config(cursor="")
                    self.root.update_idletasks()

            results = self.yolo_model(
                self.img_pil,
                conf=self.var_yolo_conf.get(),
                verbose=False
            )
            
            self.push_history()
            detection_count = 0
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes.xyxy:
                    self.rects.append(self.clamp_box([
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                        self.combo_cls.current()
                    ]))
                    detection_count += 1
            
            self.render()
            self.logger.info("YOLO detection complete: %s boxes", detection_count)
        except FileNotFoundError as exc:
            self.logger.error("Model path error: %s", exc)
            messagebox.showerror("Model Error", str(exc))
        except MemoryError:
            self.logger.exception("Out of memory during YOLO detection")
            messagebox.showerror("Memory Error", "Not enough memory for detection.")
        except RuntimeError as exc:
            self.logger.exception("YOLO runtime error")
            messagebox.showerror("Detection Error", str(exc))
        except Exception as exc:
            self.logger.exception("YOLO detection failed")
            messagebox.showerror("Detection Error", f"YOLO detection failed:\n{exc}")

    def _iter_export_images(self) -> list[tuple[str, str, str]]:
        """Return (split, image_path, label_path) entries for full-dataset export."""
        entries: list[tuple[str, str, str]] = []
        if not self.project_root:
            return entries

        split_roots = [s for s in ("train", "val", "test") if os.path.isdir(f"{self.project_root}/images/{s}")]
        if split_roots:
            for split in split_roots:
                for img_path in self._list_split_images_for_root(self.project_root, split):
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    lbl_path = f"{self.project_root}/labels/{split}/{base}.txt"
                    entries.append((split, img_path, lbl_path))
            return entries

        # Flat image folder mode
        for img_path in sorted(
            f for f in glob.glob(f"{self.project_root}/*.*")
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = f"{self.project_root}/labels/train/{base}.txt"
            entries.append(("train", img_path, lbl_path))
        return entries

    def export_all_by_selected_format(self) -> None:
        if not self.project_root:
            messagebox.showwarning(LANG_MAP[self.lang]["title"], LANG_MAP[self.lang]["export_no_project"], parent=self.root)
            return
        if self.image_files and self.img_pil:
            self.save_current()

        out_dir = filedialog.askdirectory(
            parent=self.root,
            title=LANG_MAP[self.lang].get("select_export_folder", "Select Export Folder")
        )
        if not out_dir:
            return
        out_dir = out_dir.replace("\\", "/")

        fmt = self.var_export_format.get()
        try:
            if fmt == "YOLO (.txt)":
                count = self._export_all_yolo(out_dir)
            else:
                count = self._export_all_json(out_dir)
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("export_done", "Export completed: {count} images\nOutput: {path}").format(
                    count=count,
                    path=out_dir,
                ),
                parent=self.root,
            )
        except Exception as exc:
            self.logger.exception("Export failed")
            messagebox.showerror(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("export_failed", "Export failed: {err}").format(err=exc),
                parent=self.root,
            )

    def _export_all_yolo(self, out_dir: str) -> int:
        count = 0
        for split, img_path, lbl_path in self._iter_export_images():
            dst_img_dir = f"{out_dir}/images/{split}"
            dst_lbl_dir = f"{out_dir}/labels/{split}"
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_lbl_dir, exist_ok=True)

            shutil.copy2(img_path, f"{dst_img_dir}/{os.path.basename(img_path)}")
            if os.path.isfile(lbl_path):
                shutil.copy2(lbl_path, f"{dst_lbl_dir}/{os.path.basename(lbl_path)}")
            count += 1
        return count

    def _export_all_json(self, out_dir: str) -> int:
        count = 0
        for split, img_path, lbl_path in self._iter_export_images():
            dst_img_dir = f"{out_dir}/images/{split}"
            dst_ann_dir = f"{out_dir}/annotations/{split}"
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_ann_dir, exist_ok=True)

            img_name = os.path.basename(img_path)
            base = os.path.splitext(img_name)[0]
            shutil.copy2(img_path, f"{dst_img_dir}/{img_name}")

            with Image.open(img_path) as im:
                width, height = im.width, im.height

            anns: list[dict[str, Any]] = []
            if os.path.isfile(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:])
                        x1 = (cx - w / 2) * width
                        y1 = (cy - h / 2) * height
                        x2 = (cx + w / 2) * width
                        y2 = (cy + h / 2) * height
                        cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
                        anns.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_yolo": [cx, cy, w, h],
                        })

            payload = {
                "image": img_name,
                "split": split,
                "width": width,
                "height": height,
                "annotations": anns,
            }
            atomic_write_json(f"{dst_ann_dir}/{base}.json", payload)
            count += 1
        return count
    
    def export_full_coco(self):
        """Export dataset as COCO (placeholder)."""
        messagebox.showinfo("Export", "COCO export will be implemented.")
    
    # ==================== Geometry ====================
    
    def clamp_box(self, box):
        """Clamp and normalize box coordinates to image bounds."""
        if not self.img_pil:
            return box
        
        W, H = self.img_pil.width, self.img_pil.height
        x1, x2 = sorted([box[0], box[2]])
        y1, y2 = sorted([box[1], box[3]])
        
        return [
            max(0, min(W, x1)),
            max(0, min(H, y1)),
            max(0, min(W, x2)),
            max(0, min(H, y2)),
            box[4]
        ]
    
    def get_handles(self, rect):
        """??????????"""
        x1, y1, x2, y2 = rect[:4]
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        
        return [
            (x1, y1), (xm, y1), (x2, y1), (x2, ym),
            (x2, y2), (xm, y2), (x1, y2), (x1, ym)
        ]
    
    def canvas_to_img(self, x, y):
        """Canvas ???????????????"""
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale
    
    def img_to_canvas(self, x, y):
        """????????????Canvas ??????"""
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y
    
    def push_history(self) -> None:
        """??????"""
        self.history_manager.push_snapshot(self.rects)
    
    def undo(self) -> None:
        """Undo last edit."""
        if self.history_manager.undo():
            self.render()
    
    def redo(self) -> None:
        """??"""
        if self.history_manager.redo():
            self.render()
    
    def save_and_next(self):
        """??????????????????"""
        self.save_current()
        self.current_idx = min(len(self.image_files) - 1, self.current_idx + 1)
        self.load_img()
    
    def prev_img(self):
        """????"""
        self.save_current()
        self.current_idx = max(0, self.current_idx - 1)
        self.load_img()
    
    def bind_events(self):
        """Bind mouse and keyboard shortcuts."""
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.root.bind("<Key-f>", lambda e: self.save_and_next())
        self.root.bind("<Key-F>", lambda e: self.save_and_next())
        self.root.bind("<Key-d>", lambda e: self.prev_img())
        self.root.bind("<Key-D>", lambda e: self.prev_img())
        self.root.bind("<Key-a>", lambda e: self.autolabel_red())
        self.root.bind("<Key-A>", lambda e: self.autolabel_red())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Delete>", self.delete_selected)

    def on_canvas_resize(self, e):
        if not self.img_pil:
            return
        self.fit_image_to_canvas()

    def fit_image_to_canvas(self):
        if not self.img_pil:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        iw, ih = self.img_pil.width, self.img_pil.height
        if iw <= 0 or ih <= 0:
            self.logger.warning("Invalid image size: %sx%s", iw, ih)
            return
        scale = min(cw / iw, ch / ih)
        self.scale = scale
        self.offset_x = (cw - iw * scale) / 2
        self.offset_y = (ch - ih * scale) / 2
        self.render()

# ==================== Entrypoint ====================
def main():
    root = tk.Tk()
    app = UltimateLabeller(root)
    root.mainloop()

if __name__ == "__main__":
    main()
