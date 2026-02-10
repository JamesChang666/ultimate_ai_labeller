import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk, ImageDraw
import os, glob, json, copy, datetime, shutil
import numpy as np
from collections import deque

# 檢查庫環境
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

# ==================== 專業級配色方案 (參考 Figma/Adobe XD) ====================
COLORS = {
    # 主色調 - 專業靛藍色系
    "primary": "#5551FF",           # Figma 風格主色
    "primary_hover": "#4845E4",
    "primary_light": "#7B79FF",
    "primary_bg": "#F0F0FF",
    
    # 功能色
    "success": "#0FA958",           # 成功綠
    "danger": "#F24822",            # 危險紅
    "warning": "#FFAA00",           # 警告橙
    "info": "#18A0FB",              # 資訊藍
    
    # 中性色調 - Sketch 風格
    "bg_dark": "#1E1E1E",           # 深色背景
    "bg_medium": "#2C2C2C",         # 中等背景
    "bg_light": "#F5F5F5",          # 淺色背景
    "bg_white": "#FFFFFF",          # 白色背景
    "bg_canvas": "#18191B",         # Canvas 背景
    
    # 文字色
    "text_primary": "#000000",
    "text_secondary": "#8E8E93",
    "text_tertiary": "#C7C7CC",
    "text_white": "#FFFFFF",
    
    # 邊框與分隔線
    "border": "#E5E5EA",
    "divider": "#38383A",
    
    # 標註框配色 - 專業調色盤
    "box_1": "#FF3B30",  # 紅
    "box_2": "#FF9500",  # 橙
    "box_3": "#FFCC00",  # 黃
    "box_4": "#34C759",  # 綠
    "box_5": "#5AC8FA",  # 青
    "box_6": "#5856D6",  # 紫
    "box_selected": "#00D4FF",  # 選中框
    
    # 陰影
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

# ==================== 語言包 ====================
LANG_MAP = {
    "zh": {
        "title": "AI Labeller Pro",
        "load_proj": "載入專案",
        "undo": "撤銷",
        "redo": "重做",
        "autolabel": "紅字偵測",
        "fuse": "融合標註",
        "file_info": "檔案資訊",
        "no_img": "尚未載入影像",
        "filename": "檔案",
        "progress": "進度",
        "boxes": "標註框",
        "class_mgmt": "類別管理",
        "current_class": "當前類別",
        "edit_classes": "編輯類別",
        "reassign_class": "變更選取類別",
        "clear_labels": "清除本張標註",
        "add": "新增",
        "rename": "重新命名",
        "apply": "套用",
        "class_name": "類別名稱",
        "rename_prompt": "修改 '{name}':",
        "add_prompt": "類別名稱:",
        "current": "目前",
        "to": "改為",
        "no_label_selected": "尚未選取標註。",
        "no_classes_available": "沒有可用類別。",
        "theme_light": "淺色模式",
        "theme_dark": "深色模式",
        "export_format": "匯出格式",
        "ai_tools": "AI 工具",
        "auto_detect": "自動偵測",
        "learning": "學習模式",
        "propagate": "標籤傳遞",
        "run_detection": "執行偵測",
        "export": "匯出 COCO",
        "prev": "上一張",
        "next": "下一張",
        "shortcuts": "快捷鍵",
        "shortcut_help": "快捷鍵說明",
        "dataset": "資料集",
        "lang_switch": "English",
        "delete": "刪除選取",
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
        "export_format": "Export Format",
        "ai_tools": "AI Tools",
        "auto_detect": "Auto Detect",
        "learning": "Learning",
        "propagate": "Propagate",
        "run_detection": "Run Detection",
        "export": "Export COCO",
        "prev": "Previous",
        "next": "Next",
        "shortcuts": "Shortcuts",
        "shortcut_help": "Shortcut Help",
        "dataset": "Dataset",
        "lang_switch": "中文",
        "delete": "Delete Selected",
    }
}

# ==================== 核心運算 ====================
class BoundingBoxCore:
    @staticmethod
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter = (x2_i - x1_i) * (y2_i - y1_i)
        u = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
        return inter / u if u > 0 else 0.0

    @staticmethod
    def fuse_list(boxes, iou_thresh, dist_thresh):
        if len(boxes) <= 1:
            return boxes
        
        keep_fusing = True
        while keep_fusing:
            keep_fusing = False
            merged, used = [], [False] * len(boxes)
            
            for i in range(len(boxes)):
                if used[i]:
                    continue
                    
                curr = boxes[i]
                used[i] = True
                
                for j in range(i + 1, len(boxes)):
                    if not used[j]:
                        should_merge = BoundingBoxCore.calculate_iou(curr, boxes[j]) > iou_thresh
                        
                        if not should_merge:
                            h_dist = max(0, max(curr[0], boxes[j][0]) - min(curr[2], boxes[j][2]))
                            v_overlap = min(curr[3], boxes[j][3]) - max(curr[1], boxes[j][1])
                            if v_overlap > 0 and h_dist <= dist_thresh:
                                should_merge = True
                        
                        if should_merge:
                            curr = [
                                min(curr[0], boxes[j][0]),
                                min(curr[1], boxes[j][1]),
                                max(curr[2], boxes[j][2]),
                                max(curr[3], boxes[j][3]),
                                curr[4]
                            ]
                            used[j] = True
                            keep_fusing = True
                
                merged.append(curr)
            
            boxes = merged
        
        return boxes

# ==================== 主程式 ====================
class UltimateLabeller:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.lang = "zh"
        self.theme = "dark"
        self.root.title(LANG_MAP[self.lang]["title"])
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        
        # 設定自定義字體
        self.setup_fonts()
        self.apply_theme(self.theme, rebuild=False)
        self._tooltip_after_id = None
        self._tooltip_win = None
        
        # --- 核心資料 ---
        self.project_root = ""
        self.current_split = "train"
        self.image_files = []
        self.current_idx = 0
        self.rects = []  # [x1, y1, x2, y2, cid]
        self.class_names = ["text", "figure", "table"]
        self.history = []
        self.redo_stack = []
        self.learning_mem = deque(maxlen=20)
        
        # --- 視圖狀態 ---
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
        self.HANDLE_SIZE = 8
        self.show_all_labels = True
        
        # --- AI 參數 ---
        self.yolo_model = None
        self.yolo_path = tk.StringVar(value="yolov8n.pt")
        self.var_export_format = tk.StringVar(value="YOLO (.txt)")
        self.var_auto_yolo = tk.BooleanVar(value=False)
        self.var_learning = tk.BooleanVar(value=False)
        self.var_propagate = tk.BooleanVar(value=False)
        self.var_yolo_conf = tk.DoubleVar(value=0.5)
        self.var_fusion_iou = tk.DoubleVar(value=0.1)
        self.var_fusion_dist = tk.IntVar(value=30)
        
        self.setup_custom_style()
        self.setup_ui()
        self.bind_events()
    
    def setup_fonts(self):
        """設定專業字體"""
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
        """設定 ttk 樣式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 設定 Combobox
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
    
    def delete_selected(self, e=None):
        """刪除選中的標註框"""
        if self.selected_idx is not None:
            self.push_history()
            self.rects.pop(self.selected_idx)
            self.selected_idx = None
            self.render()
    
    def setup_ui(self):
        # ==================== 頂部工具欄 ====================
        self.setup_toolbar()
        
        # ==================== 側邊欄 ====================
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
        """設置頂部工具欄"""
        toolbar = tk.Frame(self.root, bg=COLORS["bg_dark"], height=56)
        toolbar.pack(side="top", fill="x")
        toolbar.pack_propagate(False)
        
        # 左側區域
        left_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        left_frame.pack(side="left", fill="y", padx=16)
        
        # Logo 和標題
        title_frame = tk.Frame(left_frame, bg=COLORS["bg_dark"])
        title_frame.pack(side="left", pady=12)
        
        tk.Label(
            title_frame,
            text="●",
            font=("Arial", 20),
            fg=COLORS["primary"],
            bg=COLORS["bg_dark"]
        ).pack(side="left", padx=(0, 8))
        
        tk.Label(
            title_frame,
            text=LANG_MAP[self.lang]["title"],
            font=self.font_title,
            fg=self.toolbar_text_color(COLORS["bg_dark"]),
            bg=COLORS["bg_dark"]
        ).pack(side="left")
        
        # 分隔線
        tk.Frame(
            left_frame,
            width=1,
            bg=COLORS["divider"]
        ).pack(side="left", fill="y", padx=16)
        
        # 載入專案按鈕
        self.create_toolbar_button(
            left_frame,
            text=f"📁  {LANG_MAP[self.lang]['load_proj']}",
            command=self.load_project_root,
            bg=COLORS["primary"]
        ).pack(side="left", padx=4)
        
        # Dataset 選擇器
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
        
        # 中間區域 - 編輯工具
        center_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        center_frame.pack(side="left", fill="y", padx=16)
        
        # 撤銷/重做
        self.create_toolbar_icon_button(
            center_frame,
            text="↶",
            command=self.undo,
            tooltip=LANG_MAP[self.lang]["undo"]
        ).pack(side="left", padx=2)
        
        self.create_toolbar_icon_button(
            center_frame,
            text="↷",
            command=self.redo,
            tooltip=LANG_MAP[self.lang]["redo"]
        ).pack(side="left", padx=2)
        
        tk.Frame(center_frame, width=1, bg=COLORS["divider"]).pack(
            side="left", fill="y", padx=8
        )
        
        # AI 工具
        self.create_toolbar_icon_button(
            center_frame,
            text="🔴",
            command=self.autolabel_red,
            tooltip=LANG_MAP[self.lang]["autolabel"],
            bg=COLORS["danger"]
        ).pack(side="left", padx=2)
        
        self.create_toolbar_icon_button(
            center_frame,
            text="🧩",
            command=self.fuse_current,
            tooltip=LANG_MAP[self.lang]["fuse"],
            bg=COLORS["warning"]
        ).pack(side="left", padx=2)
        
        # 右側區域
        right_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        right_frame.pack(side="right", fill="y", padx=16)

        # 快捷鍵說明圖示（滑鼠停留 1 秒顯示）
        self.create_help_icon(right_frame).pack(side="right", padx=4, pady=12)

        # 主題切換
        self.create_toolbar_button(
            right_frame,
            text=self.get_theme_switch_label(),
            command=self.toggle_theme,
            bg=COLORS["bg_medium"]
        ).pack(side="right", padx=4, pady=12)

        # 語言切換
        self.create_toolbar_button(
            right_frame,
            text=f"🌐  {LANG_MAP[self.lang]['lang_switch']}",
            command=self.toggle_language,
            bg=COLORS["bg_medium"]
        ).pack(side="right", padx=4, pady=12)
    
    def create_toolbar_button(self, parent, text, command, bg=None):
        """創建工具欄按鈕"""
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
        
        # 懸停效果
        def on_enter(e):
            btn.config(bg=COLORS["primary_hover"] if bg == COLORS["primary"] else COLORS["bg_medium"])
        
        def on_leave(e):
            btn.config(bg=bg_val)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn

    def create_help_icon(self, parent):
        """建立快捷鍵說明圖示"""
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
            ("Space", LANG_MAP[self.lang]["fuse"]),
            ("Ctrl+Z", LANG_MAP[self.lang]["undo"]),
            ("Ctrl+Y", LANG_MAP[self.lang]["redo"]),
            ("Ctrl+Shift+Z", LANG_MAP[self.lang]["redo"]),
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
        win.wm_geometry(f"+{x}+{y}")
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
        self._tooltip_win = win

    def hide_shortcut_tooltip(self):
        if self._tooltip_after_id:
            self.root.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        if self._tooltip_win:
            self._tooltip_win.destroy()
            self._tooltip_win = None
    
    def create_toolbar_icon_button(self, parent, text, command, tooltip="", bg=None):
        """創建工具欄圖標按鈕"""
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
        
        # 懸停效果
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
        """使顏色變亮"""
        if color == COLORS["danger"]:
            return "#FF6B54"
        elif color == COLORS["warning"]:
            return "#FFBB33"
        return color
    
    def setup_sidebar(self):
        """設置側邊欄"""
        sidebar = tk.Frame(self.root, width=320, bg=COLORS["bg_light"])
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)
        
        # 滾動容器
        canvas_scroll = tk.Canvas(sidebar, bg=COLORS["bg_light"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = tk.Frame(canvas_scroll, bg=COLORS["bg_light"])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        # ===== 檔案資訊卡片 =====
        self.create_info_card(scrollable_frame)
        
        # ===== 類別管理卡片 =====
        self.create_class_card(scrollable_frame)
        
        # ===== AI 工具卡片 =====
        self.create_ai_card(scrollable_frame)
        
        # ===== 快捷鍵卡片 =====
        self.create_shortcut_card(scrollable_frame)
        
        # ===== 底部導航 =====
        self.create_navigation(sidebar)
        
        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

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
        """創建卡片容器"""
        card = tk.Frame(
            parent,
            bg=COLORS["bg_white"],
            relief="flat",
            borderwidth=0
        )
        card.pack(fill="x", padx=16, pady=8)
        
        # 添加微妙陰影效果（通過邊框模擬）
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
        """創建檔案資訊卡片"""
        content = self.create_card(parent, LANG_MAP[self.lang]["file_info"])
        
        # 檔案名稱
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
        
        # 進度條容器
        progress_frame = tk.Frame(content, bg=COLORS["bg_white"])
        progress_frame.pack(fill="x", pady=(8, 0))
        
        # 進度文字
        self.lbl_progress = tk.Label(
            progress_frame,
            text="0 / 0",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"]
        )
        self.lbl_progress.pack(side="left")
        
        # 標註框數量
        self.lbl_box_count = tk.Label(
            progress_frame,
            text=f"{LANG_MAP[self.lang]['boxes']}: 0",
            font=self.font_primary,
            fg=COLORS["primary"],
            bg=COLORS["bg_white"]
        )
        self.lbl_box_count.pack(side="right")
    
    def create_class_card(self, parent):
        """創建類別管理卡片"""
        content = self.create_card(parent, LANG_MAP[self.lang]["class_mgmt"])
        
        # 當前類別標籤
        tk.Label(
            content,
            text=LANG_MAP[self.lang]["current_class"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        # 類別選擇器
        self.combo_cls = ttk.Combobox(
            content,
            values=self.class_names,
            state="readonly",
            font=self.font_primary
        )
        self.combo_cls.current(0)
        self.combo_cls.pack(fill="x", pady=(0, 12))
        self.combo_cls.bind("<<ComboboxSelected>>", self.on_class_change_request)
        
        # 編輯類別按鈕
        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["edit_classes"],
            command=self.edit_classes_table
        ).pack(fill="x", pady=(0, 12))

        # 變更已標註類別
        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang]["reassign_class"],
            command=self.reassign_labeled_class
        ).pack(fill="x", pady=(0, 12))

        # 清除本張標註
        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang]["clear_labels"],
            command=self.clear_current_labels
        ).pack(fill="x", pady=(0, 12))
        
        # 匯出格式
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
            values=["YOLO (.txt)", "Individual JSON"],
            state="readonly",
            font=self.font_primary
        ).pack(fill="x")
    
    def create_ai_card(self, parent):
        """創建 AI 工具卡片"""
        content = self.create_card(parent, LANG_MAP[self.lang]["ai_tools"])
        
        # 複選框樣式
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
            text=LANG_MAP[self.lang]["learning"],
            variable=self.var_learning,
            **checkbox_style
        ).pack(fill="x", pady=4)
        
        tk.Checkbutton(
            content,
            text=LANG_MAP[self.lang]["propagate"],
            variable=self.var_propagate,
            **checkbox_style
        ).pack(fill="x", pady=4)
        
        # 執行偵測按鈕
        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["run_detection"],
            command=self.run_yolo_detection,
            bg=COLORS["success"]
        ).pack(fill="x", pady=(12, 0))
    
    def create_shortcut_card(self, parent):
        """創建快捷鍵卡片"""
        content = self.create_card(parent, LANG_MAP[self.lang]["shortcuts"])
        
        shortcuts = [
            ("F", LANG_MAP[self.lang]["next"]),
            ("D", LANG_MAP[self.lang]["prev"]),
            ("A", LANG_MAP[self.lang]["autolabel"]),
            ("Space", LANG_MAP[self.lang]["fuse"]),
            ("Ctrl+Z", LANG_MAP[self.lang]["undo"]),
            ("Ctrl+Y", LANG_MAP[self.lang]["redo"]),
            ("Ctrl+Shift+Z", LANG_MAP[self.lang]["redo"]),
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
                width=10,
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
        """創建底部導航"""
        nav_frame = tk.Frame(parent, bg=COLORS["bg_light"], height=80)
        nav_frame.pack(side="bottom", fill="x")
        nav_frame.pack_propagate(False)
        
        btn_container = tk.Frame(nav_frame, bg=COLORS["bg_light"])
        btn_container.pack(fill="both", expand=True, padx=16, pady=16)
        
        # 上一張
        self.create_nav_button(
            btn_container,
            text=f"← {LANG_MAP[self.lang]['prev']}",
            command=self.prev_img,
            side="left"
        )
        
        # 下一張
        self.create_nav_button(
            btn_container,
            text=f"{LANG_MAP[self.lang]['next']} →",
            command=self.save_and_next,
            side="right",
            primary=True
        )
    
    def create_primary_button(self, parent, text, command, bg=None):
        """創建主要按鈕"""
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
        
        # 懸停效果
        original_bg = bg or COLORS["primary"]
        hover_bg = COLORS["primary_hover"] if not bg else self.lighten_color(bg)
        
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda e: btn.config(bg=original_bg))
        
        return btn

    def create_secondary_button(self, parent, text, command):
        """創建次要按鈕"""
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
        """創建導航按鈕"""
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
        
        # 懸停效果
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
        """切換語言"""
        self.lang = "en" if self.lang == "zh" else "zh"
        self.rebuild_ui()
    
    def update_info_text(self):
        """更新檔案資訊"""
        if not self.image_files:
            self.lbl_filename.config(text=LANG_MAP[self.lang]["no_img"])
            self.lbl_progress.config(text="0 / 0")
        else:
            filename = os.path.basename(self.image_files[self.current_idx])
            self.lbl_filename.config(text=filename)
            self.lbl_progress.config(
                text=f"{self.current_idx + 1} / {len(self.image_files)}"
            )
        
        self.lbl_box_count.config(
            text=f"{LANG_MAP[self.lang]['boxes']}: {len(self.rects)}"
        )
    
    def render(self):
        """渲染畫面"""
        self.canvas.delete("all")
        
        if not self.img_pil:
            return
        
        # 1. 繪製影像
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
        
        # 2. 繪製標註框
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
            
            # 繪製矩形
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=width
            )
            
            # 繪製控制點（僅選中時）
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
                # 繪製類別標籤（帶背景）
                class_name = (
                    self.class_names[rect[4]]
                    if rect[4] < len(self.class_names)
                    else f"ID:{rect[4]}"
                )
                
                # 計算標籤位置（在框的左上角上方）
                label_y = max(y1 - 24, 8)  # 確保不超出畫面
                
                # 繪製標籤背景
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
        
        # 3. 繪製臨時拉框
        if self.temp_rect_coords:
            cx, cy, ex, ey = self.temp_rect_coords
            self.canvas.create_rectangle(
                cx, cy, ex, ey,
                outline=COLORS["primary_light"],
                width=2,
                dash=(6, 4)
            )
        
        # 4. 繪製十字定位線（專業風格）
        mx, my = self.mouse_pos
        
        # 垂直線
        self.canvas.create_line(
            mx, 0,
            mx, self.root.winfo_height(),
            fill=COLORS["primary"],
            width=1,
            dash=(2, 4)
        )
        
        # 水平線
        self.canvas.create_line(
            0, my,
            self.root.winfo_width(), my,
            fill=COLORS["primary"],
            width=1,
            dash=(2, 4)
        )
        
        # 座標提示（專業風格）
        coord_text = f"{mx}, {my}"
        coord_id = self.canvas.create_text(
            mx + 12,
            my - 12,
            text=coord_text,
            fill=COLORS["text_primary"] if self.theme == "light" else COLORS["text_white"],
            font=self.font_mono,
            anchor="nw"
        )
        
        coord_bbox = self.canvas.bbox(coord_id)
        if coord_bbox:
            padding = 4
            self.canvas.create_rectangle(
                coord_bbox[0] - padding,
                coord_bbox[1] - padding,
                coord_bbox[2] + padding,
                coord_bbox[3] + padding,
                fill=COLORS["bg_dark"],
                outline=COLORS["primary"],
                width=1,
                tags="coord_bg"
            )
            self.canvas.tag_lower("coord_bg", coord_id)
        
        # 更新資訊顯示
        self.update_info_text()
    
    # ==================== 事件處理 ====================
    
    def on_mouse_move(self, e):
        """滑鼠移動"""
        self.mouse_pos = (e.x, e.y)
        self.render()
    
    def on_mouse_down(self, e):
        """滑鼠按下"""
        if not self.img_pil:
            return
        
        ix, iy = self.canvas_to_img(e.x, e.y)
        
        # 檢查是否點擊控制點
        if self.selected_idx is not None:
            for i, (hx, hy) in enumerate(self.get_handles(self.rects[self.selected_idx])):
                dist = np.sqrt((ix - hx) ** 2 + (iy - hy) ** 2) * self.scale
                if dist < 15:
                    self.active_handle = i
                    self.drag_start = (ix, iy)
                    self.push_history()
                    return
        
        # 檢查是否點擊框
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
        """滑鼠拖曳"""
        self.mouse_pos = (e.x, e.y)
        
        if not self.img_pil or not self.drag_start:
            self.render()
            return
        
        ix, iy = self.canvas_to_img(e.x, e.y)
        W, H = self.img_pil.width, self.img_pil.height
        
        # 限制在圖片範圍內
        ix = max(0, min(W, ix))
        iy = max(0, min(H, iy))
        
        if self.selected_idx is not None and self.active_handle is not None:
            # 調整控制點
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
            # 移動框
            dx = ix - self.drag_start[0]
            dy = iy - self.drag_start[1]
            rect = self.rects[self.selected_idx]
            
            # 邊界檢查
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
            # 拉新框
            if self.temp_rect_coords:
                self.temp_rect_coords = (
                    self.temp_rect_coords[0],
                    self.temp_rect_coords[1],
                    e.x,
                    e.y
                )
        
        self.render()
    
    def on_mouse_up(self, e):
        """滑鼠放開"""
        if self.temp_rect_coords:
            ix, iy = self.canvas_to_img(e.x, e.y)
            new_box = self.clamp_box([
                self.drag_start[0],
                self.drag_start[1],
                ix,
                iy,
                self.combo_cls.current()
            ])
            
            # 檢查框大小
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
        """滾輪縮放"""
        factor = 1.1 if e.delta > 0 else 0.9
        
        self.offset_x = e.x - (e.x - self.offset_x) * factor
        self.offset_y = e.y - (e.y - self.offset_y) * factor
        self.scale *= factor
        
        self.render()
    
    # ==================== 核心功能 ====================
    
    def on_class_change_request(self, e=None):
        """類別變更"""
        if self.selected_idx is not None:
            new_cid = self.combo_cls.current()
            if self.rects[self.selected_idx][4] != new_cid:
                self.push_history()
                self.rects[self.selected_idx][4] = new_cid
                self.render()
    
    def edit_classes_table(self):
        """編輯類別表格"""
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
        
        # 按鈕
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
        """將選取的標註改成另一個類別"""
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

    def clear_current_labels(self):
        """清除本張圖所有標註"""
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
    
    def load_img(self):
        """載入影像"""
        if not self.image_files:
            return
        
        path = self.image_files[self.current_idx]
        self.update_info_text()
        self.img_pil = Image.open(path)
        
        prev_rects = copy.deepcopy(self.rects)
        self.rects = []
        self.history = []
        self.redo_stack = []
        self.selected_idx = None
        self.active_handle = None
        self.is_moving_box = False
        self.drag_start = None
        self.temp_rect_coords = None
        
        # 載入標註
        base = os.path.splitext(os.path.basename(path))[0]
        label_path = f"{self.project_root}/labels/{self.current_split}/{base}.txt"
        
        if os.path.exists(label_path):
            W, H = self.img_pil.width, self.img_pil.height
            with open(label_path, 'r') as f:
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
        else:
            if self.var_propagate.get():
                self.rects = prev_rects
            if self.var_auto_yolo.get():
                self.run_yolo_detection()
        
        self.fit_image_to_canvas()
    
    def save_current(self):
        """儲存當前標註"""
        if not self.project_root or not self.img_pil:
            return
        
        path = self.image_files[self.current_idx]
        base = os.path.splitext(os.path.basename(path))[0]
        W, H = self.img_pil.width, self.img_pil.height
        
        label_path = f"{self.project_root}/labels/{self.current_split}/{base}.txt"
        
        with open(label_path, "w") as f:
            for rect in self.rects:
                cx = (rect[0] + rect[2]) / 2 / W
                cy = (rect[1] + rect[3]) / 2 / H
                w = (rect[2] - rect[0]) / W
                h = (rect[3] - rect[1]) / H
                f.write(f"{rect[4]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    def load_project_root(self):
        """載入專案"""
        directory = filedialog.askdirectory()
        if not directory:
            return
        
        self.project_root = directory.replace('\\', '/')
        img_path = f"{self.project_root}/images/{self.current_split}"
        
        if not os.path.exists(img_path):
            os.makedirs(f"{self.project_root}/labels/{self.current_split}", exist_ok=True)
        
        self.load_split_data()
    
    def on_split_change(self, e=None):
        """資料集切換"""
        if self.project_root:
            self.save_current()
            self.current_split = self.combo_split.get()
            self.load_split_data()
    
    def load_split_data(self):
        """載入資料集"""
        img_path = f"{self.project_root}/images/{self.current_split}"
        
        if os.path.exists(img_path):
            self.image_files = sorted([
                f for f in glob.glob(f"{img_path}/*.*")
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.current_idx = 0
            self.load_img()
        
        self.render()
    
    def autolabel_red(self):
        """紅字偵測"""
        if not HAS_CV2 or not self.img_pil:
            return
        
        self.push_history()
        
        img = cv2.cvtColor(np.array(self.img_pil), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        _, red = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        red = cv2.dilate(red, kernel, iterations=3)
        
        contours, _ = cv2.findContours(
            red,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 150:
                self.rects.append(self.clamp_box([
                    x, y, x + w, y + h,
                    self.combo_cls.current()
                ]))
        
        self.fuse_current()
    
    def run_yolo_detection(self):
        """執行 YOLO 偵測"""
        if not HAS_YOLO or not self.img_pil:
            return
        
        try:
            if not self.yolo_model:
                self.yolo_model = YOLO(self.yolo_path.get())
            
            results = self.yolo_model(
                self.img_pil,
                conf=self.var_yolo_conf.get(),
                verbose=False
            )
            
            self.push_history()
            
            for result in results:
                for box in result.boxes.xyxy:
                    self.rects.append(self.clamp_box([
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                        self.combo_cls.current()
                    ]))
            
            self.fuse_current()
        except Exception as e:
            print(f"YOLO detection error: {e}")
    
    def fuse_current(self):
        """融合標註框"""
        self.rects = BoundingBoxCore.fuse_list(
            self.rects,
            self.var_fusion_iou.get(),
            self.var_fusion_dist.get()
        )
        self.render()
    
    def export_full_coco(self):
        """匯出 COCO"""
        messagebox.showinfo("Export", "COCO 匯出功能保持不變")
    
    # ==================== 輔助函數 ====================
    
    def clamp_box(self, box):
        """限制框在圖片範圍內"""
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
        """取得控制點位置"""
        x1, y1, x2, y2 = rect[:4]
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        
        return [
            (x1, y1), (xm, y1), (x2, y1), (x2, ym),
            (x2, y2), (xm, y2), (x1, y2), (x1, ym)
        ]
    
    def canvas_to_img(self, x, y):
        """Canvas 座標轉影像座標"""
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale
    
    def img_to_canvas(self, x, y):
        """影像座標轉 Canvas 座標"""
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y
    
    def push_history(self):
        """儲存歷史"""
        self.history.append(copy.deepcopy(self.rects))
        self.redo_stack.clear()
    
    def undo(self):
        """撤銷"""
        if self.history:
            self.redo_stack.append(copy.deepcopy(self.rects))
            self.rects = self.history.pop()
            self.render()
    
    def redo(self):
        """重做"""
        if self.redo_stack:
            self.history.append(copy.deepcopy(self.rects))
            self.rects = self.redo_stack.pop()
            self.render()
    
    def save_and_next(self):
        """儲存並下一張"""
        self.save_current()
        self.current_idx = min(len(self.image_files) - 1, self.current_idx + 1)
        self.load_img()
    
    def prev_img(self):
        """上一張"""
        self.save_current()
        self.current_idx = max(0, self.current_idx - 1)
        self.load_img()
    
    def bind_events(self):
        """綁定事件"""
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
        self.root.bind("<space>", lambda e: self.fuse_current())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda e: self.redo())
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
        scale = min(cw / iw, ch / ih)
        self.scale = scale
        self.offset_x = (cw - iw * scale) / 2
        self.offset_y = (ch - ih * scale) / 2
        self.render()

# ==================== 主程式入口 ====================
def main():
    """這是供 pip 套件呼叫的進入點"""
    root = tk.Tk()
    app = UltimateLabeller(root)
    root.mainloop()

if __name__ == "__main__":
    # 這樣你直接執行 python main.py 也能跑
    main()
