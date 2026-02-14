from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    min_window_width: int = 900
    min_window_height: int = 600
    default_window_size: str = "1100x700"
    handle_size: int = 8
    yolo_model_path: str = "yolo26n.pt"
    default_yolo_conf: float = 0.5
    max_learning_memory: int = 20
    auto_label_min_area: int = 150
    zoom_in_factor: float = 1.1
    zoom_out_factor: float = 0.9
    session_file_name: str = ".ai_labeller_session.json"
    mouse_handle_hit_radius_px: int = 15
    learning_profile_file_name: str = ".learning_profiles.json"
    learning_min_samples_per_class: int = 5
    learning_detect_min_area_ratio: float = 0.001
    learning_detect_max_area_ratio: float = 0.60
    learning_score_threshold: float = 7.5
    low_cpu_yolo_imgsz: int = 416
    low_cpu_yolo_max_det: int = 50
    active_learning_scan_limit: int = 80
