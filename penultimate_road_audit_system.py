"""
penultimate_road_audit_system.py

Final merged comparator engine (Option A features included).
- Dual Model Inference (Custom + COCO Base)
- Guardrail occlusion handling (vegetation mask)
- Temporal deduplication / cooldown
- Segmentation auto-download attempt (placeholder URL)
- FFmpeg/OpenCV frame extraction fallback
- ORB visual sync
- GISContextEngine + RootCauseAnalyzer
- Marking analysis (segmentation preferred, OpenCV fallback)
- PCI calculation, comparison, comparison image saving
- Pipeline logs & suppressed events collected in report

Save in project root and ensure models/ and results/ exist.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import sys
import re
import os
import subprocess
from math import sqrt
from tqdm import tqdm
import torch
import json

# Optional dependencies
try:
    import exiftool
    EXIFTOOL_AVAILABLE = True
except Exception:
    EXIFTOOL_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False
    Nominatim = None
    GeocoderTimedOut = Exception

# ---------------------------------------------------------------------
# PATHS / CONSTANTS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
COMPARISON_DIR = RESULTS_DIR / "comparisons"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "best.pt"
BASE_MODEL_PATH = MODELS_DIR / "yolov8s.pt"

SEG_MODEL_NAME = "road_markings_yolov8s-seg.pt"
SEG_MODEL_PATH = MODELS_DIR / SEG_MODEL_NAME

# FFmpeg path (relative to project root)
FFMPEG_PATH = PROJECT_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"

# Placeholder segmentation model download URL – replace with your hosted model
SEG_MODEL_URL = "https://example.com/path/to/road_markings_yolov8s-seg.pt"


# ---------------------------------------------------------------------
# HELPER: ENSURE SEGMENTATION MODEL EXISTS
# ---------------------------------------------------------------------
def ensure_segmentation_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if SEG_MODEL_PATH.exists():
        print(f"[SEG] Segmentation model found at {SEG_MODEL_PATH}")
        return True

    print(f"[SEG] Segmentation model missing. Attempting auto-download...")

    try:
        import requests
    except Exception:
        print("[SEG][WARN] 'requests' not installed. Please place segmentation model manually.")
        return False

    try:
        r = requests.get(SEG_MODEL_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(SEG_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[SEG] Successfully downloaded segmentation model → {SEG_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[SEG][ERROR] Auto-download failed: {e}")
        print(f"[SEG] Please manually place the model at: {SEG_MODEL_PATH}")
        return False


# ---------------------------------------------------------------------
# GIS ENGINE – simple offline + geopy fallback
# ---------------------------------------------------------------------
class GISContextEngine:
    def __init__(self):
        self.context_cache = {}
        self.profile_cache = {}
        self.geolocator = Nominatim(user_agent="road_audit_gis") if GEOPY_AVAILABLE else None
    
    def _offline_context(self, lat, lon):
        # Simple fallback
        return "Urban"

    def get_context(self, lat, lon):
        key = f"{lat:.4f},{lon:.4f}"
        if key in self.context_cache:
            return self.context_cache[key]
        context = "Urban"
        if GEOPY_AVAILABLE and self.geolocator is not None:
            try:
                location = self.geolocator.reverse((lat, lon), exactly_one=True, timeout=3)
                if location:
                    address = location.raw.get("address", {})
                    if any(k in address for k in ["motorway", "trunk", "highway"]):
                        context = "Highway"
                    elif "residential" in address:
                        context = "Residential"
                    elif any(k in address for k in ["intersection", "crossing"]):
                        context = "Intersection"
                    else:
                        context = "Urban"
                else:
                    context = self._offline_context(lat, lon)
            except Exception:
                context = self._offline_context(lat, lon)
        else:
            context = self._offline_context(lat, lon)

        self.context_cache[key] = context
        return context

    def build_gis_profile(self, lat, lon):
        key = f"{lat:.4f},{lon:.4f}"
        if key in self.profile_cache:
            return self.profile_cache[key]

        context = self.get_context(lat, lon)
        # defaults
        traffic_density_adt = 4000
        heavy_vehicle_share = 0.10
        recent_rainfall_mm = 20.0
        drainage_quality = "Moderate"
        soil_type = "Granular"
        accident_hotspot = False

        if context == "Highway":
            traffic_density_adt = 12000
            heavy_vehicle_share = 0.35
            drainage_quality = "Good"
        elif context == "Residential":
            traffic_density_adt = 2500
            heavy_vehicle_share = 0.05
            drainage_quality = "Poor"
            soil_type = "Clay"
            recent_rainfall_mm = 60.0
        elif context == "Intersection":
            traffic_density_adt = 9000
            heavy_vehicle_share = 0.20
            drainage_quality = "Poor"
            recent_rainfall_mm = 60.0
            accident_hotspot = True
        elif context == "Urban":
            traffic_density_adt = 6000
            heavy_vehicle_share = 0.15
            drainage_quality = "Moderate"
            soil_type = "Mixed"
            recent_rainfall_mm = 40.0

        profile = {
            "context": context,
            "traffic_density_adt": traffic_density_adt,
            "heavy_vehicle_share": heavy_vehicle_share,
            "recent_rainfall_mm": recent_rainfall_mm,
            "drainage_quality": drainage_quality,
            "soil_type": soil_type,
            "accident_hotspot": accident_hotspot,
        }
        self.profile_cache[key] = profile
        return profile


# ---------------------------------------------------------------------
# Root cause analyzer (GIS + metrics)
# ---------------------------------------------------------------------
class RootCauseAnalyzer:
    def determine_cause(self, defect_type, metrics, gis_profile):
        ctx = gis_profile.get("context", "Urban")
        rain = gis_profile.get("recent_rainfall_mm", 0.0)
        drainage = gis_profile.get("drainage_quality", "Moderate")
        soil = gis_profile.get("soil_type", "Mixed")
        traffic_adt = gis_profile.get("traffic_density_adt", 0)
        hv_share = gis_profile.get("heavy_vehicle_share", 0.1)
        hotspot = gis_profile.get("accident_hotspot", False)

        cause = "General ageing and service-related deterioration."

        if defect_type == "pothole":
            if rain >= 50 and drainage in ["Poor", "Blocked"]:
                cause = "Heavy rainfall + poor drainage → binder stripping and potholes."
            elif traffic_adt >= 10000 and hv_share >= 0.2:
                cause = "High traffic + heavy vehicles → fatigue damage and potholes."
            else:
                cause = "Local material disintegration and ageing."

        elif defect_type == "crack":
            crack_w = metrics.get("crack_width_cm", 0)
            if isinstance(soil, str) and soil.lower() == "clay":
                cause = "Clay subgrade shrink–swell → longitudinal cracking."
            elif crack_w >= 6:
                cause = "Wide cracks indicate fatigue progression due to traffic and aging."
            else:
                cause = "Thermal cycles and binder hardening → surface cracking."

        elif defect_type == "faded_markings":
            wear = metrics.get("marking_wear_pct", 0)
            if wear > 60 and traffic_adt >= 8000:
                cause = "High traffic → accelerated abrasion and marking fading."
            else:
                cause = "Ageing and UV exposure reduced visibility."

        elif defect_type == "lane_loss":
            deviation = metrics.get("lane_deviation", 0)
            if deviation > 0.3 and ctx == "Highway":
                cause = "Frequent lane changes and lateral wander → lane delineation loss."
            else:
                cause = "Inadequate maintenance intervals."

        elif defect_type in ["damaged_sign", "broken_guardrail"]:
            if hotspot or ctx == "Intersection":
                cause = "Accident-prone location → impacts to assets."
            else:
                cause = "Collision, impact, or vandalism."

        elif defect_type.startswith("missing_"):
            if ctx == "Highway":
                cause = "Probable knockdown by a vehicle on a high-speed facility."
            else:
                cause = "Possible theft or unauthorized removal."

        return cause


# ---------------------------------------------------------------------
# Visual synchronizer (ORB)
# ---------------------------------------------------------------------
class VisualSynchronizer:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _find_match(self, anchor_img, target_video_path):
        cap = cv2.VideoCapture(str(target_video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        gray1 = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.orb.detectAndCompute(gray1, None)

        best_score = 0
        best_frame = 0
        stride = max(1, fps)

        for i in range(0, max(1, total_frames), stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = self.orb.detectAndCompute(gray2, None)
            if des1 is None or des2 is None:
                continue
            matches = self.bf.match(des1, des2)
            good = [m for m in matches if m.distance < 50]
            if len(good) > best_score:
                best_score = len(good)
                best_frame = i

        cap.release()
        return best_frame, best_score

    def get_sync_offsets(self, base_video_path, present_video_path):
        print("\n[SYNC] Bi-directional visual alignment...")
        cap_b = cv2.VideoCapture(str(base_video_path))
        ret_b, frame_b = cap_b.read()
        cap_b.release()

        cap_p = cv2.VideoCapture(str(present_video_path))
        ret_p, frame_p = cap_p.read()
        cap_p.release()

        if not ret_b or not ret_p:
            print("[SYNC][WARN] Could not read one of the videos; defaulting to no offset.")
            return 0, 0

        frame_idx_p, score_p = self._find_match(frame_b, present_video_path)
        frame_idx_b, score_b = self._find_match(frame_p, base_video_path)

        if score_p > score_b and score_p > 20:
            tqdm.write(f"  Present video starts {frame_idx_p} frames later.")
            return 0, frame_idx_p
        elif score_b > score_p and score_b > 20:
            tqdm.write(f"  Base video starts {frame_idx_b} frames later.")
            return frame_idx_b, 0

        print("[SYNC] No strong match; using zero offsets.")
        return 0, 0


# ---------------------------------------------------------------------
# Comparator engine
# ---------------------------------------------------------------------
class EnhancedRoadAuditSystem:
    def __init__(self, config=None):
        print("=" * 72)
        print(" ROAD SAFETY AUDIT SYSTEM – PENULTIMATE COMPARATOR ENGINE ")
        print("=" * 72)
    

        # Device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] Using device: {self.device}")

        # default config
        self.config = {
            "pretrained_model": str(BASE_MODEL_PATH),
            "finetuned_model": str(BEST_MODEL_PATH),
            "segmentation_model": str(SEG_MODEL_PATH),
            "proc_height": 640,
            "min_confidence": 0.25,
            "fps": 5,
            "change_cooldown_frames": 3,
            "occlusion_persistence_frames": 5,
            "vegetation_hue_low": 25,
            "vegetation_hue_high": 100,
            "vegetation_saturation_min": 40,
            "vegetation_value_min": 40,
            "guardrail_occlusion_thresh": 0.08,
        }
        if config:
            self.config.update(config)

        self.gis_engine = GISContextEngine()
        self.rca = RootCauseAnalyzer()

        self.pci_stats = {"potholes": 0, "crack_len_px": 0.0, "faded_marks": 0, "total_frames": 0}
        self.pixel_to_cm_scale = 0.5
        self.global_gis_profile = None

        # temporal and occlusion state
        self.recent_events = []
        self.event_cooldown = self.config["change_cooldown_frames"]
        self.occlusion_state = {}
        self.pipeline_logs = []
        self.suppressed_events = []

        self._load_models()

    def _normalize_label(self, raw_label: str):
        """Convert YOLO model class names into internal category buckets."""
        l = raw_label.strip().lower()

        if l in ["Potholes", "pothole"]:
            return "pothole"
        if l in ["Cracks", "crack"]:
            return "crack"
        if l in ["SpeedBreaker", "speed_breaker", "speed_bump"]:
            return "speedbreaker"
        if l in ["StreetSigns", "street_signs", "street_sign", "sign", "signboard", "stop sign"]:
            return "sign"
        if l in ["StreetLights", "street_light", "streetlight"]:
            return "streetlight"
        if l in ["TrafficLights", "traffic_lights", "trafficlight", "signal", "traffic light"]:
            return "trafficlight"
        if l in ["GuardRails", "guardrail"]:
            return "guardrail"
        if l in ["ZebraCrossing", "pedestrian_crossing"]:
            return "zebra"
        
        # COCO fallbacks
        if l in ["car", "truck", "bus", "motorcycle"]:
            return "vehicle"
        if l in ["person"]:
            return "pedestrian"

        return l  # fallback



    def _reset_pci_stats(self):
        """Reset PCI statistics for a new video"""
        return {"potholes": 0, "crack_len_px": 0.0, "faded_marks": 0, "total_frames": 0}


    def _log(self, msg):
        print(msg)
        self.pipeline_logs.append(msg)

    def _load_models(self):
        # custom model
        try:
            self.custom_model = YOLO(self.config["finetuned_model"])
            print(f"[INIT] Loaded custom model: {self.config['finetuned_model']}")
        except Exception as e:
            print(f"[CRIT] Failed to load custom model: {e}")
            raise

        # base model fallback
        try:
            if Path(self.config["pretrained_model"]).exists():
                self.base_model = YOLO(self.config["pretrained_model"])
                print(f"[INIT] Loaded base model (COCO): {self.config['pretrained_model']}")
            else:
                self.base_model = None
                print("[INIT][WARN] Base model not found; processing will use Custom model only.")
        except Exception as e:
            print(f"[WARN] Failed to load base model: {e}")
            self.base_model = None

        # segmentation model
        seg_ready = ensure_segmentation_model()
        if seg_ready and SEG_MODEL_PATH.exists():
            try:
                self.seg_model = YOLO(str(SEG_MODEL_PATH))
                print(f"[INIT] Loaded segmentation model: {SEG_MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Failed to load segmentation model: {e}")
                self.pipeline_logs.append(f"Segmentation model load failure: {e}")
                self.seg_model = None
        else:
            self.seg_model = None
            print("[INIT][WARN] Segmentation model unavailable; marking analysis limited.")
            self.pipeline_logs.append("Segmentation model unavailable; using OpenCV-only marking analysis.")

    # ----------------- guardrail occlusion -----------------
    def _is_guardrail_occluded_by_vegetation(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            return False, 0.0
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        low = np.array([self.config["vegetation_hue_low"], self.config["vegetation_saturation_min"], self.config["vegetation_value_min"]])
        high = np.array([self.config["vegetation_hue_high"], 255, 255])
        mask = cv2.inRange(hsv, low, high)
        frac = np.count_nonzero(mask) / mask.size if mask.size > 0 else 0.0
        return frac >= self.config["guardrail_occlusion_thresh"], float(frac)

    # ----------------- crack width measurement -----------------
    def _measure_crack_width_px(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        return float(np.max(dist) * 2)

    def _update_scale_from_lanes(self, frame, lane_masks):
        if not lane_masks:
            return
        h, w = frame.shape[:2]
        cx = w // 2
        centers = []
        for mask in lane_masks:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            centers.append(int(np.median(xs)))
        if len(centers) < 2:
            return
        centers = sorted(centers)
        left_candidates = [c for c in centers if c < cx]
        right_candidates = [c for c in centers if c > cx]
        if not left_candidates or not right_candidates:
            return
        left = max(left_candidates)
        right = min(right_candidates)
        lane_width_px = abs(right - left)
        if lane_width_px <= 0:
            return
        lane_width_cm = 350.0  # 3.5m
        scale = lane_width_cm / lane_width_px
        self.pixel_to_cm_scale = 0.7 * self.pixel_to_cm_scale + 0.3 * scale

    # ----------------- marking analysis -----------------
    def _analyze_markings(self, frame):
        h, w = frame.shape[:2]
        markings_info = {"lane_markings_present": False, "center_line_present": False, "stop_line_present": False, "zebra_present": False, "marking_wear_pct": 0.0}
        marking_dets = []
        lane_masks = []

        if self.seg_model is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            roi = hsv[int(h * 0.6):, :]
            mask_white = cv2.inRange(roi, (0, 0, 160), (180, 60, 255))
            ratio = np.count_nonzero(mask_white) / mask_white.size if mask_white.size > 0 else 0.0
            if ratio > 0.20:
                markings_info["zebra_present"] = True
            markings_info["marking_wear_pct"] = float(max(0.0, min(100.0, (1 - ratio) * 100)))
            return markings_info, marking_dets

        try:
            result = self.seg_model(frame, verbose=False, device=self.device)[0]
        except Exception as e:
            self._log(f"[SEG][ERROR] Segmentation inference failed: {e}")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            roi = hsv[int(h * 0.6):, :]
            mask_white = cv2.inRange(roi, (0, 0, 160), (180, 60, 255))
            ratio = np.count_nonzero(mask_white) / mask_white.size if mask_white.size > 0 else 0.0
            if ratio > 0.20:
                markings_info["zebra_present"] = True
            markings_info["marking_wear_pct"] = float(max(0.0, min(100.0, (1 - ratio) * 100)))
            return markings_info, marking_dets

        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return markings_info, marking_dets

        mask_arr = masks.data.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])

        n_masks = mask_arr.shape[0]
        n_boxes = cls_ids.shape[0]
        count = min(n_masks, n_boxes) if n_boxes > 0 else n_masks

        total_area = 0
        fade_weighted_area = 0

        for idx in range(count):
            m = mask_arr[idx]
            cls_id = int(cls_ids[idx]) if idx < len(cls_ids) else -1
            cls_name = str(self.seg_model.names.get(cls_id, f"class_{cls_id}")).lower() if self.seg_model else f"class_{cls_id}"
            binary_mask = (m > 0.5).astype(np.uint8)
            area = int(np.count_nonzero(binary_mask))
            if area == 0:
                continue
            total_area += area
            det = {"label": cls_name, "area_px": area}
            if "lane" in cls_name or "edge" in cls_name:
                markings_info["lane_markings_present"] = True
                lane_masks.append(binary_mask)
            if "center" in cls_name:
                markings_info["center_line_present"] = True
            if "stop" in cls_name:
                markings_info["stop_line_present"] = True
            if "zebra" in cls_name or "pedestrian" in cls_name:
                markings_info["zebra_present"] = True
            ys, xs = np.where(binary_mask > 0)
            if len(xs) > 0:
                intensities = frame[ys, xs].mean(axis=1)
                mean_int = float(np.mean(intensities))
                fade = max(0.0, min(1.0, (200 - mean_int) / 80.0))
                det["fade_ratio"] = fade
                fade_weighted_area += fade * area
            marking_dets.append(det)

        if total_area > 0:
            avg_fade = fade_weighted_area / total_area
            markings_info["marking_wear_pct"] = float(avg_fade * 100.0)

        if lane_masks:
            self._update_scale_from_lanes(frame, lane_masks)

        return markings_info, marking_dets

    # ----------------- metadata extraction -----------------
    def extract_video_metadata(self, video_path):
        video_path = str(video_path)
        meta = {"gps_data": None, "fps": 0, "duration": 0}
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            meta["fps"] = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if meta["fps"] > 0:
                meta["duration"] = total / meta["fps"]
        cap.release()

        match = re.search(r"(\d+\.\d+)[_,\s]+(\d+\.\d+)", Path(video_path).name)
        if match:
            meta["gps_data"] = {"latitude": float(match.group(1)), "longitude": float(match.group(2))}
        elif EXIFTOOL_AVAILABLE:
            try:
                with exiftool.ExifToolHelper() as et:
                    tags = et.get_metadata(video_path)[0]
                    if "Composite:GPSLatitude" in tags and "Composite:GPSLongitude" in tags:
                        meta["gps_data"] = {"latitude": tags["Composite:GPSLatitude"], "longitude": tags["Composite:GPSLongitude"]}
            except Exception as e:
                self._log(f"[EXIF][WARN] Failed to read GPS from EXIF: {e}")

        return meta

    # ----------------- frame extraction -----------------
    def extract_frames(self, video_path, output_folder, fps=1, start_frame=0):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # ROBUST CLEANUP: Try multiple times
        max_attempts = 3
        for attempt in range(max_attempts):
            old_frames = list(output_folder.glob("frame_*.jpg"))
            if not old_frames:
                break
                
            for old in old_frames:
                try:
                    old.unlink()
                except Exception as e:
                    if attempt == max_attempts - 1:
                        self._log(f"[WARN] Could not delete old frame {old}: {e}")
            
            # Verify deletion
            remaining = list(output_folder.glob("frame_*.jpg"))
            if not remaining:
                break
            
            if attempt < max_attempts - 1:
                import time
                time.sleep(0.2)
        
        # Check if old frames still exist
        if list(output_folder.glob("frame_*.jpg")):
            self._log(f"[ERROR] Failed to clear old frames in {output_folder}")
            raise RuntimeError(f"Cannot clear old frames in {output_folder}. Close any programs using these files.")

        t_offset = 0.0
        if start_frame > 0:
            cap = cv2.VideoCapture(str(video_path))
            native_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
            cap.release()
            if native_fps > 0:
                t_offset = start_frame / native_fps

        if not FFMPEG_PATH.exists():
            self._log(f"[FFMPEG][WARN] FFmpeg not found at {FFMPEG_PATH}. Falling back to OpenCV.")
            return self._extract_frames_opencv(video_path, output_folder, fps, start_frame)

        cmd = [str(FFMPEG_PATH), "-y"]
        if t_offset > 0:
            cmd.extend(["-ss", f"{t_offset:.3f}"])
        cmd.extend(["-i", str(video_path), "-vf", f"fps={fps}", str(output_folder / "frame_%05d.jpg"), "-hide_banner", "-loglevel", "error"])

        print(f"[FFMPEG] Extracting frames from {video_path} at {fps} fps (offset {t_offset:.3f}s)")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            self._log(f"[FFMPEG][ERROR] Frame extraction failed: {e}")
            return self._extract_frames_opencv(video_path, output_folder, fps, start_frame)

        frames = sorted(str(p) for p in output_folder.glob("frame_*.jpg"))
        
        # VALIDATION: Ensure frames were actually created
        if not frames:
            self._log(f"[ERROR] No frames extracted from {video_path}")
            raise RuntimeError(f"Frame extraction failed for {video_path}")
        
        self._log(f"[FFMPEG] Extracted {len(frames)} frames successfully")
        return frames


    def _extract_frames_opencv(self, video_path, output_folder, fps, start_frame):
        output_folder = Path(output_folder)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._log(f"[CV][ERROR] Could not open video: {video_path}")
            return []
        native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        skip = max(1, int(native_fps // fps))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        idx = 0
        saved = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=max(1, total_frames), desc=f"Extracting (OpenCV) {Path(video_path).name}", leave=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % skip == 0:
                frame_resized = cv2.resize(frame, (640, 640))
                out_path = output_folder / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(out_path), frame_resized)
                frames.append(str(out_path))
                saved += 1
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        return frames

    # ----------------- collate frame results -----------------
    def _collate_frame_results(self, frame_path, custom_result, base_result, gis_profile, frame_idx):
        img = cv2.imread(frame_path)
        if img is None:
            self._log(f"[FRAME][WARN] Could not read frame image: {frame_path}")
            return None

        self.pci_stats["total_frames"] += 1

        # analyze markings (segmentation or fallback)
        markings_info, marking_dets = self._analyze_markings(img)

        # Base frame structure
        frame_data = {
            "frame": frame_path,
            "frame_idx": frame_idx,
            "gis_profile": gis_profile,
            "potholes": [],
            "cracks": [],
            "road_signs": [],
            "traffic_lights": [],
            "furniture": [],
            "markings": markings_info,
            "marking_detections": marking_dets,   # list of {label, area_px, fade_ratio?}
        }

        # update PCI faded marks count
        if markings_info.get("marking_wear_pct", 0.0) > 50:
            self.pci_stats["faded_marks"] += 1

        # --- Helper for bbox logic ---
        def norm_label(lbl):
            return str(lbl).strip().lower().replace(" ", "_")

        def make_id_key(label, bbox):
            try:
                x1, y1, x2, y2 = map(float, bbox)
            except Exception:
                x1 = y1 = x2 = y2 = 0.0
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            w = int(max(1, x2 - x1))
            h = int(max(1, y2 - y1))
            return f"{label}::{cx}_{cy}_{w}_{h}"

        def iou_bbox(b1, b2):
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            inter = inter_w * inter_h
            a1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
            a2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
            union = a1 + a2 - inter
            if union <= 0: return 0.0
            return inter / union

        # Collect detections from BOTH models
        # We define a structure: {"label": str, "conf": float, "bbox": [x1,y1,x2,y2]}
        combined_detections = []

        # 1. Process Custom Model (Priority)
        if custom_result and getattr(custom_result, "boxes", None) is not None:
            for box in custom_result.boxes:
                try:
                    conf = float(box.conf[0])
                    if conf < self.config.get("min_confidence", 0.25): continue
                    cls_idx = int(box.cls[0])
                    raw_label = str(self.custom_model.names.get(cls_idx, f"class_{cls_idx}"))
                    bbox = [float(x) for x in box.xyxy[0].tolist()]
                    
                    combined_detections.append({
                        "raw_label": raw_label,
                        "label_norm": self._normalize_label(raw_label),
                        "conf": conf,
                        "bbox": bbox,
                        "source": "custom"
                    })
                except Exception:
                    continue

        # 2. Process Base Model (Secondary - Deduplicated)
        if base_result and getattr(base_result, "boxes", None) is not None:
            for box in base_result.boxes:
                try:
                    conf = float(box.conf[0])
                    if conf < self.config.get("min_confidence", 0.25): continue
                    cls_idx = int(box.cls[0])
                    raw_label = str(self.base_model.names.get(cls_idx, f"class_{cls_idx}"))
                    bbox = [float(x) for x in box.xyxy[0].tolist()]
                    
                    label_norm = self._normalize_label(raw_label)

                    # Deduplication: Check if this overlaps significantly with an existing Custom detection
                    is_duplicate = False
                    for existing in combined_detections:
                        if iou_bbox(bbox, existing["bbox"]) > 0.45:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        combined_detections.append({
                            "raw_label": raw_label,
                            "label_norm": label_norm,
                            "conf": conf,
                            "bbox": bbox,
                            "source": "base"
                        })
                except Exception:
                    continue

        # --- Categorize Detections ---
        for item in combined_detections:
            raw_label = item["raw_label"]
            label_norm = item["label_norm"]
            conf = item["conf"]
            bbox = item["bbox"]

            id_key = make_id_key(label_norm, bbox)

            det = {
                "label": label_norm,
                "raw_label": raw_label,
                "confidence": conf,
                "bbox": bbox,
                "id_key": id_key,
                "occluded": False,
                "source_model": item["source"]
            }

            lower = label_norm

            # Guardrail occlusion: check vegetation mask inside bbox
            if "guardrail" in lower or "guard_rail" in lower:
                occluded, veg_frac = self._is_guardrail_occluded_by_vegetation(img, bbox)
                if occluded:
                    det["occluded"] = True
                    det["occlusion_vegetation_fraction"] = float(veg_frac)
                    key = f"guardrail::{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
                    state = self.occlusion_state.get(key, {"occluded_frames": 0, "last_seen_frame": frame_idx})
                    state["last_seen_frame"] = frame_idx
                    state["occluded_frames"] = state.get("occluded_frames", 0) + 1
                    self.occlusion_state[key] = state
                else:
                    key = f"guardrail::{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
                    if key in self.occlusion_state:
                        self.occlusion_state[key]["occluded_frames"] = 0
                        self.occlusion_state[key]["last_seen_frame"] = frame_idx

            # Potholes
            if "pothole" in lower:
                det["root_cause"] = self.rca.determine_cause("pothole", {}, gis_profile)
                frame_data["potholes"].append(det)
                self.pci_stats["potholes"] += 1

            # Cracks
            elif "crack" in lower:
                width_px = self._measure_crack_width_px(img, bbox)
                width_cm = width_px * self.pixel_to_cm_scale
                det["width_px"] = width_px
                det["width_cm"] = width_cm
                metrics = {"crack_width_cm": width_cm}
                det["root_cause"] = self.rca.determine_cause("crack", metrics, gis_profile)
                frame_data["cracks"].append(det)
                self.pci_stats["crack_len_px"] += width_px

            # Speed breakers / raised-type
            elif "speed" in lower and "breaker" in lower:
                frame_data["furniture"].append(det)

            # Signs
            elif "sign" in lower:
                det["root_cause"] = self.rca.determine_cause("damaged_sign", {}, gis_profile)
                frame_data["road_signs"].append(det)

            # Traffic lights / signals
            elif "trafficlight" in lower or "signal" in lower:
                frame_data["traffic_lights"].append(det)

            # Street lights and furniture
            elif "streetlight" in lower or "light" in lower:
                frame_data["furniture"].append(det)

            # Guardrails
            elif "guardrail" in lower:
                det["root_cause"] = self.rca.determine_cause("broken_guardrail", {}, gis_profile)
                frame_data["furniture"].append(det)
            
            # Vehicles / Pedestrians (Base model stuff)
            elif "vehicle" in lower or "pedestrian" in lower:
                frame_data["furniture"].append(det) # Categorize as furniture for now to track presence

            else:
                # fallback to furniture
                frame_data["furniture"].append(det)

        return frame_data


    # ----------------- PCI calculation -----------------
    def _calculate_pci_score(self, gis_profile=None):
        score = 100.0
        total_frames = max(1, self.pci_stats["total_frames"])
        pothole_density = self.pci_stats["potholes"] / total_frames
        avg_crack_width_px = self.pci_stats["crack_len_px"] / total_frames
        marking_density = self.pci_stats["faded_marks"] / total_frames
        score -= pothole_density * 40.0
        score -= avg_crack_width_px * 0.2
        score -= marking_density * 20.0
        if gis_profile:
            ctx = gis_profile.get("context", "Urban")
            drainage = gis_profile.get("drainage_quality", "Moderate")
            traffic_adt = gis_profile.get("traffic_density_adt", 0)
            if ctx == "Highway":
                score -= 2.0
            if drainage in ["Poor", "Blocked"]:
                score -= 3.0
            if traffic_adt >= 10000:
                score -= 3.0
        score = max(0, min(100, int(score)))
        if score >= 85:
            rating = "Good"
        elif score >= 70:
            rating = "Satisfactory"
        elif score >= 55:
            rating = "Fair"
        elif score >= 40:
            rating = "Poor"
        else:
            rating = "Very Poor"
        return score, rating

    # ----------------- dedupe / cooldown helpers -----------------
    def _event_key_from_change(self, event):
        return f"{event['element']}::{event.get('type','')}".lower()


    def _should_suppress_event(self, event_key, current_frame):
        cooldown = int(self.event_cooldown)

        for ev in self.recent_events:
            if ev["key"] == event_key:
                if (current_frame - ev.get("last_frame", -99999)) <= cooldown:
                    return True
                ev["last_frame"] = current_frame
                return False

        # new event
        self.recent_events.append({"key": event_key, "last_frame": current_frame})
        if len(self.recent_events) > 2000:
            self.recent_events = self.recent_events[-1000:]
        return False


    def _best_iou_match(self, base_list, pres_list, iou_thresh=0.05):
        missing = []

        for b in base_list:
            b_box = b["bbox"]
            found = False
            for p in pres_list:
                if self._iou(b_box, p["bbox"]) > iou_thresh:
                    found = True
                    break

            if not found:
                missing.append(b)

        return missing



    # ----------------- comparison (with occlusion awareness) -----------------
    def _compare_frame_by_frame(self, base_results, present_results, fps):

        def iou_bbox(b1, b2):
            # b = [x1,y1,x2,y2]
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            inter = inter_w * inter_h
            a1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
            a2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
            union = a1 + a2 - inter
            if union <= 0:
                return 0.0
            return inter / union

        # ensure cooldown is higher as requested by user
        self.event_cooldown = int(self.config.get("change_cooldown_frames", 3))
        # increase it explicitly (user asked to increase). We'll set minimum 8.
        if self.event_cooldown < 8:
            self.event_cooldown = 8

        changes = []
        min_len = min(len(base_results), len(present_results))

        # IoU threshold for a match
        IOU_THRESH = 0.35

        for i in range(min_len):
            base = base_results[i]
            pres = present_results[i]
            frame_log = []

            # Helper: perform matching for a given class list name
            def compare_class_list(key_name, human_name):
                """
                key_name: "potholes", "cracks", "road_signs", "traffic_lights", "furniture"
                human_name: pretty string for messages
                """
                base_list = base.get(key_name, []) or []
                pres_list = pres.get(key_name, []) or []

                matched_pres = set()
                matched_base = set()

                # build index
                for bi, bdet in enumerate(base_list):
                    bbbox = bdet.get("bbox", [0, 0, 0, 0])
                    blabel = bdet.get("label", "")
                    best_j = -1
                    best_iou = 0.0
                    for pj, pdet in enumerate(pres_list):
                        if pj in matched_pres:
                            continue
                        pbbox = pdet.get("bbox", [0, 0, 0, 0])
                        piou = iou_bbox(bbbox, pbbox)
                        # only compare same label family where possible
                        if blabel and pdet.get("label", "") and blabel.split("::")[0] != pdet.get("label","").split("::")[0]:
                            # allow comparison even if labels slightly differ (e.g., guardrail vs guard_rail)
                            pass
                        if piou > best_iou:
                            best_iou = piou
                            best_j = pj
                    if best_j >= 0 and best_iou >= IOU_THRESH:
                        matched_base.add(bi)
                        matched_pres.add(best_j)
                        # check severity changes (for cracks: width increase; for potholes: maybe size change)
                        if key_name == "cracks":
                            # compare width if available
                            base_w = float(bdet.get("width_cm", bdet.get("width_px", 0) or 0) or 0)
                            pres_w = float(pres_list[best_j].get("width_cm", pres_list[best_j].get("width_px", 0) or 0) or 0)
                            if pres_w > base_w + 2.0:  # widened by >2cm (heuristic)
                                event = {"element": human_name, "type": "Crack widening observed", "severity": "medium", "details": {"base_width_cm": base_w, "present_width_cm": pres_w}}
                                key = self._event_key_from_change(event)
                                # suppression via cooldown
                                if self._should_suppress_event(key, i):
                                    self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                                else:
                                    frame_log.append(event)
                        # pothole size growth
                        if key_name == "potholes":
                            # If coords available we can compare bbox area growth
                            try:
                                b_area = (bbbox[2]-bbbox[0])*(bbbox[3]-bbbox[1])
                                p_area = (pres_list[best_j]["bbox"][2]-pres_list[best_j]["bbox"][0])*(pres_list[best_j]["bbox"][3]-pres_list[best_j]["bbox"][1])
                                if p_area > b_area * 1.5:
                                    event = {"element": human_name, "type": "Pothole growth observed", "severity": "high", "details": {"base_area_px": b_area, "present_area_px": p_area}}
                                    key = self._event_key_from_change(event)
                                    if self._should_suppress_event(key, i):
                                        self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                                    else:
                                        frame_log.append(event)
                            except Exception:
                                pass

                # anything in pres_list not matched => new detections
                for pj, pdet in enumerate(pres_list):
                    if pj in matched_pres:
                        continue
                    # new object
                    event = {"element": human_name, "type": f"New {human_name} detected", "severity": "medium", "details": {"label": pdet.get("label"), "confidence": pdet.get("confidence")}}
                    key = self._event_key_from_change(event)
                    if self._should_suppress_event(key, i):
                        self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                    else:
                        frame_log.append(event)

                # anything in base_list not matched => missing in present (possible removal/damage)
                for bi, bdet in enumerate(base_list):
                    if bi in matched_base:
                        continue
                    # check occlusion for guardrails & signs: if occluded in present, suppress as occlusion
                    label = (bdet.get("label") or "").lower()
                    bbox = bdet.get("bbox", [0, 0, 0, 0])
                    occlusion_key = f"guardrail::{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
                    occluded_reason = False

                    # guardrail occlusion check
                    if "guardrail" in label:
                        state = self.occlusion_state.get(occlusion_key)
                        if state and state.get("occluded_frames", 0) >= self.config.get("occlusion_persistence_frames", 5):
                            occluded_reason = True
                            self.suppressed_events.append({"frame": i, "event": {"element": human_name, "type": "Possible missing guardrail (occluded by vegetation)"}, "reason": "occlusion"})

                    # sign occlusion: if we have occlusion_state for guardrail or if present frame marking occlusion percentage high
                    if "sign" in label or "board" in label:
                        # look for occlusion flags in present frame detections near same bbox (approx)
                        # If any furniture detection at same approximate location has occluded=True, treat as occlusion
                        for pdet in pres_list:
                            if pdet.get("occluded", False):
                                if iou_bbox(bbox, pdet.get("bbox", [0,0,0,0])) > 0.15:
                                    occluded_reason = True
                                    self.suppressed_events.append({"frame": i, "event": {"element": human_name, "type": "Missing sign suppressed (likely occluded)"}, "reason": "occlusion"})
                                    break

                    if occluded_reason:
                        continue

                    # if not occluded -> report missing/damaged
                    event = {"element": human_name, "type": f"Missing or damaged {human_name}", "severity": "high", "details": {"label": bdet.get("label"), "confidence": bdet.get("confidence")}}
                    key = self._event_key_from_change(event)
                    if self._should_suppress_event(key, i):
                        self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                    else:
                        frame_log.append(event)

            # Compare structural classes
            compare_class_list("potholes", "Pothole")
            compare_class_list("cracks", "Cracks")
            compare_class_list("road_signs", "Road Signs")
            compare_class_list("traffic_lights", "Traffic Light")
            compare_class_list("furniture", "Roadside Furniture")

            # ---------------------------
            # Markings comparison (IoU not available here; use presence/area heuristics)
            # ---------------------------
            base_marks = base.get("marking_detections", []) or []
            pres_marks = pres.get("marking_detections", []) or []

            # build simple label->area mapping
            def build_area_map(marks):
                m = {}
                for det in marks:
                    lbl = str(det.get("label","")).lower()
                    area = int(det.get("area_px", 0) or 0)
                    # accumulate area if multiple masks of same class
                    m[lbl] = m.get(lbl, 0) + area
                return m

            base_area_map = build_area_map(base_marks)
            pres_area_map = build_area_map(pres_marks)

            # consider common marking types
            marking_types = set(list(base_area_map.keys()) + list(pres_area_map.keys()) + ["lane", "center", "stop", "zebra", "edge"])
            for mtype in marking_types:
                base_area = base_area_map.get(mtype, 0)
                pres_area = pres_area_map.get(mtype, 0)
                # missing entirely
                if base_area > 0 and pres_area == 0:
                    event = {"element": "Markings", "type": f"{mtype} missing or not detected", "severity": "high", "details": {"base_area_px": base_area, "present_area_px": pres_area}}
                    key = self._event_key_from_change(event)
                    if self._should_suppress_event(key, i):
                        self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                    else:
                        frame_log.append(event)
                else:
                    # large area drop ( >60% )
                    if base_area > 0 and pres_area > 0 and pres_area < 0.4 * base_area:
                        event = {"element": "Markings", "type": f"{mtype} visibility decreased significantly", "severity": "medium", "details": {"base_area_px": base_area, "present_area_px": pres_area}}
                        key = self._event_key_from_change(event)
                        if self._should_suppress_event(key, i):
                            self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                        else:
                            frame_log.append(event)

            # Also use marking_wear_pct heuristic (existing behavior)
            try:
                if pres.get("markings", {}).get("marking_wear_pct", 0.0) > base.get("markings", {}).get("marking_wear_pct", 0.0) + 20:
                    event = {"element": "Markings", "type": "Additional fading or loss of markings", "severity": "medium"}
                    key = self._event_key_from_change(event)
                    if self._should_suppress_event(key, i):
                        self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                    else:
                        frame_log.append(event)
            except Exception:
                pass

            # Append changes for this frame if any
            if frame_log:
                changes.append({
                    "frame_id": i,
                    "timestamp_seconds": i / fps if fps > 0 else 0.0,
                    "changes": frame_log,
                    "base_frame": base.get("frame"),
                    "present_frame": pres.get("frame"),
                })

        return changes

    # ----------------- save comparison images -----------------
    def save_comparison_images(self, base_res, pres_res, changes, output_dir=COMPARISON_DIR):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for c in changes:
            fid = c["frame_id"]
            if fid >= len(base_res) or fid >= len(pres_res):
                continue
            img_b = cv2.imread(base_res[fid]["frame"])
            img_p = cv2.imread(pres_res[fid]["frame"])
            if img_b is None or img_p is None:
                continue
            if img_b.shape != img_p.shape:
                img_p = cv2.resize(img_p, (img_b.shape[1], img_b.shape[0]))
            combined = np.hstack((img_b, img_p))
            cv2.putText(combined, "BASE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "PRESENT", (combined.shape[1] // 2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(str(out / f"comp_{fid:05d}.jpg"), combined)

    # ----------------- main audit -----------------
    def run_complete_audit(self, base_path, present_path, manual_gps=None):
        base_path = Path(base_path)
        present_path = Path(present_path)

        # Sync videos and extract frames
        syncer = VisualSynchronizer()
        off_b, off_p = syncer.get_sync_offsets(str(base_path), str(present_path))

        base_out = PROJECT_ROOT / "data" / "base"
        present_out = PROJECT_ROOT / "data" / "present"
        base_out.mkdir(parents=True, exist_ok=True)
        present_out.mkdir(parents=True, exist_ok=True)

        print("[PROCESS] Extracting frames with FFmpeg/OpenCV (single pass, aligned offsets)...")

        base_frames = self.extract_frames(
            str(base_path),
            base_out,
            fps=self.config.get("fps", 1),
            start_frame=int(off_b) if off_b else 0,
        )

        present_frames = self.extract_frames(
            str(present_path),
            present_out,
            fps=self.config.get("fps", 1),
            start_frame=int(off_p) if off_p else 0,
        )

        if not base_frames or not present_frames:
            self._log("[ERROR] No frames extracted from one or both videos.")
            return {
                "error": "Frame extraction failed for one or both videos.",
                "logs": self.pipeline_logs,
            }

        # Get GPS
        base_meta = self.extract_video_metadata(str(base_path))
        gps = base_meta.get("gps_data")
        if not gps:
            present_meta = self.extract_video_metadata(str(present_path))
            gps = present_meta.get("gps_data")
        if manual_gps:
            gps = manual_gps
        if not gps:
            self._log("[WARN] No GPS found; using default (New Delhi).")
            gps = {"latitude": 28.6139, "longitude": 77.2090}

        self.global_gis_profile = self.gis_engine.build_gis_profile(gps["latitude"], gps["longitude"])

        # ==================== PROCESS BASE VIDEO ====================
        print("\n[PROCESS] Analyzing base video frames...")
        self.pci_stats = self._reset_pci_stats()  # FIX: Use method
        base_results = []
        
        for idx, f in enumerate(tqdm(base_frames, desc="AI Processing (Base)")):
            img = cv2.imread(f)
            if img is None:
                self._log(f"[BASE][WARN] Skipping unreadable frame: {f}")
                continue
            
            # --- DUAL MODEL INFERENCE ---
            custom_res = None
            base_res = None
            try:
                # Run Custom
                custom_res = self.custom_model(img, verbose=False, device=self.device)[0]
                # Run Base (if available)
                if self.base_model:
                    base_res = self.base_model(img, verbose=False, device=self.device)[0]
            except Exception as e:
                self._log(f"[BASE][ERROR] Model inference failed on {f}: {e}")
                continue
            
            # Collate (merge) results
            frame_data = self._collate_frame_results(f, custom_res, base_res, self.global_gis_profile, idx)
            if frame_data:
                base_results.append(frame_data)

        # Calculate base PCI and SAVE stats before reset
        base_pci_score, base_pci_rating = self._calculate_pci_score(self.global_gis_profile)
        base_pci_stats = self.pci_stats.copy()  # FIX: Save stats!
        
        # ==================== PROCESS PRESENT VIDEO ====================
        print("\n[PROCESS] Analyzing present video frames...")
        self.pci_stats = self._reset_pci_stats()  # FIX: Fresh stats for present
        present_results = []
        
        for idx, f in enumerate(tqdm(present_frames, desc="AI Processing (Present)")):
            img = cv2.imread(f)
            if img is None:
                self._log(f"[PRESENT][WARN] Skipping unreadable frame: {f}")
                continue
            
            # --- DUAL MODEL INFERENCE ---
            custom_res = None
            base_res = None
            try:
                # Run Custom
                custom_res = self.custom_model(img, verbose=False, device=self.device)[0]
                # Run Base (if available)
                if self.base_model:
                    base_res = self.base_model(img, verbose=False, device=self.device)[0]
            except Exception as e:
                self._log(f"[PRESENT][ERROR] Model inference failed on {f}: {e}")
                continue
            
            # Collate (merge) results
            frame_data = self._collate_frame_results(f, custom_res, base_res, self.global_gis_profile, idx)
            if frame_data:
                present_results.append(frame_data)

        present_pci_score, present_pci_rating = self._calculate_pci_score(self.global_gis_profile)
        present_pci_stats = self.pci_stats.copy()  # FIX: Save stats!

        # Compare
        print("\n[PROCESS] Comparing frame-by-frame deterioration...")
        changes = self._compare_frame_by_frame(base_results, present_results, self.config["fps"])

        # Save comparison images
        self.save_comparison_images(base_results, present_results, changes)

        # Aggregate stats
        agg = {
            "potholes": {
                "base": sum(len(r["potholes"]) for r in base_results), 
                "present": sum(len(r["potholes"]) for r in present_results)
            },
            "cracks": {
                "base": sum(len(r["cracks"]) for r in base_results), 
                "present": sum(len(r["cracks"]) for r in present_results)
            },
            "faded_marking_frames": {
                "base": sum(1 for r in base_results if r["markings"]["marking_wear_pct"] > 50), 
                "present": sum(1 for r in present_results if r["markings"]["marking_wear_pct"] > 50)
            },
        }
        agg["potholes"]["delta"] = agg["potholes"]["present"] - agg["potholes"]["base"]
        agg["cracks"]["delta"] = agg["cracks"]["present"] - agg["cracks"]["base"]
        agg["faded_marking_frames"]["delta"] = agg["faded_marking_frames"]["present"] - agg["faded_marking_frames"]["base"]

        # Build final report
        report = {
            "audit_date": datetime.now().isoformat(),
            "gps": gps,
            "gis_profile": self.global_gis_profile,
            "frames_analyzed": {
                "base": len(base_results),
                "present": len(present_results)
            },
            "pci_data": {
                "base": {"score": int(base_pci_score), "rating": str(base_pci_rating)},
                "present": {"score": int(present_pci_score), "rating": str(present_pci_rating)},
                "delta": int(present_pci_score) - int(base_pci_score),
            },
            "pci_stats": {
                "base": base_pci_stats,
                "present": present_pci_stats,
            },
            "aggregate_comparison": agg,
            "frame_level_changes": changes,
            "base_frame_data": base_results,
            "present_frame_data": present_results,
            "logs": self.pipeline_logs,
            "suppressed_events": self.suppressed_events,
        }

        # Save JSON
        try:
            outp = RESULTS_DIR / "audit_output.json"
            with open(outp, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"[OK] Audit JSON saved: {outp}")
        except Exception as e:
            self._log(f"[SAVE][WARN] Failed to write audit_output.json: {e}")

        return report
# EOF