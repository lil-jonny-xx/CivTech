"""
ULTIMATE Road Safety Audit System V9 - ENHANCED

FIXES APPLIED (V9.9.2):
- FIXED: Increased cooldown period to 10 seconds to stop repetitions.
- ADDED: "Strike System" for pavement/marking deterioration.
- ADDED: "Guardrail Filter" to stop 'bench' false positives.
- ADDED: Bounding boxes are now drawn on comparison images.
- ADDED: 'VRUs' (person, bicycle, motorcycle) tracking.
- ADDED: 'Road Furniture' (bench, fire hydrant, parking meter) tracking.
- ADDED: Separate tracking for "traffic_lights".
- ADDED: Explicit check for CUDA GPU at startup.
- FIXED: ExifTool executable path.
- FIXED: Removed 'input()' prompt for automated pipeline.
- REMOVED: All emojis from print statements.

Save as: penultimate_road_audit_system.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime
import sys
import re
import gc
import subprocess
import torch
import os
from math import radians, sin, cos, sqrt, atan2

# --- EXIFTOOL FIX 1: Hard-code the path ---
EXIFTOOL_PATH = r"D:\RoadSafetyAI\CivTech\exiftool.exe"

# Optional ExifTool (graceful degradation if not available)
try:
    import exiftool
    EXIFTOOL_AVAILABLE = True
    
    # --- EXIFTOOL FIX 2: THIS IS THE CRITICAL FIX ---
    exiftool.executable = EXIFTOOL_PATH

except ImportError:
    EXIFTOOL_AVAILABLE = False
    print(" [WARNING] 'PyExifTool' library not found. GPS metadata extraction will be limited.")
except Exception as e:
    EXIFTOOL_AVAILABLE = False
    print(f" [WARNING] Error initializing ExifTool: {e}")


class EnhancedRoadAuditSystem:
    """
    Enhanced system with all fixes
    """
    
    def __init__(self, config=None):
        print("="*70)
        print(" ENHANCED ROAD SAFETY AUDIT SYSTEM V9.9.2 (Cooldown Fix)")
        print(" All Quality Improvements Implemented")
        print("="*70)
        
        print("\n[INFO] Checking for GPU (CUDA)...")
        if torch.cuda.is_available():
            print(f"   [SUCCESS] CUDA is available!")
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   [WARNING] CUDA not available. Running on CPU (this will be slow).")
        
        # --- NEW (V9.9.1): STRIKE SYSTEM ---
        # This prevents alerts from single bad frames (e.g., shadows)
        # An issue must be present for this many consecutive frames to be reported.
        self.STRIKE_THRESHOLD = 5 
        
        # Initialize strike counters
        self.strike_counters = {
            'pavement': 0,
            'markings': 0
        }
        # --- END STRIKE SYSTEM ---
        
        # Configuration with defaults
        self.config = config or {
            'pretrained_model': 'models/yolov8s.pt',
            'finetuned_model': 'models/pothole_detector_v1.pt',
            'proc_height': 736,
            'min_confidence': 0.25,
            'nms_iou_threshold': 0.5,
            'gps_distance_threshold_km': 5.0,
            'fps': 5,
            'batch_size': 1
        }
        
        self.proc_height = self.config['proc_height']
        self.min_confidence = self.config['min_confidence']
        
        print(f"\n[INFO] Configuration:")
        print(f"   Processing Resolution: {self.proc_height}p")
        print(f"   Min Confidence: {self.min_confidence}")
        print(f"   NMS IoU Threshold: {self.config['nms_iou_threshold']}")
        
        # Check ExifTool
        print("\n[0/3] Checking dependencies...")
        self.use_exiftool = self._check_exiftool()
        
        # Load models
        print("\n[1/3] Loading pre-trained YOLO...")
        try:
            self.pretrained_yolo = YOLO(self.config['pretrained_model'])
            print(f"[SUCCESS] Loaded: {self.config['pretrained_model']}")
        except Exception as e:
            print(f"[ERROR] Error loading pre-trained model: {e}")
            sys.exit(1)
        
        print("\n[2/3] Loading fine-tuned model...")
        finetuned_path = self.config['finetuned_model']
        
        if Path(finetuned_path).exists():
            try:
                self.finetuned_model = YOLO(finetuned_path)
                print(f"[SUCCESS] Loaded: {finetuned_path}")
                print(f"   Classes: {list(self.finetuned_model.names.values())}")
                self.use_finetuned = True
            except Exception as e:
                print(f"[WARNING] Error: {e}")
                self.finetuned_model = None
                self.use_finetuned = False
        else:
            print(f"[WARNING] Not found: {finetuned_path}")
            self.finetuned_model = None
            self.use_finetuned = False
        
        print("\n[3/3] Initializing OpenCV...")
        print("[SUCCESS] OpenCV ready")
        
        print("\n" + "="*70)
        print(" PIPELINE SUMMARY")
        print("="*70)
        print("\n[INFO] Detection Methods:")
        print(f"   1. Pre-trained YOLO (conf > {self.min_confidence})")
        print(f"   2. Fine-tuned Model (conf > {self.min_confidence})" if self.use_finetuned else "   2. Fine-tuned Model (DISABLED)")
        print("   3. OpenCV Analysis")
        print("\n[INFO] Quality Features:")
        print("   - Confidence filtering")
        print("   - Duplicate removal (NMS)")
        print("   - Frame-level tracking (with cooldown & strike logic)")
        print("   - Visual comparisons (with BBoxes)")
        print("="*70)
    
    def _check_exiftool(self):
        """Check if ExifTool is available"""
        if not EXIFTOOL_AVAILABLE:
            print("   [WARNING] exiftool Python module (PyExifTool) not installed")
            return False
        
        try:
            subprocess.run([EXIFTOOL_PATH, "-ver"], 
                           check=True, capture_output=True, timeout=5)
            print("   [SUCCESS] ExifTool available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"   [ERROR] ExifTool executable not found. Looked for it at:")
            print(f"   {EXIFTOOL_PATH}")
            return False
    
    def extract_video_metadata(self, video_path):
        """Extract metadata with enhanced error handling"""
        print(f"\n[INFO] Extracting metadata: {Path(video_path).name}")
        
        metadata = {
            'video_path': video_path,
            'gps_data': None,
            'duration': 0,
            'fps': 0,
            'resolution': (0, 0)
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                metadata['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if metadata['fps'] > 0:
                    metadata['duration'] = total_frames / metadata['fps']
                metadata['resolution'] = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                cap.release()
            
            print(f"   Duration: {metadata['duration']:.1f}s, FPS: {metadata['fps']}, Res: {metadata['resolution']}")
        except cv2.error as e:
            print(f"   [WARNING] OpenCV error reading video properties: {e}")
        
        # GPS extraction
        gps_found = False
        if self.use_exiftool:
            try:
                with exiftool.ExifToolHelper() as et:
                    meta_dict_list = et.get_metadata(video_path)
                    if meta_dict_list:
                        meta_dict = meta_dict_list[0]
                        
                        if 'Composite:GPSLatitude' in meta_dict and 'Composite:GPSLongitude' in meta_dict:
                            lat = meta_dict['Composite:GPSLatitude']
                            lon = meta_dict['Composite:GPSLongitude']
                            
                            if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
                                metadata['gps_data'] = {
                                    'latitude': float(lat),
                                    'longitude': float(lon),
                                    'source': 'video_metadata'
                                }
                                print(f"   [SUCCESS] GPS (metadata): {lat:.6f}, {lon:.6f}")
                                gps_found = True
            except Exception as e:
                print(f"   [WARNING] ExifTool error: {e}")
        
        # Fallback to filename
        if not gps_found:
            filename = Path(video_path).stem
            gps_pattern = r'(\d+\.\d+)[_,\s]+(\d+\.\d+)'
            match = re.search(gps_pattern, filename)
            
            if match:
                metadata['gps_data'] = {
                    'latitude': float(match.group(1)),
                    'longitude': float(match.group(2)),
                    'source': 'filename'
                }
                print(f"   [SUCCESS] GPS (filename): {match.group(1)}, {match.group(2)}")
            else:
                print(f"   [WARNING] No GPS data found")
        
        return metadata
    
    def validate_video_compatibility(self, base_metadata, present_metadata):
        """
        Validate that videos are compatible for comparison
        """
        print("\n[INFO] Validating video compatibility...")
        
        issues = []
        warnings = []
        
        # GPS distance check
        if base_metadata['gps_data'] and present_metadata['gps_data']:
            lat1 = radians(base_metadata['gps_data']['latitude'])
            lon1 = radians(base_metadata['gps_data']['longitude'])
            lat2 = radians(present_metadata['gps_data']['latitude'])
            lon2 = radians(present_metadata['gps_data']['longitude'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance_km = 6371 * c
            
            print(f"   GPS Distance: {distance_km:.2f} km")
            
            threshold = self.config['gps_distance_threshold_km']
            if distance_km > threshold:
                issues.append(f"Videos are {distance_km:.2f} km apart (threshold: {threshold} km)")
            elif distance_km > threshold / 5:
                warnings.append(f"Videos are {distance_km:.2f} km apart")
            else:
                print(f"   [SUCCESS] GPS check passed")
        else:
            warnings.append("GPS data unavailable - cannot verify location")
        
        # Resolution check
        if base_metadata['resolution'] != present_metadata['resolution']:
            warnings.append(f"Different resolutions: {base_metadata['resolution']} vs {present_metadata['resolution']}")
        else:
            print(f"   [SUCCESS] Resolution match")
        
        # Duration check
        duration_diff = abs(base_metadata['duration'] - present_metadata['duration'])
        if duration_diff > max(base_metadata['duration'], present_metadata['duration']) * 0.5:
            warnings.append(f"Duration differs significantly: {duration_diff:.1f}s")
        else:
            print(f"   [SUCCESS] Duration similar")
        
        # Display results
        if issues:
            print("\n[ERROR] VALIDATION FAILED")
            for issue in issues:
                print(f"   - {issue}")
            print("   Aborting due to validation failure.")
            return False # Auto-fail
        
        if warnings:
            print("\n[WARNINGS]")
            for warning in warnings:
                print(f"   - {warning}")
            print("   Proceeding with warnings...")
            return True # Auto-proceed
        
        print("   [SUCCESS] All validations passed")
        return True
    
    def extract_frames(self, video_path, output_folder, fps=1):
        """Extract and resize frames with enhanced error handling"""
        print(f"\n[INFO] Extracting: {Path(video_path).name} -> {self.proc_height}p @ {fps} FPS")
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            new_h = self.proc_height
            new_w = int(original_w * (new_h / original_h))
            
            frame_skip = max(1, frame_rate // fps)
            
            frames = []
            count, saved = 0, 0
            errors = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_skip == 0:
                    try:
                        resized_frame = cv2.resize(frame, (new_w, new_h), 
                                                   interpolation=cv2.INTER_AREA)
                        frame_path = f"{output_folder}/frame_{saved:06d}.jpg"
                        cv2.imwrite(frame_path, resized_frame)
                        frames.append(frame_path)
                        saved += 1
                    except cv2.error as e:
                        errors += 1
                        print(f"\n   [WARNING] Error at frame {count}: {e}")
                        if errors > 10:
                            print(f"   [ERROR] Too many errors, aborting")
                            break
                
                count += 1
                if count % 100 == 0:
                    print(f"   Progress: {count}/{total_frames}...", end='\r')
            
            cap.release()
            del frame
            gc.collect()
            
            print(f"\n[SUCCESS] Extracted {len(frames)} frames")
            if errors > 0:
                print(f"   [WARNING] {errors} frames had errors")
            
            return frames
            
        except MemoryError:
            print(f"\n[ERROR] Out of memory!")
            print(f"   Try reducing proc_height (current: {self.proc_height})")
            raise
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {type(e).__name__}: {e}")
            raise
    
    def _filter_by_confidence(self, detections):
        return [d for d in detections if d.get('confidence', 0) >= self.min_confidence]
    
    def _remove_duplicates_nms(self, detections):
        if len(detections) < 2:
            return detections
        
        sorted_dets = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        iou_threshold = self.config['nms_iou_threshold']
        
        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            
            filtered = []
            for det in sorted_dets:
                iou = self._calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    filtered.append(det)
            
            sorted_dets = filtered
        
        return keep
    
    def _calculate_iou(self, bbox1, bbox2):
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            ix1 = max(x1_1, x1_2)
            iy1 = max(y1_1, y1_2)
            ix2 = min(x2_1, x2_2)
            iy2 = min(y2_1, y2_2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            
            intersection = (ix2 - ix1) * (iy2 - iy1)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
            
    # --- NEW (V9.8): Helper function to draw boxes ---
    def _draw_boxes(self, image, detections, color, label_prefix=""):
        """Draws bounding boxes on an image."""
        for det in detections:
            try:
                # Bounding box
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = det.get('label', label_prefix)
                conf = det.get('confidence', 0)
                text = f"{label} ({conf:.2f})"
                
                # Text background
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                
                # Text
                cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            except Exception as e:
                print(f" [WARNING] Failed to draw box: {e}")
        # No return, modifies image in-place
    # --- END NEW ---

    def _run_cv_analysis(self, frame_path):
        try:
            img = cv2.imread(frame_path)
            if img is None:
                return {'pavement': {}, 'markings': {}}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, w = gray.shape
            
            # Pavement analysis
            edges = cv2.Canny(gray, 50, 150)
            crack_density = np.sum(edges > 0) / edges.size
            
            if crack_density > 0.08:
                severity, score = 'severe', 0
            elif crack_density > 0.05:
                severity, score = 'moderate', 40
            elif crack_density > 0.02:
                severity, score = 'minor', 70
            else:
                severity, score = 'good', 100
            
            pavement = { 'crack_density': float(crack_density), 'severity': severity, 'score': score }
            
            # Marking analysis
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            marking_mask = cv2.bitwise_or(white_mask, yellow_mask)
            marking_pixels = np.sum(marking_mask > 0)
            
            if marking_pixels > 200:
                brightness = np.mean(gray[marking_mask > 0])
                if brightness < 140:
                    condition, m_score = 'severely_faded', 20
                elif brightness < 180:
                    condition, m_score = 'faded', 50
                else:
                    condition, m_score = 'good', 100
            else:
                condition, m_score = 'missing', 0
            
            markings = { 'condition': condition, 'score': m_score }
            
            return {'pavement': pavement, 'markings': markings}
            
        except cv2.error:
            return {'pavement': {}, 'markings': {}}
        except Exception:
            return {'pavement': {}, 'markings': {}}
    
    def _collate_frame_results(self, frame_path, pretrained_result, finetuned_result, cv_analysis):
        """
        Collate results with confidence filtering and NMS
        FIX (V9.9): Added 'Guardrail Filter'
        """
        results = {
            'frame': frame_path,
            'road_signs': [],
            'furniture': [],
            'vru': [],
            'potholes': [],
            'cracks': [],
            'traffic_lights': [],
            'pavement': cv_analysis.get('pavement', {}),
            'markings': cv_analysis.get('markings', {})
        }
        
        # Collect pre-trained YOLO results
        if pretrained_result and hasattr(pretrained_result, 'boxes'):
            for box in pretrained_result.boxes:
                conf = float(box.conf[0])
                
                if conf < self.min_confidence:
                    continue
                
                cls_id = int(box.cls[0])
                label = self.pretrained_yolo.names[cls_id]
                detection = { 'label': label, 'confidence': conf, 'bbox': box.xyxy[0].tolist() }
                
                if label in ['stop sign']:
                    results['road_signs'].append(detection)
                elif label in ['traffic light']:
                    results['traffic_lights'].append(detection)
                elif label in ['bench', 'fire hydrant', 'parking meter']:
                    # --- NEW (V9.9) - GUARDRAIL FILTER ---
                    # Check if a "bench" is suspiciously wide (likely a guardrail)
                    if label == 'bench':
                        bbox = box.xyxy[0].tolist()
                        box_width = bbox[2] - bbox[0]
                        # Get frame width from the original image result
                        frame_width = pretrained_result.orig_shape[1] 
                        
                        # If box is wider than 60% of the screen, ignore it.
                        if (box_width / frame_width) > 0.6:
                            continue # Skip this detection
                    # --- END GUARDRAIL FILTER ---
                    results['furniture'].append(detection)
                elif label in ['person', 'bicycle', 'motorcycle']:
                    results['vru'].append(detection)
        
        # Collect fine-tuned results
        if self.use_finetuned and finetuned_result and hasattr(finetuned_result, 'boxes'):
            for box in finetuned_result.boxes:
                conf = float(box.conf[0])
                
                if conf < self.min_confidence:
                    continue
                
                cls_id = int(box.cls[0])
                label = self.finetuned_model.names[cls_id]
                detection = { 'label': label, 'confidence': conf, 'bbox': box.xyxy[0].tolist() }
                
                if 'pothole' in label.lower():
                    results['potholes'].append(detection)
                elif 'crack' in label.lower():
                    results['cracks'].append(detection)
        
        results['potholes'] = self._remove_duplicates_nms(results['potholes'])
        results['cracks'] = self._remove_duplicates_nms(results['cracks'])
        results['road_signs'] = self._remove_duplicates_nms(results['road_signs'])
        results['traffic_lights'] = self._remove_duplicates_nms(results['traffic_lights'])
        results['furniture'] = self._remove_duplicates_nms(results['furniture'])
        results['vru'] = self._remove_duplicates_nms(results['vru'])
        
        return results
    
    def _run_analysis_pipeline(self, frame_paths, batch_size=1):
        """Run analysis with error handling"""
        if not frame_paths:
            return []
        
        print(f"\n[ML] Processing {len(frame_paths)} frames (batch={batch_size})...")
        
        all_pretrained = []
        all_finetuned = []
        total_batches = (len(frame_paths) + batch_size - 1) // batch_size
        
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            print(f"   Batch {i // batch_size + 1}/{total_batches}...", end='\r')
            
            try:
                pretrained_batch = self.pretrained_yolo(batch_paths, verbose=False, imgsz=self.proc_height)
                all_pretrained.extend(pretrained_batch)
                
                if self.use_finetuned:
                    finetuned_batch = self.finetuned_model(batch_paths, verbose=False, imgsz=self.proc_height)
                    all_finetuned.extend(finetuned_batch)
                
                del pretrained_batch
                if self.use_finetuned: del finetuned_batch
                gc.collect()
                
            except MemoryError:
                print(f"\n   [ERROR] Out of memory at batch {i // batch_size + 1}")
                print(f"   Try reducing batch_size (current: {batch_size})")
                raise
            except Exception as e:
                print(f"\n   [WARNING] Error in batch {i // batch_size + 1}: {e}")
                all_pretrained.extend([None] * len(batch_paths))
                if self.use_finetuned:
                    all_finetuned.extend([None] * len(batch_paths))
                gc.collect()
        
        if not self.use_finetuned:
            all_finetuned = [None] * len(frame_paths)
        
        print(f"\n[SUCCESS] ML complete")
        
        print(f"\n[CV] Analyzing {len(frame_paths)} frames...")
        all_results = []
        for i, frame_path in enumerate(frame_paths):
            print(f"   Frame {i+1}/{len(frame_paths)}...", end='\r')
            cv_analysis = self._run_cv_analysis(frame_path)
            frame_data = self._collate_frame_results(
                frame_path, all_pretrained[i], all_finetuned[i], cv_analysis
            )
            all_results.append(frame_data)
        
        print(f"\n[SUCCESS] CV complete")
        return all_results
    
    def _compare_frame_by_frame(self, base_results, present_results, fps):
        """
        Frame-level change detection with cooldown logic
        FIX (V9.9.2): Increased cooldown period to 10 seconds to stop repetitions.
        """
        print("\n   Performing frame-by-frame comparison (with cooldown & strike logic)...")
        
        changes = []
        min_frames = min(len(base_results), len(present_results))
        
        cooldowns = {
            'potholes': 0, 'cracks': 0, 'signs': 0,
            'traffic_lights': 0, 'furniture': 0, 'vru': 0
        }
        
        # --- THIS IS THE FIX ---
        # Was `fps * 2` (2 seconds), which was too short and caused spam.
        # Now it's `fps * 10` (10 seconds).
        cooldown_period = fps * 10
        # --- END FIX ---
        
        for i in range(min_frames):
            base = base_results[i]
            present = present_results[i]
            frame_changes = []
            
            # Compare Potholes (Cooldown)
            if i >= cooldowns['potholes']:
                base_val = len(base.get('potholes', []))
                present_val = len(present.get('potholes', []))
                if present_val > base_val:
                    frame_changes.append({'element': 'Potholes', 'type': 'new_defects', 'severity': 'high', 'from': base_val, 'to': present_val, 'change': present_val - base_val})
                    cooldowns['potholes'] = i + cooldown_period
            
            # Compare Cracks (Cooldown)
            if i >= cooldowns['cracks']:
                base_val = len(base.get('cracks', []))
                present_val = len(present.get('cracks', []))
                if present_val > base_val + 1:
                    frame_changes.append({'element': 'Cracks', 'type': 'increased', 'severity': 'medium', 'from': base_val, 'to': present_val, 'change': present_val - base_val})
                    cooldowns['cracks'] = i + cooldown_period
            
            # --- MODIFIED (V9.9.1): Use Strike System for Pavement ---
            base_pav = base.get('pavement', {})
            present_pav = present.get('pavement', {})
            if base_pav and present_pav:
                base_score = base_pav.get('score', 100)
                present_score = present_pav.get('score', 100)
                
                if present_score < base_score - 25:
                    self.strike_counters['pavement'] += 1 # Add a strike
                else:
                    self.strike_counters['pavement'] = 0 # Reset
                    
                if self.strike_counters['pavement'] == self.STRIKE_THRESHOLD:
                    frame_changes.append({'element': 'Pavement Condition', 'type': 'deterioration', 'severity': 'high' if present_score < 40 else 'medium', 'from': base_pav.get('severity', 'unknown'), 'to': present_pav.get('severity', 'unknown'), 'score_change': base_score - present_score})
                    self.strike_counters['pavement'] = 0 # Reset after reporting
            
            # --- MODIFIED (V9.9.1): Use Strike System for Markings ---
            base_mark = base.get('markings', {})
            present_mark = present.get('markings', {})
            if base_mark and present_mark:
                base_cond = base_mark.get('condition', 'unknown')
                present_cond = present_mark.get('condition', 'unknown')
                
                if base_cond == 'good' and present_cond in ['faded', 'severely_faded', 'missing']:
                    self.strike_counters['markings'] += 1 # Add a strike
                else:
                    self.strike_counters['markings'] = 0 # Reset
                    
                if self.strike_counters['markings'] == self.STRIKE_THRESHOLD:
                    frame_changes.append({'element': 'Road Markings', 'type': 'fading', 'severity': 'high' if present_cond == 'missing' else 'medium', 'from': base_cond, 'to': present_cond})
                    self.strike_counters['markings'] = 0 # Reset after reporting
            
            # Compare Signs (Cooldown)
            if i >= cooldowns['signs']:
                base_val = len(base.get('road_signs', []))
                present_val = len(present.get('road_signs', []))
                if present_val < base_val:
                    frame_changes.append({'element': 'Road Signs', 'type': 'missing', 'severity': 'high', 'from': base_val, 'to': present_val})
                    cooldowns['signs'] = i + cooldown_period
            
            # Compare Traffic Lights (Cooldown)
            if i >= cooldowns['traffic_lights']:
                base_val = len(base.get('traffic_lights', []))
                present_val = len(present.get('traffic_lights', []))
                if present_val < base_val:
                    frame_changes.append({'element': 'Traffic Lights', 'type': 'missing_or_offline', 'severity': 'high', 'from': base_val, 'to': present_val})
                    cooldowns['traffic_lights'] = i + cooldown_period

            # Compare Furniture (Cooldown)
            if i >= cooldowns['furniture']:
                base_val = len(base.get('furniture', []))
                present_val = len(present.get('furniture', []))
                if present_val < base_val:
                    frame_changes.append({'element': 'Road Furniture', 'type': 'missing', 'severity': 'low', 'from': base_val, 'to': present_val})
                    cooldowns['furniture'] = i + cooldown_period

            # Compare VRUs (Cooldown)
            if i >= cooldowns['vru']:
                base_val = len(base.get('vru', []))
                present_val = len(present.get('vru', []))
                if present_val > (base_val * 2) and present_val > 5:
                    frame_changes.append({'element': 'VRU Presence', 'type': 'significant_increase', 'severity': 'medium', 'from': base_val, 'to': present_val})
                    cooldowns['vru'] = i + cooldown_period
            
            if frame_changes:
                changes.append({
                    'frame_id': i,
                    'timestamp_seconds': i / fps,
                    'changes': frame_changes
                })
        
        return changes
    
    def save_comparison_images(self, base_results, present_results, frame_level_changes, output_folder='results/comparisons'):
        """
        Save visual comparisons for frames with changes
        --- MODIFIED (V9.8): Now draws all bounding boxes ---
        """
        print(f"\n   Creating visual comparisons with bounding boxes...")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Define colors
        COLOR_DEFECT = (0, 0, 255)     # Red for potholes/cracks
        COLOR_ASSET = (0, 255, 0)      # Green for signs/lights
        COLOR_FURNITURE = (255, 200, 0) # Light Blue for furniture
        COLOR_VRU = (255, 0, 255)    # Magenta for VRUs
        
        saved = 0
        for change_data in frame_level_changes[:10]: # First 10 changes
            frame_id = change_data['frame_id']
            try:
                # Get the full data for this frame
                base_data = base_results[frame_id]
                present_data = present_results[frame_id]
                
                # Load images
                base_img = cv2.imread(base_data['frame'])
                present_img = cv2.imread(present_data['frame'])
                if base_img is None or present_img is None: continue
                
                # --- Draw all boxes on both images ---
                # Draw on BASE image
                self._draw_boxes(base_img, base_data['potholes'], COLOR_DEFECT, "Pothole")
                self._draw_boxes(base_img, base_data['cracks'], COLOR_DEFECT, "Crack")
                self._draw_boxes(base_img, base_data['road_signs'], COLOR_ASSET, "Sign")
                self._draw_boxes(base_img, base_data['traffic_lights'], COLOR_ASSET, "Light")
                self._draw_boxes(base_img, base_data['furniture'], COLOR_FURNITURE, "Furniture")
                self._draw_boxes(base_img, base_data['vru'], COLOR_VRU, "VRU")

                # Draw on PRESENT image
                self._draw_boxes(present_img, present_data['potholes'], COLOR_DEFECT, "Pothole")
                self._draw_boxes(present_img, present_data['cracks'], COLOR_DEFECT, "Crack")
                self._draw_boxes(present_img, present_data['road_signs'], COLOR_ASSET, "Sign")
                self._draw_boxes(present_img, present_data['traffic_lights'], COLOR_ASSET, "Light")
                self._draw_boxes(present_img, present_data['furniture'], COLOR_FURNITURE, "Furniture")
                self._draw_boxes(present_img, present_data['vru'], COLOR_VRU, "VRU")
                # --- End drawing ---

                # Ensure same height
                h = min(base_img.shape[0], present_img.shape[0])
                base_resized = cv2.resize(base_img, (int(base_img.shape[1] * h / base_img.shape[0]), h))
                present_resized = cv2.resize(present_img, (int(present_img.shape[1] * h / present_img.shape[0]), h))
                
                # Combine side by side
                comparison = np.hstack([base_resized, present_resized])
                
                # Add labels
                cv2.putText(comparison, 'BASE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(comparison, 'PRESENT', (base_resized.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Add change summary
                y_offset = 100
                for change in change_data['changes'][:3]: # First 3 changes
                    text = f"{change['element']}: {change['type']}"
                    cv2.putText(comparison, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30
                
                # Save
                output_path = f"{output_folder}/comparison_frame_{frame_id:04d}.jpg"
                cv2.imwrite(output_path, comparison)
                saved += 1
            except Exception as e:
                print(f"\n   [WARNING] Error creating comparison for frame {frame_id}: {e}")
        
        print(f"   [SUCCESS] Saved {saved} comparison images to {output_folder}/")
    
    def _create_aggregate_summary(self, analysis_results):
        """
        Create aggregate summary
        FIX (V9.7): Added 'total_vru'
        """
        summary = {
            'total_potholes': 0, 'total_cracks': 0, 'total_road_signs': 0,
            'total_traffic_lights': 0, 'total_furniture': 0, 'total_vru': 0,
            'pavement_scores': [], 'marking_scores': []
        }
        
        if not analysis_results: return summary
        
        for frame in analysis_results:
            summary['total_potholes'] += len(frame.get('potholes', []))
            summary['total_cracks'] += len(frame.get('cracks', []))
            summary['total_road_signs'] += len(frame.get('road_signs', []))
            summary['total_traffic_lights'] += len(frame.get('traffic_lights', []))
            summary['total_furniture'] += len(frame.get('furniture', []))
            summary['total_vru'] += len(frame.get('vru', [])) # <-- NEW V9.7
            
            pavement = frame.get('pavement', {})
            if pavement and 'score' in pavement:
                summary['pavement_scores'].append(pavement['score'])
            
            markings = frame.get('markings', {})
            if markings and 'score' in markings:
                summary['marking_scores'].append(markings['score'])
        
        summary['avg_pavement_score'] = np.mean(summary['pavement_scores']) if summary['pavement_scores'] else 0
        summary['avg_marking_score'] = np.mean(summary['marking_scores']) if summary['marking_scores'] else 0
        
        return summary
    
    def _generate_report(self, base_results, present_results, 
                           base_metadata, present_metadata, 
                           frame_level_changes, fps):
        """
        Generate comprehensive report
        FIX (V9.7): Added 'vru' to comparison
        """
        
        print("   Calculating summaries...")
        base_summary = self._create_aggregate_summary(base_results)
        present_summary = self._create_aggregate_summary(present_results)
        
        print("   Comparing aggregates...")
        comparison = {
            'potholes': {
                'base': base_summary['total_potholes'],
                'present': present_summary['total_potholes'],
                'change': present_summary['total_potholes'] - base_summary['total_potholes']
            },
            'cracks': {
                'base': base_summary['total_cracks'],
                'present': present_summary['total_cracks'],
                'change': present_summary['total_cracks'] - base_summary['total_cracks']
            },
            'pavement_condition': {
                'base': base_summary['avg_pavement_score'],
                'present': present_summary['avg_pavement_score'],
                'change': present_summary['avg_pavement_score'] - base_summary['avg_pavement_score']
            },
            'marking_condition': {
                'base': base_summary['avg_marking_score'],
                'present': present_summary['avg_marking_score'],
                'change': present_summary['avg_marking_score'] - base_summary['avg_marking_score']
            },
            'road_signs': {
                'base': base_summary['total_road_signs'],
                'present': present_summary['total_road_signs'],
                'change': present_summary['total_road_signs'] - base_summary['total_road_signs']
            },
            'traffic_lights': {
                'base': base_summary['total_traffic_lights'],
                'present': present_summary['total_traffic_lights'],
                'change': present_summary['total_traffic_lights'] - base_summary['total_traffic_lights']
            },
            'furniture': {
                'base': base_summary['total_furniture'],
                'present': present_summary['total_furniture'],
                'change': present_summary['total_furniture'] - base_summary['total_furniture']
            },
            'vru': {
                'base': base_summary['total_vru'],
                'present': present_summary['total_vru'],
                'change': present_summary['total_vru'] - base_summary['total_vru']
            }
        }
        
        # Count issues by severity
        issue_counts = {'high': 0, 'medium': 0, 'low': 0}
        for change in frame_level_changes:
            for item in change['changes']:
                severity = item.get('severity', 'low')
                issue_counts[severity] += 1
        
        return {
            'audit_date': datetime.now().isoformat(),
            'system_version': 'Enhanced V9.9.2',
            'configuration': self.config,
            'models_used': {
                'pretrained_yolo': self.config['pretrained_model'],
                'finetuned_model': self.config['finetuned_model'] if self.use_finetuned else 'None',
                'traditional_cv': f'Enabled @ {self.proc_height}p'
            },
            'base_video': base_metadata,
            'present_video': present_metadata,
            'frames_analyzed': {
                'base': len(base_results),
                'present': len(present_results)
            },
            'aggregate_comparison': comparison,
            'frame_level_changes': frame_level_changes,
            'total_frames_with_changes': len(frame_level_changes),
            'issue_summary': {
                'total_issues': sum(issue_counts.values()),
                'by_severity': issue_counts
            },
            'base_summary': base_summary,
            'present_summary': present_summary,
            'quality_metrics': {
                'confidence_threshold': self.min_confidence,
                'nms_applied': True,
                'duplicate_removal': 'Active (10s cooldown)', # Updated from 2s
                'guardrail_filter': 'Active',
                'deterioration_logic': 'Strike System (n=5)'
            }
        }
    
    def _print_summary(self, report):
        """
        Print comprehensive summary
        FIX (V9.7): Added 'VRUs'
        """
        print("\n" + "="*70)
        print(" ENHANCED AUDIT SUMMARY (V9.9.2)")
        print("="*70)
        
        comp = report['aggregate_comparison']
        
        def arrow(change):
            if change > 0: return "+"
            if change < 0: return "-"
            return " "
        
        print("\n[INFO] AGGREGATE COMPARISON")
        print(f"\n   Potholes:       {comp['potholes']['base']} -> {comp['potholes']['present']} "
              f"({arrow(comp['potholes']['change'])}{abs(comp['potholes']['change'])})")
        
        print(f"   Cracks:         {comp['cracks']['base']} -> {comp['cracks']['present']} "
              f"({arrow(comp['cracks']['change'])}{abs(comp['cracks']['change'])})")
        
        print(f"\n   Pavement:       {comp['pavement_condition']['base']:.1f} -> "
              f"{comp['pavement_condition']['present']:.1f} "
              f"({arrow(comp['pavement_condition']['change'])} "
              f"{abs(comp['pavement_condition']['change']):.1f})")
        
        print(f"   Markings:       {comp['marking_condition']['base']:.1f} -> "
              f"{comp['marking_condition']['present']:.1f} "
              f"({arrow(comp['marking_condition']['change'])} "
              f"{abs(comp['marking_condition']['change']):.1f})")
        
        print(f"\n   Signs:          {comp['road_signs']['base']} -> {comp['road_signs']['present']} "
              f"({arrow(comp['road_signs']['change'])}{abs(comp['road_signs']['change'])})")
        
        print(f"   Traffic Lights: {comp['traffic_lights']['base']} -> {comp['traffic_lights']['present']} "
              f"({arrow(comp['traffic_lights']['change'])}{abs(comp['traffic_lights']['change'])})")

        print(f"   Furniture:      {comp['furniture']['base']} -> {comp['furniture']['present']} "
              f"({arrow(comp['furniture']['change'])}{abs(comp['furniture']['change'])})")
        
        print(f"   VRUs:           {comp['vru']['base']} -> {comp['vru']['present']} "
              f"({arrow(comp['vru']['change'])}{abs(comp['vru']['change'])})")
        
        print(f"\n[INFO] FRAME-LEVEL CHANGES")
        print(f"   Frames with changes: {report['total_frames_with_changes']}")
        print(f"   Total issues: {report['issue_summary']['total_issues']}")
        
        severity = report['issue_summary']['by_severity']
        print(f"\n[INFO] BY SEVERITY")
        print(f"   High:     {severity['high']}")
        print(f"   Medium:   {severity['medium']}")
        print(f"   Low:      {severity['low']}")
        
        print(f"\n[INFO] QUALITY ASSURANCE")
        qm = report['quality_metrics']
        print(f"   Confidence threshold: {qm['confidence_threshold']}")
        print(f"   NMS (duplicate removal): {qm['nms_applied']}")
        print(f"   Frame-level tracking: {qm['duplicate_removal']}")
        print(f"   Guardrail Filter: {qm['guardrail_filter']}")
        print(f"   Deterioration Logic: {qm['deterioration_logic']}")
        
        print("\n" + "="*70)
    
    def run_complete_audit(self, base_video, present_video):
        """Run complete enhanced audit"""
        print("\n" + "="*70)
        print(" STARTING ENHANCED AUDIT")
        print("="*70)
        
        # Reset strike counters for a new run
        self.strike_counters = {'pavement': 0, 'markings': 0}

        fps = self.config['fps']
        batch_size = self.config['batch_size']
        
        print("\n[STEP 0/5] Validating videos...")
        base_metadata = self.extract_video_metadata(base_video)
        present_metadata = self.extract_video_metadata(present_video)
        
        if not self.validate_video_compatibility(base_metadata, present_metadata):
            print("\n[ERROR] Validation failed. Aborting.")
            return None
        
        print("\n[STEP 1/5] Extracting frames...")
        base_frames = self.extract_frames(base_video, 'data/base_frames', fps=fps)
        present_frames = self.extract_frames(present_video, 'data/present_frames', fps=fps)
        
        if not base_frames or not present_frames:
            print("[ERROR] Frame extraction failed")
            return None
        
        print("\n[STEP 2/5] Analyzing BASE video...")
        base_results = self._run_analysis_pipeline(base_frames, batch_size=batch_size)
        
        print("\n[STEP 3/5] Analyzing PRESENT video...")
        present_results = self._run_analysis_pipeline(present_frames, batch_size=batch_size)
        
        print("\n[STEP 4/5] Performing comparisons...")
        frame_level_changes = self._compare_frame_by_frame(base_results, present_results, fps)
        
        print(f"   [SUCCESS] Found changes in {len(frame_level_changes)} frames")
        
        if frame_level_changes:
            self.save_comparison_images(base_results, present_results, frame_level_changes)
        
        print("\n[STEP 5/5] Generating report...")
        report = self._generate_report(
            base_results, present_results, base_metadata,
            present_metadata, frame_level_changes, fps
        )
        
        Path('results').mkdir(exist_ok=True)
        report_path = 'results/audit_report_v9_enhanced.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[SUCCESS] Report saved: {report_path}")
        
        self._print_summary(report)
        
        return report