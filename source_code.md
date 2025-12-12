# streamlit_app.py
"""
Streamlit front-end with proper data clearing and consistency.
All visualizations now use the same authoritative JSON source.
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import random

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from penultimate_road_audit_system import EnhancedRoadAuditSystem
from irc_solution_generator import IRCSolutionGenerator
from latex_report_generator import LatexReportGenerator

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
COMPARISON_DIR = RESULTS_DIR / "comparisons"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_models():
    missing = []
    for name in ["best.pt", "yolov8s.pt", "road_markings_yolov8s-seg.pt"]:
        if not (MODELS_DIR / name).exists():
            missing.append(name)
    return missing


def clear_all_previous_data():
    """Clear ALL previous run data to ensure fresh results"""
    status = st.empty()
    status.info("Cleaning previous run data...")
    
    errors = []
    
    # Clear frame extraction folders
    for folder_name in ["base", "present"]:
        folder = DATA_DIR / folder_name
        if folder.exists():
            try:
                shutil.rmtree(folder, ignore_errors=True)
                # Wait a moment for OS to release locks
                import time
                time.sleep(0.1)
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Failed to clear {folder_name}: {e}")
    
    # Clear comparison images
    if COMPARISON_DIR.exists():
        try:
            # Delete individual files first (more reliable on Windows)
            for img in COMPARISON_DIR.glob("*.jpg"):
                try:
                    img.unlink()
                except:
                    pass
            # Try to remove directory
            shutil.rmtree(COMPARISON_DIR, ignore_errors=True)
        except Exception as e:
            errors.append(f"Failed to clear comparisons: {e}")
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear JSON outputs with retry logic
    for json_file in ["audit_output.json", "irc_output.json"]:
        file_path = RESULTS_DIR / json_file
        if file_path.exists():
            for attempt in range(3):
                try:
                    file_path.unlink()
                    break
                except Exception as e:
                    if attempt == 2:
                        errors.append(f"Failed to delete {json_file}: {e}")
                    import time
                    time.sleep(0.1)
    
    # Clear LaTeX files
    latex_patterns = ["report.tex", "report.pdf", "report.aux", "report.log", "report.out"]
    for pattern in latex_patterns:
        file_path = RESULTS_DIR / pattern
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                errors.append(f"Failed to delete {pattern}: {e}")
    
    # Clear engine logs
    log_file = RESULTS_DIR / "engine_trace.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception as e:
            errors.append(f"Failed to clear logs: {e}")
    
    if errors:
        status.warning(f"Cleanup completed with warnings:\n" + "\n".join(errors))
    else:
        status.success("Previous data cleared successfully")
    
    # Force a small delay to ensure OS catches up
    import time
    time.sleep(0.2)


# Session state initialization
if "last_meteo_error" not in st.session_state:
    st.session_state.last_meteo_error = None
if "engine_logs" not in st.session_state:
    st.session_state.engine_logs = []
if "last_audit_file" not in st.session_state:
    st.session_state.last_audit_file = str(RESULTS_DIR / "audit_output.json")
if "run_timestamp" not in st.session_state:
    st.session_state.run_timestamp = None
if "audit_completed" not in st.session_state:
    st.session_state.audit_completed = False
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None


def save_uploaded_video(uploaded, filename: str):
    if uploaded is None:
        return None
    target = UPLOAD_DIR / filename
    with open(target, "wb") as f:
        f.write(uploaded.read())
    return target


def safe_load_json(path: Path):
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load {path.name}: {e}")
        return None


def write_engine_log(msg: str):
    ts = datetime.now().isoformat()
    st.session_state.engine_logs.append(f"{ts} {msg}")
    with open(RESULTS_DIR / "engine_trace.log", "a", encoding="utf-8") as lf:
        lf.write(f"{ts} {msg}\n")


def build_heatmap_points(center_lat, center_lon, intensity_map, n_pts=150):
    """Generate realistic heatmap points based on intensity"""
    pts = []
    
    # Intensity scaling: T0->0.2, T1->0.6, T2->1.2, T3->2.0
    intensity_scale = intensity_map.get("scale", 0.8)
    spread = intensity_map.get("spread", 0.003)
    
    for _ in range(n_pts):
        # Create clustered distribution
        cluster_lat = center_lat + random.gauss(0, spread)
        cluster_lon = center_lon + random.gauss(0, spread)
        weight = max(0.1, random.gauss(intensity_scale, intensity_scale * 0.3))
        pts.append([cluster_lat, cluster_lon, weight])
    
    return pts


def create_gis_map(center, traffic_intensity, rainfall_intensity, zoom=14):
    """Create folium map with traffic and rainfall heatmaps"""
    m = folium.Map(location=center, zoom_start=zoom, tiles='OpenStreetMap')
    
    # Add center marker
    folium.Marker(
        center,
        popup=f"Audit Location<br>Lat: {center[0]:.6f}<br>Lon: {center[1]:.6f}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Traffic intensity mapping
    traffic_map = {
        "T0": {"scale": 0.3, "spread": 0.002, "color": "YlOrRd"},
        "T1": {"scale": 0.8, "spread": 0.003, "color": "YlOrRd"},
        "T2": {"scale": 1.4, "spread": 0.004, "color": "YlOrRd"},
        "T3": {"scale": 2.2, "spread": 0.005, "color": "YlOrRd"}
    }
    
    # Rainfall intensity mapping
    rain_map = {
        "R0": {"scale": 0.3, "spread": 0.002, "color": "Blues"},
        "R1": {"scale": 0.8, "spread": 0.003, "color": "Blues"},
        "R2": {"scale": 1.4, "spread": 0.004, "color": "Blues"},
        "R3": {"scale": 2.2, "spread": 0.005, "color": "Blues"}
    }
    
    # Generate traffic heatmap
    traffic_config = traffic_map.get(traffic_intensity, traffic_map["T1"])
    traffic_pts = build_heatmap_points(center[0], center[1], traffic_config, n_pts=180)
    
    traffic_group = folium.FeatureGroup(name="Traffic Density", show=True)
    HeatMap(
        traffic_pts,
        radius=18,
        blur=12,
        max_zoom=13,
        gradient={0.2: 'yellow', 0.5: 'orange', 0.8: 'red', 1.0: 'darkred'}
    ).add_to(traffic_group)
    traffic_group.add_to(m)
    
    # Generate rainfall heatmap
    rain_config = rain_map.get(rainfall_intensity, rain_map["R1"])
    rain_pts = build_heatmap_points(center[0], center[1], rain_config, n_pts=140)
    
    rain_group = folium.FeatureGroup(name="Rainfall Pattern", show=False)
    HeatMap(
        rain_pts,
        radius=16,
        blur=10,
        max_zoom=13,
        gradient={0.2: 'lightblue', 0.5: 'blue', 0.8: 'darkblue', 1.0: 'navy'}
    ).add_to(rain_group)
    rain_group.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m


# UI Layout
st.set_page_config(page_title="Road Safety Audit System", page_icon="🛣️", layout="wide")
st.title("Road Safety Audit System – Comparator Engine")
st.markdown("Upload **Base** and **Present** corridor videos for automated road safety audit with GIS analysis.")

missing_models = check_models()
if missing_models:
    st.warning("Missing models: " + ", ".join(missing_models))
else:
    st.success("All required models detected")

# Sidebar
st.sidebar.header("Configuration")
gps_mode = st.sidebar.radio("GPS Source", ("Video metadata", "Manual entry"))
manual_gps = None
if gps_mode == "Manual entry":
    lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.6f")
    manual_gps = {"latitude": float(lat), "longitude": float(lon)}

st.sidebar.markdown("---")
st.sidebar.subheader("GIS Parameters")
traffic_input = st.sidebar.selectbox(
    "Traffic Intensity",
    ["T0 (Low)", "T1 (Moderate)", "T2 (High)", "T3 (Very High)"],
    index=1
)
rainfall_input = st.sidebar.selectbox(
    "Rainfall Pattern",
    ["R0 (Dry)", "R1 (Light)", "R2 (Moderate)", "R3 (Heavy)"],
    index=1
)

# Extract intensity codes
traffic_code = traffic_input.split()[0]
rainfall_code = rainfall_input.split()[0]

st.sidebar.markdown("---")
st.sidebar.info(f"Device: **{get_device().upper()}**")

# Video upload
col1, col2 = st.columns(2)
with col1:
    base_video = st.file_uploader("📹 Upload Base Video", type=["mp4", "mov", "avi", "mkv"])
with col2:
    present_video = st.file_uploader("📹 Upload Present Video", type=["mp4", "mov", "avi", "mkv"])

run_btn = st.button("Run Complete Audit", type="primary", disabled=not (base_video and present_video))

# Run pipeline
if run_btn:
    # Clear all previous data first
    clear_all_previous_data()
    
    # VERIFY cleanup was successful
    audit_json_path = RESULTS_DIR / "audit_output.json"
    if audit_json_path.exists():
        st.error("Failed to clear old audit_output.json. Please close any programs using this file and try again.")
        st.stop()
    
    st.session_state.engine_logs = []
    st.session_state.run_timestamp = datetime.now().isoformat()
    st.session_state.audit_completed = False  # Reset completion status
    st.session_state.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    write_engine_log(f"[INIT] New audit run started - ID: {st.session_state.current_run_id}")
    
    with st.spinner("Saving uploaded videos..."):
        base_path = save_uploaded_video(base_video, "base_video.mp4")
        present_path = save_uploaded_video(present_video, "present_video.mp4")
    
    if not base_path or not present_path:
        st.error("Error saving videos")
        st.stop()
    
    st.subheader("Step 1: Running Audit Engine")
    progress = st.progress(0)
    
    try:
        system = EnhancedRoadAuditSystem({
            "pretrained_model": str(MODELS_DIR / "yolov8s.pt"),
            "finetuned_model": str(MODELS_DIR / "best.pt"),
            "segmentation_model": str(MODELS_DIR / "road_markings_yolov8s-seg.pt"),
            "proc_height": 640,
            "fps": 5,
            "min_confidence": 0.25,
        })
        
        progress.progress(10)
        write_engine_log("[ENGINE] Starting audit run...")
        
        report = system.run_complete_audit(str(base_path), str(present_path), manual_gps=manual_gps)
        
        # Add run metadata to report
        report["run_id"] = st.session_state.current_run_id
        report["run_timestamp"] = st.session_state.run_timestamp
        
        # Force write to disk
        audit_json_path = RESULTS_DIR / "audit_output.json"
        with open(audit_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Verify file was written
        if not audit_json_path.exists():
            st.error("Failed to save audit_output.json")
            st.stop()
        
        file_size = audit_json_path.stat().st_size
        write_engine_log(f"[ENGINE] Audit complete, JSON saved ({file_size} bytes)")
        st.success(f"Audit engine completed (saved {file_size:,} bytes)")
        
    except Exception as e:
        import traceback
        err = f"Audit failed: {e}\n{traceback.format_exc()}"
        write_engine_log(err)
        st.error(f"{err}")
        st.stop()
    
    progress.progress(70)
    
    # IRC Recommendations
    st.subheader("Step 2: Generating IRC Recommendations")
    try:
        irc_engine = IRCSolutionGenerator(str(RESULTS_DIR / "audit_output.json"))
        irc_output = irc_engine.generate()
        write_engine_log("[IRC] Recommendations generated")
        st.success("IRC recommendations generated")
    except Exception as e:
        write_engine_log(f"[IRC] Failed: {e}")
        st.warning(f"IRC generation failed: {e}")
    
    progress.progress(85)
    
    # LaTeX Report
    st.subheader("Step 3: Generating PDF Report")
    try:
        irc_path = RESULTS_DIR / "irc_output.json"
        report_gen = LatexReportGenerator(
            audit_json=str(RESULTS_DIR / "audit_output.json"),
            irc_json=str(irc_path) if irc_path.exists() else None
        )
        tex_path, pdf_path = report_gen.generate()
        
        if pdf_path and Path(pdf_path).exists():
            write_engine_log("[PDF] Generated successfully")
            st.success("PDF report generated")
        else:
            write_engine_log("[PDF] Not generated (pdflatex missing?)")
            st.warning("PDF not generated (install pdflatex)")
    except Exception as e:
        write_engine_log(f"[PDF] Failed: {e}")
        st.warning(f"PDF generation failed: {e}")
    
    progress.progress(100)
    st.balloons()
    st.success("**Audit Complete!** Results loaded below.")
    st.session_state.audit_completed = True  # Mark audit as complete

# Display Results (always from disk)
st.markdown("---")
audit_json_path = RESULTS_DIR / "audit_output.json"

# Only load if audit has been completed in this session
if st.session_state.audit_completed:
    audit_report = safe_load_json(audit_json_path)
else:
    audit_report = None

tabs = st.tabs(["Overview", "GIS Analysis", "Visuals", "Downloads"])

# Overview Tab
with tabs[0]:
    st.header("Audit Overview")
    
    if not st.session_state.audit_completed:
        st.info("Upload videos and click **'Run Complete Audit'** to begin analysis.")
        st.markdown("""
        ### What This System Does:
        - 🔍 Detects road defects (potholes, cracks, faded markings)
        - 📊 Calculates Pavement Condition Index (PCI)
        - 🗺️ Performs GIS analysis with traffic/rainfall heatmaps
        - 📋 Generates IRC maintenance recommendations
        - 📄 Creates comprehensive PDF reports
        """)
    elif not audit_report:
        st.error("Audit completed but results file missing. Please re-run the audit.")
    else:
        # Verify this is the current run's data
        report_run_id = audit_report.get("run_id", "unknown")
        if st.session_state.current_run_id and report_run_id != st.session_state.current_run_id:
            st.error(f"Data mismatch detected! Expected run ID: {st.session_state.current_run_id}, Got: {report_run_id}")
            st.warning("This indicates old data. Please re-run the audit.")
            st.stop()
        
        # Display run timestamp
        if st.session_state.run_timestamp:
            st.caption(f"Run timestamp: {st.session_state.run_timestamp}")
        
        # Debug: Show data source info
        if audit_json_path.exists():
            file_time = datetime.fromtimestamp(audit_json_path.stat().st_mtime).isoformat()
            file_size = audit_json_path.stat().st_size
            st.caption(f"Data source: {audit_json_path.name} (modified: {file_time}, size: {file_size:,} bytes)")
        
        # PCI Metrics
        pci = audit_report.get("pci_data", {})
        base_pci = pci.get("base", {})
        pres_pci = pci.get("present", {})
        delta_pci = pci.get("delta", 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Base PCI", base_pci.get("score", "-"), help=base_pci.get("rating", ""))
        col2.metric("Present PCI", pres_pci.get("score", "-"), help=pres_pci.get("rating", ""))
        col3.metric("PCI Change", f"{delta_pci:+d}" if isinstance(delta_pci, int) else delta_pci)
        
        # Aggregate comparison
        st.subheader("Defect Summary")
        agg = audit_report.get("aggregate_comparison", {})
        
        if agg:
            rows = []
            for defect_type, counts in agg.items():
                rows.append({
                    "Defect": defect_type.replace("_", " ").title(),
                    "Base": counts.get("base", 0),
                    "Present": counts.get("present", 0),
                    "Change": counts.get("delta", 0)
                })
            
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")
            
            # Bar chart
            chart_df = df.melt(id_vars=["Defect"], value_vars=["Base", "Present"], 
                              var_name="Period", value_name="Count")
            fig = px.bar(chart_df, x="Defect", y="Count", color="Period", 
                        barmode="group", title="Base vs Present Comparison")
            st.plotly_chart(fig, width="stretch")

# GIS Tab
with tabs[1]:
    st.header("GIS Context & Heatmaps")
    
    if not st.session_state.audit_completed:
        st.info("Run an audit first to view GIS analysis and heatmaps.")
    elif not audit_report:
        st.error("Audit data missing. Please re-run the audit.")
    else:
        gps = audit_report.get("gps", {})
        if not gps.get("latitude"):
            st.warning("GPS coordinates not found")
            center = (28.6139, 77.2090)
        else:
            center = (gps["latitude"], gps["longitude"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Location & Intensity Heatmaps")
            st.markdown(f"**Coordinates:** {center[0]:.6f}, {center[1]:.6f}")
            st.markdown(f"**Traffic:** {traffic_input} | **🌧️ Rainfall:** {rainfall_input}")
            
            # Create and display map
            gis_map = create_gis_map(center, traffic_code, rainfall_code)
            st_folium(gis_map, width=700, height=500)
        
        with col2:
            st.subheader("📋 GIS Profile")
            gis_profile = audit_report.get("gis_profile", {})
            if gis_profile:
                for key, value in gis_profile.items():
                    st.metric(key.replace("_", " ").title(), value)
            else:
                st.info("No GIS profile data")

# Visuals Tab
with tabs[2]:
    st.header("Visual Analysis")
    
    if not st.session_state.audit_completed:
        st.info("Run an audit first to view visual analysis and comparison images.")
    elif not audit_report:
        st.error("Audit data missing. Please re-run the audit.")
    else:
        # Frame evolution chart
        st.subheader("Defect Evolution (Present Video)")
        present_frames = audit_report.get("present_frame_data", [])
        
        if present_frames:
            frame_data = []
            for f in present_frames:
                frame_data.append({
                    "Frame": f.get("frame_idx", 0),
                    "Potholes": len(f.get("potholes", [])),
                    "Cracks": len(f.get("cracks", [])),
                    "Marking Wear %": f.get("markings", {}).get("marking_wear_pct", 0)
                })
            
            df = pd.DataFrame(frame_data)
            fig = px.line(df, x="Frame", y=["Potholes", "Cracks", "Marking Wear %"],
                         title="Defect Trends Across Video Frames")
            st.plotly_chart(fig, width="stretch")
        
        # Comparison images
        st.subheader("🔄 Before/After Comparisons")
        comp_images = sorted(COMPARISON_DIR.glob("comp_*.jpg"))
        
        if not comp_images:
            st.info("No comparison images generated")
        else:
            cols = st.columns(3)
            for idx, img in enumerate(comp_images[:15]):
                with cols[idx % 3]:
                    st.image(str(img), caption=f"Frame {img.stem.replace('comp_', '')}", 
                            width="stretch")

# Downloads Tab
with tabs[3]:
    st.header("Downloads & Logs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Download Files")
        
        # Audit JSON
        if (RESULTS_DIR / "audit_output.json").exists():
            with open(RESULTS_DIR / "audit_output.json", "rb") as f:
                st.download_button(
                    "⬇audit_output.json",
                    f,
                    file_name="audit_output.json",
                    mime="application/json"
                )
        
        # IRC JSON
        if (RESULTS_DIR / "irc_output.json").exists():
            with open(RESULTS_DIR / "irc_output.json", "rb") as f:
                st.download_button(
                    "⬇irc_output.json",
                    f,
                    file_name="irc_output.json",
                    mime="application/json"
                )
        
        # PDF
        pdf_file = RESULTS_DIR / "report.pdf"
        if pdf_file.exists():
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "⬇Final PDF Report",
                    f,
                    file_name="Road_Audit_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("PDF not available (requires pdflatex)")
    
    with col2:
        st.subheader("Engine Logs")
        
        log_file = RESULTS_DIR / "engine_trace.log"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                logs = f.read()
                st.text_area("Recent logs", logs[-2000:], height=300)
        else:
            st.info("No logs available")

st.markdown("---")
st.caption("© 2025 Road Safety Audit System | Powered by YOLOv8 + IRC Guidelines")







# penultimate_road_audit_system.py

Final merged comparator engine (Option A features included).
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
                print(f"[INIT] Loaded base model (fallback): {self.config['pretrained_model']}")
            else:
                self.base_model = None
                print("[INIT][WARN] Base model not found; fallback disabled.")
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
    def _collate_frame_results(self, frame_path, custom_result, seg_result_unused, gis_profile, frame_idx):
        img = cv2.imread(frame_path)
        if img is None:
            self._log(f"[FRAME][WARN] Could not read frame image: {frame_path}")
            return None

        self.pci_stats["total_frames"] += 1

        markings_info, marking_dets = self._analyze_markings(img)

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
            "marking_detections": marking_dets,
        }

        if markings_info["marking_wear_pct"] > 50:
            self.pci_stats["faded_marks"] += 1

        if custom_result is not None and getattr(custom_result, "boxes", None) is not None:
            for box in custom_result.boxes:
                try:
                    conf = float(box.conf[0])
                except Exception:
                    conf = float(getattr(box, "conf", 0) or 0)
                if conf < self.config["min_confidence"]:
                    continue

                try:
                    cls_idx = int(box.cls[0])
                    raw_label = str(self.custom_model.names.get(cls_idx, f"class_{cls_idx}"))
                except Exception:
                    raw_label = str(getattr(box, "cls", "unknown"))

                try:
                    bbox = box.xyxy[0].tolist()
                except Exception:
                    bbox = [0, 0, 0, 0]

                det = {"label": raw_label, "confidence": conf, "bbox": bbox, "occluded": False}
                lower = raw_label.lower()

                # guardrail occlusion
                if "guardrail" in lower or "guard rail" in lower:
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

                # potholes
                if "pothole" in lower:
                    det["root_cause"] = self.rca.determine_cause("pothole", {}, gis_profile)
                    frame_data["potholes"].append(det)
                    self.pci_stats["potholes"] += 1

                # cracks
                elif "crack" in lower:
                    width_px = self._measure_crack_width_px(img, bbox)
                    width_cm = width_px * self.pixel_to_cm_scale
                    det["width_px"] = width_px
                    det["width_cm"] = width_cm
                    metrics = {"crack_width_cm": width_cm}
                    det["root_cause"] = self.rca.determine_cause("crack", metrics, gis_profile)
                    frame_data["cracks"].append(det)
                    self.pci_stats["crack_len_px"] += width_px

                # speed breakers
                elif "speed" in lower and "breaker" in lower:
                    frame_data["furniture"].append(det)

                # signs
                elif "sign" in lower:
                    det["root_cause"] = self.rca.determine_cause("damaged_sign", {}, gis_profile)
                    frame_data["road_signs"].append(det)

                # traffic lights
                elif "trafficlight" in lower or "signal" in lower:
                    frame_data["traffic_lights"].append(det)

                # street lights
                elif "streetlight" in lower:
                    frame_data["furniture"].append(det)

                # guardrails
                elif "guardrail" in lower:
                    det["root_cause"] = self.rca.determine_cause("broken_guardrail", {}, gis_profile)
                    frame_data["furniture"].append(det)

                else:
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
        """
        Improved suppression:
         - If same key was seen within cooldown frames -> suppress
         - Update last_frame and suppressed_until appropriately
         - Keep recent_events small
        """
        cooldown = int(self.config.get("change_cooldown_frames", 3))
        for ev in self.recent_events:
            if ev["key"] == event_key:
                # If event occurred recently (within cooldown) -> suppress
                if (current_frame - ev.get("last_frame", -9999)) <= cooldown:
                    # keep last_frame unchanged (still last seen)
                    return True
                # otherwise update last_frame and allow event
                ev["last_frame"] = current_frame
                return False

        # new event -> record and allow
        self.recent_events.append({"key": event_key, "last_frame": current_frame})
        # trim
        if len(self.recent_events) > 2000:
            self.recent_events = self.recent_events[-1000:]
        return False


    # ----------------- comparison (with occlusion awareness) -----------------
    def _compare_frame_by_frame(self, base_results, present_results, fps):
        changes = []
        min_len = min(len(base_results), len(present_results))
        for i in range(min_len):
            base = base_results[i]
            pres = present_results[i]
            frame_log = []
            # pothole increase
            if len(pres["potholes"]) > len(base["potholes"]):
                event = {"element": "Pothole", "type": "New pothole(s) observed", "severity": "high"}
                key = self._event_key_from_change(event)
                if self._should_suppress_event(key, i):
                    self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                else:
                    frame_log.append(event)
            # crack increase
            if len(pres["cracks"]) > len(base["cracks"]):
                event = {"element": "Cracks", "type": "New or widened cracks observed", "severity": "medium"}
                key = self._event_key_from_change(event)
                if self._should_suppress_event(key, i):
                    self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                else:
                    frame_log.append(event)
            # marking wear
            if pres["markings"]["marking_wear_pct"] > base["markings"]["marking_wear_pct"] + 20:
                event = {"element": "Markings", "type": "Additional fading or loss of markings", "severity": "medium"}
                key = self._event_key_from_change(event)
                if self._should_suppress_event(key, i):
                    self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                else:
                    frame_log.append(event)
            # signs missing
            if len(pres["road_signs"]) < len(base["road_signs"]):
                event = {"element": "Road Signs", "type": "Possible missing or damaged sign(s)", "severity": "high"}
                key = self._event_key_from_change(event)
                if self._should_suppress_event(key, i):
                    self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                else:
                    frame_log.append(event)
            # roadside furniture (guardrail occlusion aware)
            if len(pres["furniture"]) < len(base["furniture"]):
                occlusion_suppressed = False
                for b_det in base["furniture"]:
                    if "guardrail" in b_det.get("label", "").lower():
                        key_coords = f"guardrail::{int(b_det['bbox'][0])}_{int(b_det['bbox'][1])}_{int(b_det['bbox'][2])}_{int(b_det['bbox'][3])}"
                        state = self.occlusion_state.get(key_coords)
                        if state and state.get("occluded_frames", 0) >= self.config["occlusion_persistence_frames"]:
                            occlusion_suppressed = True
                            self.suppressed_events.append({"frame": i, "event": {"element": "Roadside Furniture", "type": "Possible missing guardrail (occluded by vegetation)"}, "reason": "occlusion"})
                            break
                if not occlusion_suppressed:
                    event = {"element": "Roadside Furniture", "type": "Missing or damaged roadside asset(s)", "severity": "high"}
                    key = self._event_key_from_change(event)
                    if self._should_suppress_event(key, i):
                        self.suppressed_events.append({"frame": i, "event": event, "reason": "cooldown"})
                    else:
                        frame_log.append(event)
            if frame_log:
                changes.append({"frame_id": i, "timestamp_seconds": i / fps if fps > 0 else 0.0, "changes": frame_log, "base_frame": base.get("frame"), "present_frame": pres.get("frame")})
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
            try:
                custom_res = self.custom_model(img, verbose=False, device=self.device)[0]
            except Exception as e:
                self._log(f"[BASE][ERROR] Custom model inference failed on {f}: {e}")
                continue
            frame_data = self._collate_frame_results(f, custom_res, None, self.global_gis_profile, idx)
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
            try:
                custom_res = self.custom_model(img, verbose=False, device=self.device)[0]
            except Exception as e:
                self._log(f"[PRESENT][ERROR] Custom model inference failed on {f}: {e}")
                continue
            frame_data = self._collate_frame_results(f, custom_res, None, self.global_gis_profile, idx)
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








# irc_solution_generator.py

Generates IRC-based recommendations from audit_output.json.
Fully compatible with:
 - penultimate_road_audit_system.py
 - streamlit_app.py
 - cli_runner.py

Key improvements:
 - Implements all prior rules + added guardrail, drainage, signage, prioritization rules
 - Includes IRC code references in suggested actions (where applicable)
 - Defensive: handles missing keys gracefully, never crashes on partial JSON
 - Produces 'results/irc_output.json' and returns the report dict
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Helpful typing aliases
JSONDict = Dict[str, Any]


class IRCSolutionGenerator:
    def __init__(self, audit_json_path: str):
        self.audit_json_path = Path(audit_json_path)
        if not self.audit_json_path.exists():
            raise FileNotFoundError(f"Audit file not found: {audit_json_path}")

        with open(self.audit_json_path, "r", encoding="utf-8") as f:
            try:
                self.audit: JSONDict = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON: {e}")

        self.recommendations: List[JSONDict] = []

    # ---------------------------
    # Utility helpers
    # ---------------------------
    def _safe_get(self, path: List[str], default=None):
        """Safe nested dict getter: path is list of keys."""
        node = self.audit
        for p in path:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def _append_rec(self, issue: str, severity: str, count: int, actions: List[str], priority: str = "Normal", notes: str = ""):
        rec = {
            "issue": issue,
            "severity": severity,
            "count": count,
            "priority": priority,
            "suggested_actions": actions,
            "notes": notes,
        }
        self.recommendations.append(rec)

    # ---------------------------
    # Rule: Potholes
    # ---------------------------
    def rule_potholes(self):
        # We use aggregate_comparison.potholes.present and delta
        base = self._safe_get(["aggregate_comparison", "potholes", "base"], 0) or 0
        present = self._safe_get(["aggregate_comparison", "potholes", "present"], 0) or 0
        delta = self._safe_get(["aggregate_comparison", "potholes", "delta"], present - base)
        count = int(present)

        if count <= 0 and (delta is None or int(delta) <= 0):
            return

        # Severity & actions mapping (IRC references included)
        if delta <= 0:
            severity = "None"
            actions = [
                "No new potholes detected. Maintain periodic inspection schedule.",
            ]
            priority = "Low"
        elif delta <= 2:
            severity = "Moderate"
            actions = [
                "Perform immediate temporary patching with bituminous cold mix for safety (Short-term).",
                "Schedule permanent patch (hot mix) within 30 days — see IRC:SP:100 (sections on patching methodology).",
                "Record GPS locations and photograph pothole extents for contract works."
            ]
            priority = "Medium"
        else:
            severity = "High"
            actions = [
                "Full-depth patching recommended (remove failed material, replace with hot mix). Follow IRC:SP:100 guidelines for permanent patches.",
                "Inspect and repair subgrade/drainage near pothole locations before resurfacing to avoid recurrence.",
                "If multiple clusters are present, consider sectional resurfacing or overlay (assess structural condition)."
            ]
            priority = "High"

        notes = f"Base count={base}, Present count={present}, Delta={delta}."
        self._append_rec("Potholes", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule: Cracks
    # ---------------------------
    def rule_cracks(self):
        base = self._safe_get(["aggregate_comparison", "cracks", "base"], 0) or 0
        present = self._safe_get(["aggregate_comparison", "cracks", "present"], 0) or 0
        delta = self._safe_get(["aggregate_comparison", "cracks", "delta"], present - base)
        count = int(present)

        if count <= 0 and (delta is None or int(delta) <= 0):
            return

        # Use crack width statistics if available in frame data (take max width observed)
        max_crack_width_cm = 0.0
        frames = self._safe_get(["present_frame_data"], []) or []
        for f in frames:
            for c in f.get("cracks", []):
                w = c.get("width_cm") or c.get("width_cm", 0) or c.get("width_px", 0)
                try:
                    w = float(w)
                except Exception:
                    w = 0.0
                if w > max_crack_width_cm:
                    max_crack_width_cm = w

        # Severity based on delta and width
        if delta <= 0:
            severity = "None"
            actions = ["No crack progression detected. Continue monitoring."]
            priority = "Low"
        else:
            if max_crack_width_cm >= 6 or delta > 5:
                severity = "High"
                actions = [
                    "Widespread/wide cracks detected. Consider surface renewal or overlay depending on structural evaluation.",
                    "If cracks indicate fatigue, plan for milling + resurfacing (see IRC:SP:76 / IRC guidelines on overlays).",
                    "Conduct structural investigation of pavement if subgrade issues suspected."
                ]
                priority = "High"
            else:
                severity = "Moderate"
                actions = [
                    "Apply crack sealing using elastomeric sealants after cleaning (refer IRC:116 for procedures).",
                    "Ensure cracks are cleaned and dried before sealant application."
                ]
                priority = "Medium"

        notes = f"Base={base}, Present={present}, Delta={delta}, Max crack width (cm)={max_crack_width_cm:.2f}"
        self._append_rec("Cracks", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule: Faded / Missing Markings
    # ---------------------------
    def rule_faded_markings(self):
        base = self._safe_get(["aggregate_comparison", "faded_marking_frames", "base"], 0) or 0
        present = self._safe_get(["aggregate_comparison", "faded_marking_frames", "present"], 0) or 0
        delta = self._safe_get(["aggregate_comparison", "faded_marking_frames", "delta"], present - base)
        count = int(present)

        if count <= 0 and (delta is None or int(delta) <= 0):
            return

        # Severity heuristics
        if delta <= 0:
            severity = "None"
            actions = ["Markings stable. Maintain current repaint cycle."]
            priority = "Low"
        elif delta <= 20:
            severity = "Moderate"
            actions = [
                "Repaint lane & center markings using thermoplastic paint with glass beads to meet retro-reflectivity (IRC:35).",
                "Prioritize critical locations (intersections, pedestrian crossings, school zones).",
            ]
            priority = "Medium"
        else:
            severity = "High"
            actions = [
                "Major loss of marking visibility—perform full re-marking with thermoplastic materials and glass beads as per IRC:35.",
                "If budget permits, consider high-durability thermoplastic or premix systems (longer service life).",
                "Improve nighttime retro-reflectivity testing schedule after re-marking."
            ]
            priority = "High"

        notes = f"Base faded frames={base}, Present faded frames={present}, Delta={delta}."
        self._append_rec("Road Markings", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule: Roadside Signage & Furniture (signs, lights)
    # ---------------------------
    def rule_roadside_furniture(self):
        # For signage, check aggregate if available (we use 'road_signs' delta if present)
        # fallback: use furniture delta (difference in counts)
        base_signs = self._safe_get(["aggregate_comparison", "road_signs", "base"], None)
        present_signs = self._safe_get(["aggregate_comparison", "road_signs", "present"], None)
        delta_signs = None
        if base_signs is not None and present_signs is not None:
            try:
                delta_signs = int(present_signs) - int(base_signs)
            except Exception:
                delta_signs = None

        # If signage aggregate isn't present, attempt to detect via furniture delta
        base_f = self._safe_get(["aggregate_comparison", "furniture", "base"], None)
        present_f = self._safe_get(["aggregate_comparison", "furniture", "present"], None)
        delta_f = None
        if base_f is not None and present_f is not None:
            try:
                delta_f = int(present_f) - int(base_f)
            except Exception:
                delta_f = None

        missing = 0
        if delta_signs is not None and delta_signs < 0:
            missing = abs(delta_signs)
            reason = "signs"
        elif delta_f is not None and delta_f < 0:
            missing = abs(delta_f)
            reason = "furniture"
        else:
            missing = 0
            reason = ""

        if missing <= 0:
            return

        actions = [
            "Replace damaged/missing signs per IRC:67 standards (retro-reflective sheeting, correct sizing).",
            "Use Type-XI or appropriate reflective sheeting for high-speed corridors as per IRC recommendations.",
            "Inspect mounting posts and foundations; repair or replace to restore correct sight distance and stability."
        ]
        notes = f"Missing count inferred from {reason} delta."

        self._append_rec("Roadside Signage / Furniture", "High", missing, actions, "High", notes)

    # ---------------------------
    # Rule: Guardrails & Occlusion (vegetation)
    # ---------------------------
    def rule_guardrails(self):
        # Look into per-frame furniture items for guardrail labels & occlusion flags
        present_frames = self._safe_get(["present_frame_data"], []) or []
        guardrail_total = 0
        guardrail_occluded = 0

        for f in present_frames:
            for det in f.get("furniture", []) + f.get("road_signs", []):
                label = str(det.get("label", "")).lower()
                if "guardrail" in label or "guard rail" in label:
                    guardrail_total += 1
                    if det.get("occluded", False) or det.get("occlusion_vegetation_fraction", 0) > 0.05:
                        guardrail_occluded += 1

        if guardrail_total == 0 and guardrail_occluded == 0:
            return

        # If occluded frames are significant, recommend vegetation management
        if guardrail_occluded > 0:
            actions = [
                "Vegetation trimming/clearance recommended to restore guardrail visibility and inspect for damage.",
                "If guardrail cannot be inspected due to persistent occlusion, schedule manual inspection after clearance.",
                "Where guardrail is missing or structurally damaged, replace per IRC:96 guidelines and realign as necessary."
            ]
            notes = f"Detected {guardrail_occluded} occluded guardrail detections across frames; total guardrail detections={guardrail_total}."
            priority = "Medium" if guardrail_occluded < 3 else "High"
            self._append_rec("Guardrails (Occluded / Possibly Damaged)", "Medium", guardrail_occluded, actions, priority, notes)

    # ---------------------------
    # Rule: Drainage & Surface (inferred from potholes/crack patterns)
    # ---------------------------
    def rule_drainage_and_surface(self):
        # Infer from GIS profile (drainage_quality) and pothole/crack density
        gis = self._safe_get(["gis_profile"], {}) or {}
        drainage = gis.get("drainage_quality", "Unknown")
        pothole_delta = int(self._safe_get(["aggregate_comparison", "potholes", "delta"], 0) or 0)
        crack_delta = int(self._safe_get(["aggregate_comparison", "cracks", "delta"], 0) or 0)

        if drainage in ["Poor", "Blocked"] or pothole_delta > 3 or crack_delta > 5:
            actions = [
                f"Investigate roadside drainage condition (current rating: {drainage}). Clear blockages, regrade channels and repair inlets as needed.",
                "If water ingress is evident under pavement, plan sub-surface drainage improvement prior to resurfacing (reduce recurrence).",
                "Coordinate drainage repairs with pavement rehabilitation works to maximise life-cycle benefit."
            ]
            severity = "High" if drainage in ["Poor", "Blocked"] or pothole_delta > 5 else "Medium"
            notes = f"Drainage rating={drainage}; pothole delta={pothole_delta}; crack delta={crack_delta}."
            priority = "High" if severity == "High" else "Medium"
            self._append_rec("Drainage & Pavement Moisture Issues", severity, 1, actions, priority, notes)

    # ---------------------------
    # Post-process: Prioritization & Summary
    # ---------------------------
    def rule_prioritization(self):
        # Add an overall priority summary (High if any High)
        priorities = {"High": 0, "Medium": 0, "Low": 0}
        for r in self.recommendations:
            p = r.get("priority", "Normal")
            if p in priorities:
                priorities[p] += 1
            elif p.lower() == "high":
                priorities["High"] += 1
            elif p.lower() == "medium":
                priorities["Medium"] += 1
            else:
                priorities["Low"] += 1

        overall_priority = "Low"
        if priorities["High"] > 0:
            overall_priority = "High"
        elif priorities["Medium"] > 0:
            overall_priority = "Medium"

        return {
            "overall_priority": overall_priority,
            "counts": priorities,
        }

    # ---------------------------
    # Main generator
    # ---------------------------
    def generate(self) -> JSONDict:
        self.recommendations = []

        # Execute rules; order matters for sensible output
        try:
            self.rule_potholes()
            self.rule_cracks()
            self.rule_faded_markings()
            self.rule_guardrails()
            self.rule_roadside_furniture()
            self.rule_drainage_and_surface()
        except Exception as e:
            # Defensive: don't crash generation for unexpected structure
            self.recommendations.append({
                "issue": "RuleEngineError",
                "severity": "High",
                "count": 0,
                "priority": "High",
                "suggested_actions": [],
                "notes": f"Rule engine encountered an exception: {str(e)}"
            })

        # Prioritization summary
        priority_summary = self.rule_prioritization()

        irc_report: JSONDict = {
            "generated_on": datetime.now().isoformat(),
            "source_audit": str(self.audit_json_path),
            "gps": self.audit.get("gps", {}),
            "gis_profile": self.audit.get("gis_profile", {}),
            "pci_summary": self.audit.get("pci_data", {}),
            "aggregate_comparison": self.audit.get("aggregate_comparison", {}),
            "recommendations": self.recommendations,
            "priority_summary": priority_summary,
            "notes": "Recommendations reference general IRC guidelines (e.g., IRC:35, IRC:67, IRC:SP:100, IRC:116). Adapt to local contract specifications as needed.",
        }

        # Write to results/irc_output.json (safe)
        out_path = Path("results") / "irc_output.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(irc_report, f, indent=2, ensure_ascii=False)

        return irc_report


# If run as script, quick smoke-test (does not execute unless invoked)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python irc_solution_generator.py <path_to_audit_json>")
        print("This will write results/irc_output.json")
        sys.exit(1)

    gen = IRCSolutionGenerator(sys.argv[1])
    report = gen.generate()
    print("Generated IRC recommendations (results/irc_output.json).")








# latex_report_generator.py
-------------------------------------------------------------------
Ensures fresh PDF generation every time with proper cleanup.
"""

import json
from pathlib import Path
import subprocess
from datetime import datetime
import glob
import shutil


class LatexReportGenerator:
    def __init__(self, audit_json, irc_json=None):
        self.audit_json = Path(audit_json)
        if not self.audit_json.exists():
            raise FileNotFoundError(f"Audit JSON missing: {audit_json}")

        if irc_json and Path(irc_json).exists():
            self.irc_json = Path(irc_json)
        else:
            self.irc_json = None

        # Load audit JSON
        with open(self.audit_json, "r", encoding="utf-8") as f:
            self.audit = json.load(f)

        # Load IRC JSON
        if self.irc_json:
            with open(self.irc_json, "r", encoding="utf-8") as f:
                self.irc = json.load(f)
        else:
            self.irc = {"recommendations": [], "priority_summary": {}}

        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)

        # Use timestamp to ensure uniqueness
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tex_path = self.output_dir / "report.tex"
        self.pdf_path = self.output_dir / "report.pdf"

    def _cleanup_old_files(self):
        """Remove old LaTeX auxiliary files and PDF - with verification"""
        patterns = ["report.*"]
        failed_deletes = []
        
        for pattern in patterns:
            for f in self.output_dir.glob(pattern):
                if not f.is_file():
                    continue
                
                # Try multiple times (Windows file locks)
                deleted = False
                for attempt in range(3):
                    try:
                        f.unlink()
                        deleted = True
                        break
                    except Exception as e:
                        if attempt == 2:
                            failed_deletes.append((str(f), str(e)))
                        import time
                        time.sleep(0.1)
        
        if failed_deletes:
            print("[LaTeX][WARN] Could not delete some files:")
            for fname, error in failed_deletes:
                print(f"  - {fname}: {error}")
            print("[LaTeX] This may cause PDF regeneration issues.")

    def _escape(self, text):
        """Properly escape LaTeX special characters"""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        # Replace in specific order to avoid double-escaping
        replacements = [
            ("\\", "\\textbackslash{}"),
            ("{", "\\{"),
            ("}", "\\}"),
            ("$", "\\$"),
            ("&", "\\&"),
            ("%", "\\%"),
            ("_", "\\_"),
            ("#", "\\#"),
            ("~", "\\textasciitilde{}"),
            ("^", "\\textasciicircum{}"),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text

    def _latex_image_block(self):
        """Embed comparison images"""
        out = ""
        comparison_dir = Path("results/comparisons")
        
        if not comparison_dir.exists():
            return "\\textit{No comparison images found.}\n"
        
        images = sorted(comparison_dir.glob("comp_*.jpg"))

        if not images:
            return "\\textit{No comparison images found.}\n"

        # Limit to 30 images to keep PDF reasonable
        for img in images[:30]:
            # Convert to relative path from results/ directory and normalize
            try:
                # Get path relative to results directory
                img_rel = img.relative_to(self.output_dir.parent)
                # Use forward slashes for LaTeX (works on Windows and Unix)
                img_path = str(img_rel).replace("\\", "/")
            except ValueError:
                # Fallback if relative path fails
                img_path = str(img).replace("\\", "/")
            
            out += (
                "\\begin{figure}[h!]\n"
                "\\centering\n"
                f"\\includegraphics[width=0.92\\textwidth]{{{img_path}}}\n"
                "\\caption{Before/After Comparison}\n"
                "\\end{figure}\n"
                "\\clearpage\n\n"
            )

        return out

    def build_tex(self):
        """Build complete LaTeX document"""
        gps = self.audit.get("gps", {})
        pci = self.audit.get("pci_data", {})
        gis = self.audit.get("gis_profile", {})
        agg = self.audit.get("aggregate_comparison", {})
        irc_recs = self.irc.get("recommendations", [])
        priority_summary = self.irc.get("priority_summary", {})
        
        # Get current timestamp
        report_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")

        # Document preamble
        tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{float}
\geometry{margin=0.8in}

\begin{document}

\title{\textbf{Road Safety Audit Report}}
\author{Automated Road Audit System}
\date{""" + report_date + r"""}
\maketitle

\tableofcontents
\clearpage

"""

        # GPS Location Section
        tex += r"""
\section{GPS Location}
\begin{tabular}{ll}
\textbf{Latitude:} & """ + self._escape(str(gps.get("latitude", "N/A"))) + r""" \\
\textbf{Longitude:} & """ + self._escape(str(gps.get("longitude", "N/A"))) + r""" \\
\end{tabular}

"""

        # GIS Profile Section
        tex += r"""
\section{GIS Profile}
"""
        if gis:
            tex += r"""\begin{longtable}{|l|p{10cm}|}
\hline
\textbf{Attribute} & \textbf{Value} \\ \hline
\endhead
"""
            for k, v in gis.items():
                tex += f"{self._escape(k.replace('_', ' ').title())} & {self._escape(str(v))} \\\\ \\hline\n"
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No GIS profile data available.}\n\n"

        # PCI Section
        tex += r"""
\section{Pavement Condition Index (PCI)}
"""
        base_pci = pci.get("base", {})
        pres_pci = pci.get("present", {})
        delta_pci = pci.get("delta", 0)
        
        tex += r"""\begin{tabular}{|l|l|}
\hline
\textbf{Metric} & \textbf{Value} \\ \hline
Base PCI Score & """ + str(base_pci.get("score", "-")) + r""" \\
Base Rating & """ + self._escape(str(base_pci.get("rating", "-"))) + r""" \\ \hline
Present PCI Score & """ + str(pres_pci.get("score", "-")) + r""" \\
Present Rating & """ + self._escape(str(pres_pci.get("rating", "-"))) + r""" \\ \hline
\textbf{Delta (Change)} & \textbf{""" + str(delta_pci) + r"""} \\ \hline
\end{tabular}

"""

        # Aggregate Comparison
        tex += r"""
\section{Aggregate Defect Comparison}
"""
        if agg:
            tex += r"""\begin{longtable}{|l|c|c|c|}
\hline
\textbf{Defect Type} & \textbf{Base} & \textbf{Present} & \textbf{Delta} \\ \hline
\endhead
"""
            for defect_name, counts in agg.items():
                label = self._escape(defect_name.replace("_", " ").title())
                base_val = counts.get("base", 0)
                pres_val = counts.get("present", 0)
                delta_val = counts.get("delta", 0)
                tex += f"{label} & {base_val} & {pres_val} & {delta_val} \\\\ \\hline\n"
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No aggregate comparison data available.}\n\n"

        # IRC Recommendations
        tex += r"""
\section{IRC Maintenance Recommendations}
"""
        if irc_recs:
            tex += r"""\begin{longtable}{|p{3cm}|p{2cm}|p{1.8cm}|p{7cm}|}
\hline
\textbf{Issue} & \textbf{Severity} & \textbf{Priority} & \textbf{Suggested Actions} \\ \hline
\endhead
"""
            for rec in irc_recs:
                issue = self._escape(rec.get("issue", ""))
                severity = self._escape(rec.get("severity", ""))
                priority = self._escape(rec.get("priority", ""))
                
                actions = rec.get("suggested_actions", [])
                actions_text = " \\newline ".join([self._escape(a) for a in actions])
                
                tex += f"{issue} & {severity} & {priority} & {actions_text} \\\\ \\hline\n"
            
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No IRC recommendations available.}\n\n"

        # Priority Summary
        tex += r"""
\section{Maintenance Priority Summary}
"""
        if priority_summary:
            counts = priority_summary.get("counts", {})
            tex += r"""\begin{tabular}{|l|c|}
\hline
\textbf{Priority Level} & \textbf{Count} \\ \hline
Overall Priority & """ + self._escape(str(priority_summary.get("overall_priority", "N/A"))) + r""" \\ \hline
High Priority Items & """ + str(counts.get("High", 0)) + r""" \\ \hline
Medium Priority Items & """ + str(counts.get("Medium", 0)) + r""" \\ \hline
Low Priority Items & """ + str(counts.get("Low", 0)) + r""" \\ \hline
\end{tabular}

"""
        else:
            tex += "\\textit{No priority summary available.}\n\n"

        # Frame-level changes
        tex += r"""
\section{Frame-Level Deterioration Summary}
"""
        changes = self.audit.get("frame_level_changes", [])
        if changes:
            # Limit to first 100 changes for readability
            tex += r"""\begin{longtable}{|c|c|p{9cm}|}
\hline
\textbf{Frame} & \textbf{Time (s)} & \textbf{Changes Detected} \\ \hline
\endhead
"""
            for change in changes[:100]:
                frame_id = change.get("frame_id", "-")
                timestamp = round(change.get("timestamp_seconds", 0), 2)
                
                change_items = change.get("changes", [])
                change_desc = " \\newline ".join([
                    f"{self._escape(c.get('element', ''))}: {self._escape(c.get('type', ''))}"
                    for c in change_items
                ])
                
                tex += f"{frame_id} & {timestamp} & {change_desc} \\\\ \\hline\n"
            
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No frame-level deterioration detected.}\n\n"

        # Comparison images
        tex += r"""
\section{Before/After Comparison Images}
"""
        tex += self._latex_image_block()

        # End document
        tex += r"""
\end{document}
"""

        return tex

    def generate(self):
        """Generate LaTeX and compile to PDF"""
        
        # Clean up old files first
        self._cleanup_old_files()
        
        # Build LaTeX content
        print("[LaTeX] Building document...")
        tex_content = self.build_tex()

        # Write .tex file
        with open(self.tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        
        print(f"[LaTeX] Written to {self.tex_path}")

        # Try to compile with pdflatex
        try:
            print("[LaTeX] Compiling PDF (this may take a moment)...")
            
            # Run pdflatex twice for proper cross-references
            for run in [1, 2]:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", str(self.tex_path.name)],
                    cwd=str(self.output_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                    check=False
                )
                
                if run == 1:
                    print("[LaTeX] First pass complete")
                else:
                    print("[LaTeX] Second pass complete")
            
            # Check if PDF was created
            if self.pdf_path.exists():
                print(f"[LaTeX] ✅ PDF generated: {self.pdf_path}")
                return (str(self.tex_path), str(self.pdf_path))
            else:
                print("[LaTeX] ⚠️ PDF not created. Check LaTeX logs.")
                
                # Try to show error from log
                log_file = self.output_dir / "report.log"
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        log_lines = f.readlines()
                        # Find error lines
                        for i, line in enumerate(log_lines):
                            if "! " in line or "Error" in line:
                                print(f"[LaTeX Error] {line.strip()}")
                                if i + 1 < len(log_lines):
                                    print(f"             {log_lines[i+1].strip()}")
                
                return (str(self.tex_path), None)
                
        except FileNotFoundError:
            print("[LaTeX] ⚠️ pdflatex not found. Install TeX Live or MikTeX.")
            print("[LaTeX] .tex file created, but PDF compilation skipped.")
            return (str(self.tex_path), None)
            
        except subprocess.TimeoutExpired:
            print("[LaTeX] ⚠️ Compilation timeout. Document may be too large.")
            return (str(self.tex_path), None)
            
        except Exception as e:
            print(f"[LaTeX] ⚠️ Compilation error: {e}")
            return (str(self.tex_path), None)


# CLI test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python latex_report_generator.py <audit_json> [irc_json]")
        sys.exit(1)
    
    audit_path = sys.argv[1]
    irc_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    gen = LatexReportGenerator(audit_path, irc_path)
    tex, pdf = gen.generate()
    
    print(f"\nGenerated files:")
    print(f"  LaTeX: {tex}")
    print(f"  PDF:   {pdf or 'Not generated'}")







# cli_tool.py — FINAL VERSION

Run from terminal:
    python cli_tool.py --base base.mp4 --present now.mp4 --lat 28.6 --lon 77.2
"""

import argparse
from pathlib import Path
import json

from penultimate_road_audit_system import EnhancedRoadAuditSystem
from irc_solution_generator import IRCSolutionGenerator
from latex_report_generator import LatexReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Road Safety Audit CLI Tool")
    parser.add_argument("--base", required=True, help="Base video path")
    parser.add_argument("--present", required=True, help="Present video path")
    parser.add_argument("--lat", type=float, help="Manual latitude")
    parser.add_argument("--lon", type=float, help="Manual longitude")

    args = parser.parse_args()

    manual_gps = None
    if args.lat and args.lon:
        manual_gps = {"latitude": float(args.lat), "longitude": float(args.lon)}

    system = EnhancedRoadAuditSystem({
        "finetuned_model": "models/best.pt",
        "pretrained_model": "models/yolov8s.pt",
        "segmentation_model": "models/road_markings_yolov8s-seg.pt",
        "fps": 5,
    })

    print("\n=== Running Road Safety Audit ===\n")
    audit = system.run_complete_audit(args.base, args.present, manual_gps=manual_gps)

    # Save audit_output.json
    audit_json_path = Path("results/audit_output.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"✓ Audit JSON saved: {audit_json_path}")

    # IRC solution
    irc = IRCSolutionGenerator(str(audit_json_path)).generate()
    print("✓ IRC recommendations generated.")

    # PDF report
    tex, pdf = LatexReportGenerator(str(audit_json_path), "results/irc_output.json").generate()
    print(f"✓ LaTeX file: {tex}")
    if pdf:
        print(f"✓ PDF file: {pdf}")
    else:
        print("⚠ PDF generation skipped or LaTeX unavailable.")

    print("\n=== Completed Successfully ===")


if __name__ == "__main__":
    main()