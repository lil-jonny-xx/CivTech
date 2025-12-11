# streamlit_app.py - FIXED VERSION
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
    status.info("🧹 Cleaning previous run data...")
    
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
        status.warning(f"⚠️ Cleanup completed with warnings:\n" + "\n".join(errors))
    else:
        status.success("✅ Previous data cleared successfully")
    
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
st.title("🛣️ Road Safety Audit System – Comparator Engine")
st.markdown("Upload **Base** and **Present** corridor videos for automated road safety audit with GIS analysis.")

missing_models = check_models()
if missing_models:
    st.warning("⚠️ Missing models: " + ", ".join(missing_models))
else:
    st.success("✅ All required models detected")

# Sidebar
st.sidebar.header("⚙️ Configuration")
gps_mode = st.sidebar.radio("GPS Source", ("Video metadata", "Manual entry"))
manual_gps = None
if gps_mode == "Manual entry":
    lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.6f")
    manual_gps = {"latitude": float(lat), "longitude": float(lon)}

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ GIS Parameters")
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
st.sidebar.info(f"💻 Device: **{get_device().upper()}**")

# Video upload
col1, col2 = st.columns(2)
with col1:
    base_video = st.file_uploader("📹 Upload Base Video", type=["mp4", "mov", "avi", "mkv"])
with col2:
    present_video = st.file_uploader("📹 Upload Present Video", type=["mp4", "mov", "avi", "mkv"])

run_btn = st.button("🚀 Run Complete Audit", type="primary", disabled=not (base_video and present_video))

# Run pipeline
if run_btn:
    # Clear all previous data first
    clear_all_previous_data()
    
    # VERIFY cleanup was successful
    audit_json_path = RESULTS_DIR / "audit_output.json"
    if audit_json_path.exists():
        st.error("❌ Failed to clear old audit_output.json. Please close any programs using this file and try again.")
        st.stop()
    
    st.session_state.engine_logs = []
    st.session_state.run_timestamp = datetime.now().isoformat()
    st.session_state.audit_completed = False  # Reset completion status
    st.session_state.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    write_engine_log(f"[INIT] New audit run started - ID: {st.session_state.current_run_id}")
    
    with st.spinner("📤 Saving uploaded videos..."):
        base_path = save_uploaded_video(base_video, "base_video.mp4")
        present_path = save_uploaded_video(present_video, "present_video.mp4")
    
    if not base_path or not present_path:
        st.error("❌ Error saving videos")
        st.stop()
    
    st.subheader("🔍 Step 1: Running Audit Engine")
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
            st.error("❌ Failed to save audit_output.json")
            st.stop()
        
        file_size = audit_json_path.stat().st_size
        write_engine_log(f"[ENGINE] Audit complete, JSON saved ({file_size} bytes)")
        st.success(f"✅ Audit engine completed (saved {file_size:,} bytes)")
        
    except Exception as e:
        import traceback
        err = f"Audit failed: {e}\n{traceback.format_exc()}"
        write_engine_log(err)
        st.error(f"❌ {err}")
        st.stop()
    
    progress.progress(70)
    
    # IRC Recommendations
    st.subheader("📋 Step 2: Generating IRC Recommendations")
    try:
        irc_engine = IRCSolutionGenerator(str(RESULTS_DIR / "audit_output.json"))
        irc_output = irc_engine.generate()
        write_engine_log("[IRC] Recommendations generated")
        st.success("✅ IRC recommendations generated")
    except Exception as e:
        write_engine_log(f"[IRC] Failed: {e}")
        st.warning(f"⚠️ IRC generation failed: {e}")
    
    progress.progress(85)
    
    # LaTeX Report
    st.subheader("📄 Step 3: Generating PDF Report")
    try:
        irc_path = RESULTS_DIR / "irc_output.json"
        report_gen = LatexReportGenerator(
            audit_json=str(RESULTS_DIR / "audit_output.json"),
            irc_json=str(irc_path) if irc_path.exists() else None
        )
        tex_path, pdf_path = report_gen.generate()
        
        if pdf_path and Path(pdf_path).exists():
            write_engine_log("[PDF] Generated successfully")
            st.success("✅ PDF report generated")
        else:
            write_engine_log("[PDF] Not generated (pdflatex missing?)")
            st.warning("⚠️ PDF not generated (install pdflatex)")
    except Exception as e:
        write_engine_log(f"[PDF] Failed: {e}")
        st.warning(f"⚠️ PDF generation failed: {e}")
    
    progress.progress(100)
    st.balloons()
    st.success("🎉 **Audit Complete!** Results loaded below.")
    st.session_state.audit_completed = True  # Mark audit as complete

# Display Results (always from disk)
st.markdown("---")
audit_json_path = RESULTS_DIR / "audit_output.json"

# Only load if audit has been completed in this session
if st.session_state.audit_completed:
    audit_report = safe_load_json(audit_json_path)
else:
    audit_report = None

tabs = st.tabs(["📊 Overview", "🗺️ GIS Analysis", "📸 Visuals", "📥 Downloads"])

# Overview Tab
with tabs[0]:
    st.header("Audit Overview")
    
    if not st.session_state.audit_completed:
        st.info("👆 Upload videos and click **'Run Complete Audit'** to begin analysis.")
        st.markdown("""
        ### What This System Does:
        - 🔍 Detects road defects (potholes, cracks, faded markings)
        - 📊 Calculates Pavement Condition Index (PCI)
        - 🗺️ Performs GIS analysis with traffic/rainfall heatmaps
        - 📋 Generates IRC maintenance recommendations
        - 📄 Creates comprehensive PDF reports
        """)
    elif not audit_report:
        st.error("⚠️ Audit completed but results file missing. Please re-run the audit.")
    else:
        # Verify this is the current run's data
        report_run_id = audit_report.get("run_id", "unknown")
        if st.session_state.current_run_id and report_run_id != st.session_state.current_run_id:
            st.error(f"⚠️ Data mismatch detected! Expected run ID: {st.session_state.current_run_id}, Got: {report_run_id}")
            st.warning("This indicates old data. Please re-run the audit.")
            st.stop()
        
        # Display run timestamp
        if st.session_state.run_timestamp:
            st.caption(f"🕐 Run timestamp: {st.session_state.run_timestamp}")
        
        # Debug: Show data source info
        if audit_json_path.exists():
            file_time = datetime.fromtimestamp(audit_json_path.stat().st_mtime).isoformat()
            file_size = audit_json_path.stat().st_size
            st.caption(f"📄 Data source: {audit_json_path.name} (modified: {file_time}, size: {file_size:,} bytes)")
        
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
        st.info("👆 Run an audit first to view GIS analysis and heatmaps.")
    elif not audit_report:
        st.error("⚠️ Audit data missing. Please re-run the audit.")
    else:
        gps = audit_report.get("gps", {})
        if not gps.get("latitude"):
            st.warning("⚠️ GPS coordinates not found")
            center = (28.6139, 77.2090)
        else:
            center = (gps["latitude"], gps["longitude"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Location & Intensity Heatmaps")
            st.markdown(f"**📍 Coordinates:** {center[0]:.6f}, {center[1]:.6f}")
            st.markdown(f"**🚗 Traffic:** {traffic_input} | **🌧️ Rainfall:** {rainfall_input}")
            
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
        st.info("👆 Run an audit first to view visual analysis and comparison images.")
    elif not audit_report:
        st.error("⚠️ Audit data missing. Please re-run the audit.")
    else:
        # Frame evolution chart
        st.subheader("📈 Defect Evolution (Present Video)")
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
        st.subheader("📦 Download Files")
        
        # Audit JSON
        if (RESULTS_DIR / "audit_output.json").exists():
            with open(RESULTS_DIR / "audit_output.json", "rb") as f:
                st.download_button(
                    "⬇️ audit_output.json",
                    f,
                    file_name="audit_output.json",
                    mime="application/json"
                )
        
        # IRC JSON
        if (RESULTS_DIR / "irc_output.json").exists():
            with open(RESULTS_DIR / "irc_output.json", "rb") as f:
                st.download_button(
                    "⬇️ irc_output.json",
                    f,
                    file_name="irc_output.json",
                    mime="application/json"
                )
        
        # PDF
        pdf_file = RESULTS_DIR / "report.pdf"
        if pdf_file.exists():
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "⬇️ Final PDF Report",
                    f,
                    file_name="Road_Audit_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("PDF not available (requires pdflatex)")
    
    with col2:
        st.subheader("📋 Engine Logs")
        
        log_file = RESULTS_DIR / "engine_trace.log"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                logs = f.read()
                st.text_area("Recent logs", logs[-2000:], height=300)
        else:
            st.info("No logs available")

st.markdown("---")
st.caption("© 2025 Road Safety Audit System | Powered by YOLOv8 + IRC Guidelines")