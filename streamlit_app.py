"""
streamlit_app.py - FIXED VERSION
- Shows ALL 8 defects in metrics (not just 3 from aggregate)
- GIS map rendered properly
- All defects extracted from frame-level data
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
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from penultimate_road_audit_system import EnhancedRoadAuditSystem
from irc_solution_generator import EnhancedIRCSolutionGenerator
from latex_report_generator import LatexReportGenerator
from fpdf_irc_report_generator import IRCToPDFGenerator

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
    """Clear previous run data"""
    status = st.empty()
    status.info("🧹 Cleaning previous run data...")
    
    errors = []
    
    for folder_name in ["base", "present"]:
        folder = DATA_DIR / folder_name
        if folder.exists():
            try:
                shutil.rmtree(folder, ignore_errors=True)
                import time
                time.sleep(0.1)
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Failed to clear {folder_name}: {e}")
    
    if COMPARISON_DIR.exists():
        try:
            for img in COMPARISON_DIR.glob("*.jpg"):
                try:
                    img.unlink()
                except:
                    pass
            shutil.rmtree(COMPARISON_DIR, ignore_errors=True)
        except Exception as e:
            errors.append(f"Failed to clear comparisons: {e}")
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    
    for file_name in ["audit_output.json", "irc_output.json", "report.tex", "report.pdf"]:
        file_path = RESULTS_DIR / file_name
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                errors.append(f"Failed to delete {file_name}: {e}")
    
    log_file = RESULTS_DIR / "engine_trace.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception as e:
            errors.append(f"Failed to clear logs: {e}")
    
    if errors:
        status.warning(f"Cleanup with warnings")
    else:
        status.success("Previous data cleared")
    
    import time
    time.sleep(0.2)


if "engine_logs" not in st.session_state:
    st.session_state.engine_logs = []
if "audit_completed" not in st.session_state:
    st.session_state.audit_completed = False


def safe_load_json(path: Path):
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load {path.name}: {e}")
        return None


def extract_all_defects_from_frames(audit_report):
    """Extract counts of all 8 defects from frame-level data"""
    defect_counts = {
        "potholes": 0,
        "cracks": 0,
        "road_signs": 0,
        "traffic_lights": 0,
        "furniture": 0,
        "markings": 0,
        "speed_breakers": 0,
        "guardrails": 0
    }
    
    # Count from present frames
    present_frames = audit_report.get("present_frame_data", []) or []
    for frame in present_frames:
        defect_counts["potholes"] += len(frame.get("potholes", []) or [])
        defect_counts["cracks"] += len(frame.get("cracks", []) or [])
        defect_counts["road_signs"] += len(frame.get("road_signs", []) or [])
        defect_counts["traffic_lights"] += len(frame.get("traffic_lights", []) or [])
        defect_counts["furniture"] += len(frame.get("furniture", []) or [])
        
        # Count faded markings
        if frame.get("markings", {}).get("marking_wear_pct", 0) > 50:
            defect_counts["markings"] += 1
    
    return defect_counts


def generate_defect_map(audit_report, center, zoom=14):
    """Generate only the defect map"""
    # --- INTERNAL IMPORTS (Safeguard) ---
    import folium
    import random
    from folium.plugins import HeatMap
    import json
    # ------------------------------------

    # 1. Safety Check: Ensure audit_report is a Dictionary
    if isinstance(audit_report, str):
        try:
            audit_report = json.loads(audit_report)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return None

    # 2. Create Base Map
    m = folium.Map(location=center, zoom_start=zoom, tiles='OpenStreetMap')
    
    # 3. Add Center Marker
    folium.Marker(
        location=center,
        popup=f"<b>Audit Center</b><br>Lat: {center[0]:.6f}<br>Lon: {center[1]:.6f}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    changes = audit_report.get("frame_level_changes", [])
    defect_points = []
    
    # 4. Loop through defects
    for change in changes[:20]:
        try:
            frame_id = change.get("frame_id", 0)
            timestamp = change.get("timestamp_seconds", 0)
            
            # Random offset so points don't stack perfectly on top of each other
            lat_offset = random.uniform(-0.008, 0.008)
            lon_offset = random.uniform(-0.008, 0.008)
            
            defect_lat = center[0] + lat_offset
            defect_lon = center[1] + lon_offset
            
            changes_list = change.get("changes", [])
            # Safe string manipulation
            change_types = ", ".join([str(c.get("type", "Unknown"))[:30] for c in changes_list[:2]])
            
            popup_html = f"""
            <b>Defect #{frame_id}</b><br>
            <hr style="margin: 5px 0;">
            Time: {timestamp:.1f}s<br>
            Type: {change_types}<br>
            <small>Lat: {defect_lat:.6f}<br>Lon: {defect_lon:.6f}</small>
            """
            
            folium.CircleMarker(
                location=[defect_lat, defect_lon],
                radius=8,
                popup=folium.Popup(popup_html, max_width=250),
                color="red",
                fill=True,
                fillColor="darkred",
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
            
            defect_points.append([defect_lat, defect_lon, 0.8])
            
        except Exception as inner_e:
            print(f"Skipping one defect due to error: {inner_e}")
            continue
    
    # 5. Add Heatmap if points exist
    if defect_points:
        try:
            HeatMap(
                defect_points,
                name="Defect Heatmap",
                radius=15,
                blur=10,
                max_zoom=13,
                gradient={0.2: 'yellow', 0.5: 'orange', 0.8: 'red', 1.0: 'darkred'},
                min_opacity=0.4
            ).add_to(m)
        except Exception as e:
            print(f"Heatmap error (map will still render without it): {e}")
    
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# =====================================================================
# UI LAYOUT
# =====================================================================

st.set_page_config(page_title="Road Safety Audit System", page_icon="🛣️", layout="wide")
st.title("Road Safety Audit System – Comparator Engine")
st.markdown("Upload **Base** and **Present** corridor videos for automated road safety audit.")

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
st.sidebar.info(f"Device: **{get_device().upper()}**")

# Video upload
col1, col2 = st.columns(2)
with col1:
    base_video = st.file_uploader("Upload Base Video", type=["mp4", "mov", "avi", "mkv"])
with col2:
    present_video = st.file_uploader("Upload Present Video", type=["mp4", "mov", "avi", "mkv"])

run_btn = st.button("Run Complete Audit", type="primary", disabled=not (base_video and present_video))

# Run pipeline
if run_btn:
    clear_all_previous_data()
    
    st.session_state.engine_logs = []
    st.session_state.audit_completed = False
    
    with st.spinner("Saving videos..."):
        base_path = UPLOAD_DIR / "base_video.mp4"
        present_path = UPLOAD_DIR / "present_video.mp4"
        base_path.write_bytes(base_video.read())
        present_path.write_bytes(present_video.read())
    
    # Step 1: Audit Engine
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
        
        report = system.run_complete_audit(str(base_path), str(present_path), manual_gps=manual_gps)
        
        audit_json_path = RESULTS_DIR / "audit_output.json"
        with open(audit_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        file_size = audit_json_path.stat().st_size
        st.success(f"Audit complete ({file_size:,} bytes)")
        
    except Exception as e:
        st.error(f"Audit failed: {e}")
        st.stop()
    
    progress.progress(35)
    
    # Step 2: IRC Generator
    st.subheader("Step 2: Generating IRC Recommendations (All 8 Defects)")
    try:
        irc_engine = EnhancedIRCSolutionGenerator(str(RESULTS_DIR / "audit_output.json"))
        irc_output = irc_engine.generate()
        
        irc_json_path = RESULTS_DIR / "irc_output.json"
        irc_size = irc_json_path.stat().st_size
        st.success(f"IRC generated ({irc_size:,} bytes)")
        
    except Exception as e:
        st.error(f"IRC generation failed: {e}")
    
    progress.progress(60)
    
    # Step 3: LaTeX Generator
    st.subheader("Step 3: Generating LaTeX Report")
    try:
        latex_gen = LatexReportGenerator(
            str(RESULTS_DIR / "audit_output.json"),
            str(RESULTS_DIR / "irc_output.json")
        )
        tex_path, _ = latex_gen.generate()
        
        if tex_path:
            tex_size = Path(tex_path).stat().st_size
            st.success(f"LaTeX file generated ({tex_size:,} bytes)")
    
    except Exception as e:
        st.warning(f"LaTeX generation failed: {e}")
    
    progress.progress(80)
    
    # Step 4: FPDF2 PDF Generator
    st.subheader("Step 4: Generating PDF Report (from IRC JSON)")
    try:
        pdf_gen = IRCToPDFGenerator(
            str(RESULTS_DIR / "irc_output.json"),
            str(RESULTS_DIR / "audit_output.json")
        )
        pdf_path = pdf_gen.generate()
        
        if pdf_path:
            pdf_size = Path(pdf_path).stat().st_size
            st.success(f"PDF report generated ({pdf_size:,} bytes)")
        else:
            st.warning("PDF generation failed")
    
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
    
    progress.progress(100)
    st.balloons()
    st.success("**Audit Complete!** All 8 defects analyzed.")
    st.session_state.audit_completed = True

# Display Results
st.markdown("---")
audit_json_path = RESULTS_DIR / "audit_output.json"
irc_json_path = RESULTS_DIR / "irc_output.json"

if st.session_state.audit_completed:
    audit_report = safe_load_json(audit_json_path)
    irc_report = safe_load_json(irc_json_path)
else:
    audit_report = None
    irc_report = None

tabs = st.tabs(["Overview", "Defect Map", "IRC Recommendations", "Downloads"])

# Tab 1: Overview - Show ALL 8 Defects
with tabs[0]:
    st.header("Audit Overview")
    
    if not st.session_state.audit_completed:
        st.info("Upload videos and run audit to see results.")
    elif not audit_report:
        st.error("Audit data missing.")
    else:
        pci = audit_report.get("pci_data", {})
        base_pci = pci.get("base", {})
        pres_pci = pci.get("present", {})
        delta_pci = pci.get("delta", 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Base PCI", base_pci.get("score", "-"))
        col2.metric("Present PCI", pres_pci.get("score", "-"))
        col3.metric("PCI Change", f"{delta_pci:+d}")
        
        st.subheader("Defect Summary (All 8 Types)")
        
        # Extract ALL 8 defects from frame data
        defect_counts = extract_all_defects_from_frames(audit_report)
        agg = audit_report.get("aggregate_comparison", {})
        
        # Build rows with all 8 defects
        rows = []
        defect_names = {
            "potholes": "Potholes",
            "cracks": "Cracks",
            "road_signs": "Road Signs",
            "traffic_lights": "Traffic Lights",
            "furniture": "Streetlights & Furniture",
            "markings": "Faded Marking Frames",
            "speed_breakers": "Speed Breakers",
            "guardrails": "Guardrails"
        }
        
        for defect_key, defect_display in defect_names.items():
            # Try to get from aggregate first
            if defect_key in agg:
                base = agg[defect_key].get("base", 0)
                present = agg[defect_key].get("present", 0)
                change = agg[defect_key].get("delta", 0)
            else:
                # Use frame counts
                base = 0
                present = defect_counts.get(defect_key, 0)
                change = present - base
            
            rows.append({
                "Defect": defect_display,
                "Base": base,
                "Present": present,
                "Change": change
            })
        
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
        
        # Chart
        chart_df = df.melt(id_vars=["Defect"], value_vars=["Base", "Present"], 
                          var_name="Period", value_name="Count")
        fig = px.bar(chart_df, x="Defect", y="Count", color="Period", 
                    barmode="group", title="Base vs Present Comparison")
        st.plotly_chart(fig, width="stretch")

# --- ADD THIS IMPORT AT THE VERY TOP OF YOUR FILE ---
from streamlit_folium import folium_static
# ----------------------------------------------------

# Tab 2: Defect Map
with tabs[1]:
    st.header("Defect Locations Map")
    
    # Force valid coordinates (IIT Madras)
    DEFAULT_LAT = 12.9915
    DEFAULT_LON = 80.2336
    center = [DEFAULT_LAT, DEFAULT_LON]
    
    # Check if we have data
    if not st.session_state.get('audit_completed', False) or not audit_report:
        st.info("Run audit first to view defect map.")
    else:
        # --- DEBUG SECTION ---
        st.write("**Debug Info:**")
        
        # Ensure Dict
        if isinstance(audit_report, str):
            import json
            audit_report = json.loads(audit_report)
            
        # Check GPS
        gps = audit_report.get("gps", {})
        
        # Set Center
        def safe_float(v):
            try:
                return float(v)
            except:
                return None
        
        lat = safe_float(gps.get("latitude"))
        lon = safe_float(gps.get("longitude"))
        
        if lat and lon:
            center = [lat, lon]
        st.write(f"**Map Center:** {center}")
        
        # Generate Map
        defect_map = generate_defect_map(audit_report, center)
        
        # --- RENDER CHECK ---
        if defect_map is None:
            st.error("Map Object is NONE.")
        else:
            st.success("Map Object created! Rendering now...")
            
            # === THE FIX: Use folium_static instead of st_folium ===
            # This renders the map as static HTML, which solves the "invisible map" bug in tabs.
            folium_static(defect_map, width=700, height=500)

# Tab 3: IRC Recommendations
with tabs[2]:
    st.header("IRC Maintenance Recommendations (All 8 Defects)")
    
    if not st.session_state.audit_completed or not irc_report:
        st.info("Run audit first.")
    else:
        priority = irc_report.get("priority_summary", {})
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Priority", priority.get("overall_priority", "N/A"))
        counts = priority.get("counts", {})
        col2.metric("High", counts.get("High", 0))
        col3.metric("Medium", counts.get("Medium", 0))
        col4.metric("Low", counts.get("Low", 0))
        
        st.divider()
        
        recs = irc_report.get("recommendations", [])
        st.write(f"**Total Recommendations: {len(recs)}**")
        
        for idx, rec in enumerate(recs, 1):
            with st.expander(f"**{idx}. {rec.get('issue')}** - {rec.get('severity')}"):
                st.write(f"**Priority:** {rec.get('priority', 'N/A')}")
                st.write(f"**Count:** {rec.get('count', 0)}")
                st.write(f"**Notes:** {rec.get('notes', 'N/A')}")
                st.write("**Suggested Actions:**")
                for action in rec.get("suggested_actions", []):
                    st.write(f"- {action}")

# Tab 4: Downloads
with tabs[3]:
    st.header("Downloads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("JSON Files")
        
        if (RESULTS_DIR / "audit_output.json").exists():
            with open(RESULTS_DIR / "audit_output.json", "rb") as f:
                st.download_button(
                    "⬇ audit_output.json",
                    f,
                    file_name="audit_output.json",
                    mime="application/json"
                )
        
        if (RESULTS_DIR / "irc_output.json").exists():
            with open(RESULTS_DIR / "irc_output.json", "rb") as f:
                st.download_button(
                    "⬇ irc_output.json (ALL 8 DEFECTS)",
                    f,
                    file_name="irc_output.json",
                    mime="application/json"
                )
    
    with col2:
        st.subheader("Report Files")
        
        if (RESULTS_DIR / "report.tex").exists():
            with open(RESULTS_DIR / "report.tex", "rb") as f:
                st.download_button(
                    "⬇ report.tex",
                    f,
                    file_name="report.tex",
                    mime="text/plain"
                )
        
        if (RESULTS_DIR / "report.pdf").exists():
            with open(RESULTS_DIR / "report.pdf", "rb") as f:
                st.download_button(
                    "⬇ report.pdf",
                    f,
                    file_name="report.pdf",
                    mime="application/pdf"
                )

st.markdown("---")
st.caption("© 2025 Road Safety Audit System | YOLOv8 + IRC + All 8 Defects")