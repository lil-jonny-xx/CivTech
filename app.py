"""
Complete Streamlit Frontend for Enhanced Road Safety Audit System V9

UI REFRESH (V9.8):
- Replaced horizontal tabs with a vertical navigation rail in the sidebar.
- Created a new "Configuration" page in the main content area.
- Moved config sliders (FPS, Threshold) to the new "Configuration" page.
- Implemented a high-contrast "Glassmorphism" dark mode UI.
- Fixed the invisible sidebar collapse arrow.
- Set sidebar to be collapsed by default.

FIX (V9.8.3):
- Replaced complex SVG page_icon with a simple emoji to fix
  the 'keyboard_double_arrow_right' text bug.
- Repaired crash on Configuration page related to st.selectbox index.
- Repaired pyarrow crash on Summary page by casting all table data to strings.
- Replaced all deprecated `use_container_width` flags with `width`.

Save as: app.py
Run: streamlit run app.py --server.maxUploadSize 1024
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import cv2
from PIL import Image
import tempfile
import shutil
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import System Modules ---
@st.cache_resource
def load_audit_system():
    try:
        from penultimate_road_audit_system import EnhancedRoadAuditSystem
        return EnhancedRoadAuditSystem
    except ImportError:
        st.error("FATAL: 'penultimate_road_audit_system.py' not found.")
        return None

@st.cache_resource
def load_irc_engine():
    try:
        from irc_solution_engine import IRCSolutionEngine
        return IRCSolutionEngine
    except ImportError:
        st.error("FATAL: 'irc_solution_engine.py' not found.")
        return None

@st.cache_resource
def load_latex_generator():
    try:
        from latex_report_generator import LatexReportGenerator
        return LatexReportGenerator
    except ImportError:
        st.error("FATAL: 'latex_report_generator.py' not found.")
        return None

EnhancedRoadAuditSystem = load_audit_system()
IRCSolutionEngine = load_irc_engine()
LatexReportGenerator = load_latex_generator()

AUDIT_AVAILABLE = EnhancedRoadAuditSystem is not None
IRC_AVAILABLE = IRCSolutionEngine is not None
LATEX_REPORT_AVAILABLE = LatexReportGenerator is not None
# --- End System Modules ---


# Page config
st.set_page_config(
    page_title="Road Safety Audit",
    # --- THIS IS THE FIX ---
    # Replaced the complex SVG data string with a simple emoji
    page_icon="🛣️",
    # --- END FIX ---
    layout="wide",
    initial_sidebar_state="collapsed" # Per user request
)

# --- Custom CSS (Glassmorphism Dark UI) ---
st.markdown("""
<style>
    /* Base */
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background-image: linear-gradient(180deg, #0A101F, #0D142B);
        background-attachment: fixed;
        color: #E0E0E0;
    }

    /* Main Header */
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        text-align: left;
        padding-bottom: 1rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #3A7BFD, #C642FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #9A9DA1;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar */
    .st-emotion-cache-16txtl3 {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    .st-emotion-cache-16txtl3 [data-testid="stSidebarHeader"] {
        color: #FFFFFF;
        font-size: 1.5rem;
    }
    .st-emotion-cache-ue6h4q {
        color: #E0E0E0;
    }
    
    /* Sidebar Collapse Arrow */
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #E0E0E0;
    }

    /* Vertical Nav (st.radio in sidebar) */
    .stRadio [role="radiogroup"] {
        background: transparent;
        padding: 0;
    }
    .stRadio [role="radio"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
        color: #9A9DA1;
    }
    /* Selected Nav Item */
    .stRadio [role="radio"][aria-checked="true"] {
        background-color: rgba(59, 130, 246, 0.2);
        color: #FFFFFF;
        font-weight: 600;
        border-right: 3px solid #3B82F6;
    }
    /* Nav Item Hover */
    .stRadio [role="radio"]:hover {
        background-color: #1E293B;
    }

    /* "Glassmorphism" Card Container */
    .glass-container {
        background: rgba(30, 41, 59, 0.5); /* Semi-transparent */
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Upload Box Style */
    .stFileUploader {
        border: 2px dashed #3B82F6;
        border-radius: 8px;
        padding: 1rem;
        background-color: rgba(15, 23, 42, 0.5);
    }
    
    /* Primary Button */
    .stButton [data-testid="stButton"] button {
        background: linear-gradient(90deg, #3A7BFD, #C642FE);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(58, 123, 253, 0.3);
    }
    .stButton [data-testid="stButton"] button:hover {
        box-shadow: 0 6px 20px rgba(58, 123, 253, 0.5);
        transform: translateY(-2px);
    }
    
    /* Secondary Download Buttons */
    .stDownloadButton [data-testid="stButton"] button {
        background-color: #1E293B;
        color: #E0E0E0;
        border: 1px solid #3B82F6;
    }
    .stDownloadButton [data-testid="stButton"] button:hover {
        background-color: #3B82F6;
        color: white;
    }
    
    /* Metric Label */
    .st-emotion-cache-1g8m9pl p {
        font-size: 1rem;
        color: #9A9DA1;
    }
    /* Metric Value */
    .st-emotion-cache-1g8m9pl div {
        font-size: 2.25rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: transparent;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Expander */
    .st-emotion-cache-p5msec {
        background-color: #1E293B;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)
# --- End Custom CSS ---


# Initialize session state
if 'audit_complete' not in st.session_state:
    st.session_state.audit_complete = False
    st.session_state.report = None
    st.session_state.irc_solutions = None
    st.session_state.base_video_path = None
    st.session_state.present_video_path = None
    st.session_state.tex_report_path = None
    st.session_state.temp_dir = tempfile.mkdtemp()

# --- Configuration Store ---
# Store config in session state to be accessible from all pages
if 'config' not in st.session_state:
    st.session_state.config = {
        'proc_height': 736,
        'min_confidence': 0.25,
        'fps': 5,
        'nms_iou_threshold': 0.5,
        'gps_distance_threshold_km': 5.0,
        'batch_size': 1,
        'pretrained_model': 'models/yolov8s.pt',
        'finetuned_model': 'models/pothole_detector_v1.pt'
    }

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Navigation")
    
    page_options = [
        "Upload", 
        "Configuration", 
        "Summary", 
        "Details", 
        "Gallery", 
        "Reports"
    ]
    
    # Use st.radio to create the vertical nav rail
    page = st.radio("Go to:", page_options, label_visibility="collapsed")

    st.markdown("---")
    st.subheader("System Information")
    st.info("""
    **Version:** V9.8.2 (Bug Fix)
    **Detection:** ML + CV Analysis
    **Standards:** IRC 103, 67, 35, 87, 93, SP:84
    """)
# --- End Sidebar ---


# Main header
st.markdown('<div class="main-header">Road Safety Audit System</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated temporal analysis of road infrastructure degradation.</p>', unsafe_allow_html=True)


# ==========================================
# PAGE 1: Upload Videos
# ==========================================
if page == "Upload":
    st.header("Upload Videos for Analysis")
    st.write("Provide the historical (Base) and current (Present) videos for comparison.")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        with st.container():
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.subheader("Base Video (Historical)")
            base_video = st.file_uploader(
                "Upload base video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key='base_upload',
                label_visibility="collapsed"
            )
            
            if base_video:
                st.success(f"{base_video.name} ({base_video.size / (1024*1024):.1f} MB)")
                base_vid_path = os.path.join(st.session_state.temp_dir, "base_video.mp4")
                with open(base_vid_path, "wb") as f:
                    f.write(base_video.read())
                st.session_state.base_video_path = base_vid_path
                st.video(base_video)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.subheader("Present Video (Current)")
            present_video = st.file_uploader(
                "Upload present video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key='present_upload',
                label_visibility="collapsed"
            )
            
            if present_video:
                st.success(f"{present_video.name} ({present_video.size / (1024*1024):.1f} MB)")
                present_vid_path = os.path.join(st.session_state.temp_dir, "present_video.mp4")
                with open(present_vid_path, "wb") as f:
                    f.write(present_video.read())
                st.session_state.present_video_path = present_vid_path
                st.video(present_video)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if base_video and present_video:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            if st.button("Begin Analysis", type="primary", width='stretch'):
                if not AUDIT_AVAILABLE or not IRC_AVAILABLE or not LATEX_REPORT_AVAILABLE:
                    st.error("Error: Core system files are missing. Please check file imports.")
                else:
                    progress_bar = st.progress(0, text="Starting Analysis...")
                    
                    try:
                        with st.spinner("Running analysis... This may take several minutes."):
                            status = st.empty()
                            
                            status.text("Initializing system...")
                            progress_bar.progress(10, text="Initializing system...")
                            # Use config from session state
                            system = EnhancedRoadAuditSystem(st.session_state.config)
                            
                            status.text("Processing videos...")
                            progress_bar.progress(30, text="Processing videos...")
                            
                            report = system.run_complete_audit(
                                st.session_state.base_video_path,
                                st.session_state.present_video_path
                            )
                            
                            progress_bar.progress(70, text="Generating IRC solutions...")
                            
                            if IRC_AVAILABLE and report:
                                status.text("Generating IRC solutions...")
                                irc_engine = IRCSolutionEngine()
                                st.session_state.irc_solutions = irc_engine.generate_solutions(report)
                            
                            if LATEX_REPORT_AVAILABLE and report and st.session_state.irc_solutions:
                                status.text("Generating LaTeX report...")
                                progress_bar.progress(90, text="Generating report...")
                                report_generator = LatexReportGenerator(report, st.session_state.irc_solutions)
                                tex_report_path = f"{st.session_state.temp_dir}/final_audit_report.tex"
                                report_generator.generate_tex(tex_report_path)
                                st.session_state.tex_report_path = tex_report_path

                            progress_bar.progress(100, text="Analysis Complete!")
                            
                            st.session_state.report = report
                            st.session_state.audit_complete = True
                            st.success("Analysis complete! View results in the other tabs.")
                    
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")
                        import traceback
                        st.exception(traceback.format_exc())
                    
                    finally:
                        if st.session_state.base_video_path and os.path.exists(st.session_state.base_video_path):
                            os.remove(st.session_state.base_video_path)
                            st.session_state.base_video_path = None
                        if st.session_state.present_video_path and os.path.exists(st.session_state.present_video_path):
                            os.remove(st.session_state.present_video_path)
                            st.session_state.present_video_path = None
    else:
        st.info("Please upload both videos to begin analysis.")

# ==========================================
# PAGE 2: Configuration
# ==========================================
elif page == "Configuration":
    st.header("System Configuration")
    st.write("Adjust the core parameters for the analysis pipeline.")

    with st.container():
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Analysis Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            # --- BUG FIX (V9.8.1) ---
            # Find the index of the current value in the options list
            options = [512, 640, 736, 1024]
            current_value = st.session_state.config.get('proc_height', 736)
            try:
                current_index = options.index(current_value)
            except ValueError:
                current_index = 2 # Default to 736
            
            st.session_state.config['proc_height'] = st.selectbox(
                "Processing Resolution (Height)", 
                options, 
                index=current_index
            )
            # --- END BUG FIX ---

        with c2:
            st.session_state.config['fps'] = st.slider(
                "Frames Per Second (FPS) to Analyze", 1, 10, st.session_state.config.get('fps', 5), 1
            )
        
        st.session_state.config['min_confidence'] = st.slider(
            "Minimum Confidence Threshold", 0.1, 0.9, st.session_state.config.get('min_confidence', 0.25), 0.05
        )
        
        st.success("Configuration saved. These settings will be used when you 'Begin Analysis'.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: Analysis Summary
# ==========================================
elif page == "Summary":
    st.header("Analysis Summary")
    
    if not st.session_state.audit_complete:
        st.warning("No results. Please upload and analyze videos on the 'Upload' tab.")
    else:
        report = st.session_state.report
        
        st.subheader("High-Level Summary")
        with st.container():
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frames Analyzed", report['frames_analyzed']['present'])
            with col2:
                st.metric("Frames with Changes", report['total_frames_with_changes'])
            with col3:
                st.metric("Total Issues Found", report['issue_summary']['total_issues'])
            with col4:
                severity = report['issue_summary']['by_severity']
                st.metric("Critical / High Priority", severity['high'], delta_color="inverse")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Infrastructure Comparison")
        comp = report['aggregate_comparison']
        
        comparison_data = []
        for key, values in comp.items():
            element = key.replace('_', ' ').title()
            
            # --- BUG FIX (V9.8.2): Cast all to string for Arrow compatibility ---
            base_val = f"{values['base']:.1f}" if isinstance(values['base'], float) else str(values['base'])
            present_val = f"{values['present']:.1f}" if isinstance(values['present'], float) else str(values['present'])
            change_val = f"{values['change']:+.1f}" if isinstance(values['change'], float) else f"{values['change']:+d}"
            
            comparison_data.append({
                'Infrastructure Element': element,
                'Base': base_val,
                'Present': present_val,
                'Change': change_val
            })
            # --- END BUG FIX ---
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, width='stretch', hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            defect_data = pd.DataFrame({
                'Category': ['Potholes', 'Potholes', 'Cracks', 'Cracks'],
                'Count': [
                    comp['potholes']['base'], comp['potholes']['present'],
                    comp['cracks']['base'], comp['cracks']['present']
                ],
                'Status': ['Base', 'Present', 'Base', 'Present']
            })
            
            fig = px.bar(defect_data, x='Category', y='Count', color='Status',
                         title='Defects Comparison (Potholes & Cracks)', barmode='group',
                         template='plotly_dark')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            severity_data = report['issue_summary']['by_severity']
            fig = px.pie(values=list(severity_data.values()),
                         names=list(severity_data.keys()),
                         title='Issues by Severity',
                         color_discrete_map={'high': '#d62728', 'medium': '#ff7f0e', 'low': '#ffbb78'},
                         template='plotly_dark')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')

# ==========================================
# PAGE 4: Detailed Comparison
# ==========================================
elif page == "Details":
    st.header("Frame-Level Analysis")
    
    if not st.session_state.audit_complete:
        st.warning("No results available.")
    else:
        report = st.session_state.report
        frame_changes = report.get('frame_level_changes', [])
        
        if not frame_changes:
            st.info("No significant frame-level changes detected.")
        else:
            st.success(f"Found changes in {len(frame_changes)} frames.")
            
            st.subheader("Change Timeline")
            timeline_data = []
            for change in frame_changes:
                for item in change['changes']:
                    timeline_data.append({
                        'Frame': change['frame_id'],
                        'Time (s)': change['timestamp_seconds'],
                        'Element': item['element'],
                        'Severity': item['severity']
                    })
            
            df_timeline = pd.DataFrame(timeline_data)
            fig = px.scatter(df_timeline, x='Time (s)', y='Element',
                           color='Severity', size=[10]*len(df_timeline),
                           title='Issues Over Time',
                           hover_data=['Frame'],
                           color_discrete_map={'high': '#d62728', 'medium': '#ff7f0e', 'low': '#2ca02c'},
                           template='plotly_dark')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
            
            st.markdown("---")
            
            st.subheader("Frame Details")
            severity_filter = st.multiselect(
                "Filter by severity", ['high', 'medium', 'low'], default=['high', 'medium', 'low']
            )
            
            for change in frame_changes[:20]: # Show top 20
                frame_id = change['frame_id']
                timestamp = change['timestamp_seconds']
                
                filtered_changes = [c for c in change['changes'] if c['severity'] in severity_filter]
                
                if filtered_changes:
                    with st.expander(f"Frame {frame_id} (at {timestamp:.1f}s) - {len(filtered_changes)} issues"):
                        for item in filtered_changes:
                            sev_color = "#d62728" if item['severity'] == 'high' else "#ff7f0e" if item['severity'] == 'medium' else "#9A9DA1"
                            st.markdown(f" - **<span style='color:{sev_color};'>{item['element']}</span>**: {item['type']}", unsafe_allow_html=True)
                            st.markdown(f"   (From: `{item.get('from', 'N/A')}` → To: `{item.get('to', 'N/A')}`)")

# ==========================================
# PAGE 5: Visual Gallery
# ==========================================
elif page == "Gallery":
    st.header("Visual Comparisons")
    
    if not st.session_state.audit_complete:
        st.warning("No results available.")
    else:
        comparisons_dir = Path('results/comparisons')
        
        if comparisons_dir.exists():
            comparison_images = sorted(list(comparisons_dir.glob('*.jpg')))
            
            if comparison_images:
                st.success(f"Found {len(comparison_images)} comparison images (showing top 10).")
                
                cols_per_row = 2
                for i in range(0, len(comparison_images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(comparison_images):
                            with cols[j]:
                                img = Image.open(comparison_images[idx])
                                st.image(img, caption=comparison_images[idx].name, width='stretch')
            else:
                st.info("No comparison images were generated (no high-priority changes found).")
        else:
            st.info("Comparison images will appear here after analysis.")

# ==========================================
# PAGE 6: IRC Solutions & Reports
# ==========================================
elif page == "Reports":
    st.header("IRC Solutions & Download Reports")
    
    if not st.session_state.audit_complete:
        st.warning("No results available.")
    else:
        report = st.session_state.report
        irc_solutions = st.session_state.irc_solutions
        
        if irc_solutions:
            st.subheader("IRC-Compliant Solutions")
            
            total_solutions = len(irc_solutions)
            critical_actions = len([s for s in irc_solutions if "Critical" in s['priority']])
            urgent_actions = len([s for s in irc_solutions if "Urgent" in s['priority']])
            
            with st.container():
                st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Solutions Generated", total_solutions)
                with col2:
                    st.metric("Critical (P1) Actions", critical_actions)
                with col3:
                    st.metric("Urgent (P2) Actions", urgent_actions)
                st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Priority Action Matrix")
            priority_df = pd.DataFrame(irc_solutions)
            st.dataframe(priority_df, width='stretch', hide_index=True)
        
        else:
            st.info("IRC solutions could not be generated.")
        
        st.markdown("---")
        st.subheader("Download Reports")
        
        with st.container():
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                json_str = json.dumps(report, indent=2)
                st.download_button(
                    "Download Audit Report (JSON)",
                    data=json_str,
                    file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width='stretch',
                    type="primary"
                )
                
                if irc_solutions:
                    json_str = json.dumps(irc_solutions, indent=2)
                    st.download_button(
                        "Download IRC Solutions (JSON)",
                        data=json_str,
                        file_name=f"irc_solutions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        width='stretch'
                    )
            
            with col2:
                if st.session_state.tex_report_path and Path(st.session_state.tex_report_path).exists():
                    with open(st.session_state.tex_report_path, "r", encoding="utf-8") as f:
                        tex_data = f.read()
                    st.download_button(
                        "Download Full Report (LaTeX .tex)",
                        data=tex_data,
                        file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                        mime="application/x-tex",
                        width='stretch'
                    )
                else:
                    st.info("Run analysis to generate LaTeX report.")
            st.markdown('</div>', unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Road Safety Audit System V9.8.3 | Automated Infrastructure Analysis</p>
</div>
""", unsafe_allow_html=True)