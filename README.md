Project Overview

The Automated Road Safety Audit System (ARSAS) is a state-of-the-art, fully automated road-audit platform that performs comparative assessment of road corridor conditions using two videos:
Base Video → Older corridor condition
Present Video → Latest corridor condition
The system performs:
YOLO-based defect detection
Road marking segmentation
GIS-driven contextual understanding
Crack width measurement
Lane geometry-based scaling
Frame-by-frame deterioration analysis
PCI scoring (Pavement Condition Index)
Root-cause analysis
IRC maintenance guideline generation
Auto-generated LaTeX → PDF report
Streamlit dashboard for visualization
This project was built by:

Jonathan V Paul
Ronak Barwar


Core Features
1. Dual-Video Comparator Engine (Penultimate Engine)
This is the heart of the system, designed to:
Synchronize Base vs Present videos using ORB-based visual alignment
Extract frames from both using FFmpeg / OpenCV fallback
Run YOLO object detection for road defects
Run segmentation for lane markings (optional)
Measure crack widths
Compute marking deterioration
Track potholes and surface failures
Generate PCI scores
Compare frame-by-frame:
New potholes
Widened cracks
Faded markings
Missing traffic infrastructure
Save comparison images
Output structured JSON for downstream analysis

2. GIS Context Engine
Determines road environment context:
Urban / Residential / Highway / Intersection
Traffic volume (ADT)
Heavy vehicle share
Rainfall patterns
Drainage quality
Soil type
Accident hotspot likelihood
This contextual data feeds into root-cause and IRC recommendations.

3. Root Cause Analyzer (RCA)
Uses defect characteristics + GIS profile to determine:
Fatigue cracking
Thermal cracking
Water stagnation–induced potholes
Abrasion-induced marking wear
Heavy-traffic degradation
Safety-furniture impact damage
Each defect receives a text explanation of underlying causes.

4. IRC Maintenance Recommendation Engine
Implements Indian Road Congress standards:
Potholes → IRC:82
Cracks → IRC:82 + MORTH 300
Faded markings → IRC:35
Signs → IRC:67
Guardrails / barriers → MORTH 800
Traffic signals → IRC:SP:58
Produces actionable:
Repair methods
Materials required
Reference sections
Notes for engineers
Outputs saved as irc_output.json.

5. Auto-LaTeX Reporting
Generates:
Title page
Audit metadata
GIS summary
PCI table
Aggregate defect comparison
Frame-level change logs
Embedded comparison images
Attached IRC recommendations
Produces:
results/report.tex
results/report.pdf

6. Streamlit Web Application
Includes:
Video upload
GPU/CPU auto-detection
Status logs
Progress bar
PCI KPI section
Defect comparison charts
GIS map (Folium)
Frame-level deterioration logs
Side-by-side comparison frames
JSON and PDF download buttons

🔧 Tech Stack

Deep Learning:
YOLO (Ultralytics)
OpenCV
PyTorch

GIS + Geo-Analysis:
Geopy
R-tree
GDAL / Fiona / Rasterio / Pyogrio

Backend:
Python
FFmpeg

Interface:
Streamlit
Plotly
Folium

Reporting:
LaTeX (MiKTeX / TeXLive)




Directory Structure
project/
│
├── models/
│   ├── best.pt
│   ├── yolov8s.pt
│   └── road_markings_yolov8s-seg.pt
│
├── penultimate_road_audit_system.py
├── irc_solution_generator.py
├── latex_report_generator.py
├── streamlit_app.py
│
├── uploads/
├── results/
│   ├── audit_output.json
│   ├── irc_output.json
│   ├── report.tex
│   ├── report.pdf
│   └── comparisons/
│
└── requirements.txt





How to run the system
1. Install dependencies
pip install -r requirements.txt

2. Run Streamlit app
streamlit run streamlit_app.py

3. Upload Base + Present videos
Accepted formats: .mp4, .avi, .mov, .mkv

4. Click “Run Complete Audit”
The system will generate:
audit_output.json
irc_output.json
comparison images
report.pdf



Outputs You Get
✓ Pavement Deterioration Insights
Pothole growth
Crack width change
Increased signage damage
Marking fade progression

✓ GIS-adjusted Severity
Traffic-adjusted defect severity
Rainfall-adjusted damage causes
Soil-based cracking tendencies

✓ Engineering-Grade PDF Report
Suitable for PWD, NHAI, Smart City projects
Contains side-by-side images
IRC-compliant action suggestions


Maintenance & Extensibility
You can extend:
New YOLO classes
More GIS layers (DEM, rainfall API, land use shapefiles)
Deeper segmentation models
LLM-based natural-language explanation generator
Multi-video long-sequence stitching


Acknowledgements
Ultralytics YOLO
MiKTeX / TeXLive
OpenCV community
GDAL, Fiona, Rasterio maintainers
<<<<<<< HEAD
IRC Documentation
=======
IRC Documentation
>>>>>>> 69cf25b08c997a45634a87a7a21d21342d4f6de1
