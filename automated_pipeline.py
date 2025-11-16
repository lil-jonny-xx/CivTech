"""
Automated Road Audit Pipeline

This script ties all components together:
1. Runs the audit system (v9.7)
2. Runs the IRC solution engine (v9.7)
3. Runs the LaTeX report generator (v9.7)

FIX (V9.7): Version bump.

This is an updated file. Save as: automated_pipeline.py
Run: python automated_pipeline.py
"""
import json
from pathlib import Path
from datetime import datetime
import traceback

# Import all system components
try:
    from penultimate_road_audit_system import EnhancedRoadAuditSystem
    from irc_solution_engine import IRCSolutionEngine
    from latex_report_generator import LatexReportGenerator
except ImportError as e:
    print(f"[ERROR] Missing system files. Ensure all .py files are in the same directory.")
    print(f"Details: {e}")
    exit(1)

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print(" AUTOMATED ROAD AUDIT PIPELINE (V9.7 - LaTeX)")
    print("="*70)
    
    # 1. Configuration
    config = {
        'pretrained_model': 'models/yolov8s.pt',
        'finetuned_model': 'models/pothole_detector_v1.pt',
        'proc_height': 736,
        'min_confidence': 0.25,
        'nms_iou_threshold': 0.5,
        'gps_distance_threshold_km': 5.0,
        'fps': 5,
        'batch_size': 1
    }
    
    # Find videos
    base_video = 'data/inputs/base_video.mp4'
    present_video = 'data/inputs/present_video.mp4'
    
    if not Path(base_video).exists() or not Path(present_video).exists():
        print(f"[ERROR] Input videos not found.")
        print(f"Expected: {base_video}")
        print(f"Expected: {present_video}")
        return
    
    # Define output paths
    report_json_path = 'results/audit_report_v9_enhanced.json'
    report_tex_path = 'final_audit_report.tex' # Save to root
    Path('results').mkdir(exist_ok=True)
    
    # --- STAGE 1: RUN AUDIT ---
    try:
        print("\n--- STAGE 1: RUNNING ROAD AUDIT ---")
        system = EnhancedRoadAuditSystem(config)
        report_data = system.run_complete_audit(base_video, present_video)
        
        if not report_data:
            print("[ERROR] Audit failed to complete. Exiting pipeline.")
            return
            
        print("[SUCCESS] Audit complete. JSON report saved.")
    
    except MemoryError:
        print("\n\n[ERROR] PIPELINE FAILED: OUT OF MEMORY!")
        print("Solutions:")
        print("   1. Reduce 'proc_height' (try 640 or 512)")
        print("   2. Reduce 'fps' (try 3 or 2)")
        return
    except Exception as e:
        print(f"\n\n[ERROR] PIPELINE FAILED at Stage 1 (Audit): {e}")
        traceback.print_exc()
        return

    # --- STAGE 2: RUN IRC SOLUTION ENGINE ---
    try:
        print("\n--- STAGE 2: GENERATING IRC SOLUTIONS ---")
        with open(report_json_path, 'r') as f:
            report_data_for_irc = json.load(f)
            
        engine = IRCSolutionEngine()
        irc_solutions = engine.generate_solutions(report_data_for_irc)
        print("[SUCCESS] IRC solutions generated.")
        
    except Exception as e:
        print(f"\n\n[ERROR] PIPELINE FAILED at Stage 2 (IRC Engine): {e}")
        traceback.print_exc()
        return

    # --- STAGE 3: GENERATE LATEX REPORT ---
    try:
        print("\n--- STAGE 3: GENERATING LATEX REPORT ---")
        generator = LatexReportGenerator(report_data, irc_solutions)
        generator.generate_tex(report_tex_path) # Path is now root
        print(f"[SUCCESS] Final .tex report generated: {report_tex_path}")
        print(f"   To create PDF, run: pdflatex {report_tex_path}")
        
    except Exception as e:
        print(f"\n\n[ERROR] PIPELINE FAILED at Stage 3 (LaTeX Report): {e}")
        traceback.print_exc()
        return

    # --- COMPLETE ---
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)
    print("\nFinal Outputs:")
    print(f" 1. JSON Report: {report_json_path}")
    print(f" 2. LaTeX Report: {report_tex_path}")
    print(f" 3. Visuals:      results/comparisons/")
    
if __name__ == "__main__":
    main()