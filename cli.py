"""
cli_tool.py — FINAL VERSION

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
