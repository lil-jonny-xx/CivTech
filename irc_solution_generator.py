"""
irc_solution_generator.py — FINAL (expanded, deterministic, bug-fixed)

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
