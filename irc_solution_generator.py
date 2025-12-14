"""
irc_solution_generator.py — COMPLETE ENHANCED VERSION (all 8 defects)

Generates IRC-based recommendations from audit_output.json.
NOW EXTRACTS ALL DEFECT TYPES:
  1. Potholes
  2. Cracks
  3. Road Markings (faded/missing)
  4. Road Signs (missing/damaged)
  5. Traffic Lights (missing/damaged)
  6. Streetlights & Furniture (missing/damaged)
  7. Speed Breakers
  8. Guardrails (occluded/damaged)

Key improvements:
- Counts defects from both aggregate_comparison AND frame-level detections
- Analyzes frame_level_changes for severity & change patterns
- IRC references for all 8 defect types
- Defensive JSON parsing with safe fallbacks
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

JSONDict = Dict[str, Any]


class EnhancedIRCSolutionGenerator:
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
        self.defect_counters = defaultdict(lambda: {"base": 0, "present": 0, "delta": 0})

    # ---------------------------
    # Utility helpers
    # ---------------------------
    def _safe_get(self, path: List[str], default=None):
        """Safe nested dict getter."""
        node = self.audit
        for p in path:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def _append_rec(self, issue: str, severity: str, count: int, actions: List[str], 
                    priority: str = "Normal", notes: str = ""):
        rec = {
            "issue": issue,
            "severity": severity,
            "count": count,
            "priority": priority,
            "suggested_actions": actions,
            "notes": notes,
        }
        self.recommendations.append(rec)

    def _count_defect_in_frames(self, defect_type: str) -> Dict[str, int]:
        """
        Count defects from frame-level data.
        defect_type: "potholes", "cracks", "road_signs", "traffic_lights", "furniture", etc.
        """
        base_count = 0
        present_count = 0

        # Count in base frames
        base_frames = self._safe_get(["base_frame_data"], []) or []
        for frame in base_frames:
            base_count += len(frame.get(defect_type, []) or [])

        # Count in present frames
        present_frames = self._safe_get(["present_frame_data"], []) or []
        for frame in present_frames:
            present_count += len(frame.get(defect_type, []) or [])

        return {
            "base": base_count,
            "present": present_count,
            "delta": present_count - base_count
        }

    def _extract_frame_changes_for_type(self, element_name: str) -> List[str]:
        """
        Extract specific change types from frame_level_changes.
        Returns list of change descriptions matching the element.
        """
        changes_list = self._safe_get(["frame_level_changes"], []) or []
        relevant_changes = []
        
        for frame_change in changes_list:
            for change in frame_change.get("changes", []):
                if element_name.lower() in change.get("element", "").lower():
                    relevant_changes.append(change.get("type", "Unknown change"))
        
        return relevant_changes

    # ---------------------------
    # Rule 1: Potholes
    # ---------------------------
    def rule_potholes(self):
        agg = self._safe_get(["aggregate_comparison", "potholes"], {}) or {}
        base = int(agg.get("base", 0) or 0)
        present = int(agg.get("present", 0) or 0)
        delta = int(agg.get("delta", present - base) or 0)
        count = present

        if count <= 0 and delta <= 0:
            return

        if delta <= 0:
            severity = "None"
            actions = ["No new potholes detected. Maintain periodic inspection schedule."]
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
                "Full-depth patching recommended (remove failed material, replace with hot mix). Follow IRC:SP:100.",
                "Inspect and repair subgrade/drainage near pothole locations before resurfacing to avoid recurrence.",
                "If multiple clusters present, consider sectional resurfacing or overlay."
            ]
            priority = "High"

        notes = f"Base={base}, Present={present}, Delta={delta}."
        self._append_rec("Potholes", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule 2: Cracks
    # ---------------------------
    def rule_cracks(self):
        agg = self._safe_get(["aggregate_comparison", "cracks"], {}) or {}
        base = int(agg.get("base", 0) or 0)
        present = int(agg.get("present", 0) or 0)
        delta = int(agg.get("delta", present - base) or 0)
        count = present

        if count <= 0 and delta <= 0:
            return

        # Extract max crack width from present frames
        max_crack_width_cm = 0.0
        present_frames = self._safe_get(["present_frame_data"], []) or []
        for frame in present_frames:
            for crack in frame.get("cracks", []) or []:
                w = float(crack.get("width_cm") or crack.get("width_px", 0) or 0)
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
                    "Widespread/wide cracks detected. Consider surface renewal or overlay.",
                    "If cracks indicate fatigue, plan for milling + resurfacing (see IRC:SP:76).",
                    "Conduct structural investigation if subgrade issues suspected."
                ]
                priority = "High"
            else:
                severity = "Moderate"
                actions = [
                    "Apply crack sealing using elastomeric sealants after cleaning (refer IRC:116).",
                    "Ensure cracks are cleaned and dried before sealant application."
                ]
                priority = "Medium"

        notes = f"Base={base}, Present={present}, Delta={delta}, Max width (cm)={max_crack_width_cm:.2f}"
        self._append_rec("Cracks", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule 3: Road Markings
    # ---------------------------
    def rule_faded_markings(self):
        agg = self._safe_get(["aggregate_comparison", "faded_marking_frames"], {}) or {}
        base = int(agg.get("base", 0) or 0)
        present = int(agg.get("present", 0) or 0)
        delta = int(agg.get("delta", present - base) or 0)
        count = present

        if count <= 0 and delta <= 0:
            return

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
                "Major loss of marking visibility—perform full re-marking with thermoplastic materials and glass beads (IRC:35).",
                "Consider high-durability thermoplastic or premix systems for longer service life.",
                "Improve nighttime retro-reflectivity testing schedule after re-marking."
            ]
            priority = "High"

        notes = f"Base={base}, Present={present}, Delta={delta}."
        self._append_rec("Road Markings", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule 4: Road Signs
    # ---------------------------
    def rule_road_signs(self):
        # Try aggregate first
        agg = self._safe_get(["aggregate_comparison", "road_signs"], {})
        
        if agg:
            base = int(agg.get("base", 0) or 0)
            present = int(agg.get("present", 0) or 0)
            delta = int(agg.get("delta", present - base) or 0)
        else:
            # Fall back to frame-level counting
            counts = self._count_defect_in_frames("road_signs")
            base = counts["base"]
            present = counts["present"]
            delta = counts["delta"]

        # Detect missing signs via frame changes
        changes = self._extract_frame_changes_for_type("Road Signs")
        missing_count = len([c for c in changes if "missing" in c.lower()])
        new_count = len([c for c in changes if "new" in c.lower()])

        # Use the higher count
        count = max(present, missing_count)

        if count <= 0 and delta <= 0 and missing_count == 0:
            return

        if missing_count > 0 or delta < 0:
            severity = "High"
            priority = "High"
            actions = [
                "Replace missing/damaged signs per IRC:67 standards (retro-reflective sheeting, correct sizing).",
                "Use Type-XI or appropriate reflective sheeting for high-speed corridors as per IRC recommendations.",
                "Inspect mounting posts and foundations; repair or replace to restore sight distance and stability."
            ]
            notes = f"Frame changes detected: {missing_count} missing, {new_count} new. Base={base}, Present={present}, Delta={delta}."
        else:
            severity = "None"
            priority = "Low"
            actions = ["Signs are intact. Maintain periodic inspection for damage."]
            notes = f"Base={base}, Present={present}, Delta={delta}."

        self._append_rec("Road Signs", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule 5: Traffic Lights
    # ---------------------------
    def rule_traffic_lights(self):
        # Try aggregate
        agg = self._safe_get(["aggregate_comparison", "traffic_lights"], {})
        
        if agg:
            base = int(agg.get("base", 0) or 0)
            present = int(agg.get("present", 0) or 0)
            delta = int(agg.get("delta", present - base) or 0)
        else:
            # Fall back to frame-level
            counts = self._count_defect_in_frames("traffic_lights")
            base = counts["base"]
            present = counts["present"]
            delta = counts["delta"]

        # Detect changes
        changes = self._extract_frame_changes_for_type("Traffic Light")
        missing_count = len([c for c in changes if "missing" in c.lower()])

        count = max(present, missing_count)

        if count <= 0 and delta <= 0 and missing_count == 0:
            return

        if missing_count > 0 or delta < 0:
            severity = "High"
            priority = "High"
            actions = [
                "Replace missing or non-functional traffic signals to restore traffic control.",
                "Verify signal timing and phasing per IRC:103 traffic control guidelines.",
                "Inspect foundations and wiring; replace if corroded or damaged.",
                "Restore visibility by trimming vegetation around signal heads."
            ]
            notes = f"Missing: {missing_count}. Base={base}, Present={present}, Delta={delta}."
        else:
            severity = "None"
            priority = "Low"
            actions = ["Traffic signals are functional. Continue maintenance schedule."]
            notes = f"Base={base}, Present={present}, Delta={delta}."

        self._append_rec("Traffic Lights", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule 6: Streetlights & Furniture
    # ---------------------------
    def rule_streetlights_and_furniture(self):
        # Try aggregate
        agg = self._safe_get(["aggregate_comparison", "furniture"], {})
        
        if agg:
            base = int(agg.get("base", 0) or 0)
            present = int(agg.get("present", 0) or 0)
            delta = int(agg.get("delta", present - base) or 0)
        else:
            counts = self._count_defect_in_frames("furniture")
            base = counts["base"]
            present = counts["present"]
            delta = counts["delta"]

        # Detect streetlight-specific changes
        changes = self._extract_frame_changes_for_type("Roadside Furniture")
        missing_count = len([c for c in changes if "missing" in c.lower() or "damaged" in c.lower()])

        count = max(present, missing_count)

        if count <= 0 and delta <= 0 and missing_count == 0:
            return

        if missing_count > 0 or delta < 0:
            severity = "High"
            priority = "High"
            actions = [
                "Replace missing or damaged streetlights to improve nighttime safety (IRC guidelines on street lighting).",
                "Inspect poles for corrosion, cracks, or tilt; repair or replace as necessary.",
                "Ensure adequate spacing and luminance levels per IRC:83 standards.",
                "Clean light fixtures and replace non-functional lamps regularly."
            ]
            notes = f"Missing/damaged count: {missing_count}. Base={base}, Present={present}, Delta={delta}."
        else:
            severity = "None"
            priority = "Low"
            actions = ["Streetlights and furniture are functional. Maintain routine maintenance schedule."]
            notes = f"Base={base}, Present={present}, Delta={delta}."

        self._append_rec("Streetlights & Roadside Furniture", severity, count, actions, priority, notes)

    # ---------------------------
    # Rule 7: Speed Breakers
    # ---------------------------
    def rule_speed_breakers(self):
        # Count from frame data
        counts = self._count_defect_in_frames("furniture")  # Speed breakers often in furniture
        base = counts["base"]
        present = counts["present"]
        delta = counts["delta"]

        # Try to get specific speed breaker count from aggregate if available
        agg = self._safe_get(["aggregate_comparison", "speed_breakers"], {})
        if agg:
            base = int(agg.get("base", 0) or 0)
            present = int(agg.get("present", 0) or 0)
            delta = int(agg.get("delta", present - base) or 0)

        if present <= 0 and delta == 0:
            return

        actions = [
            "Inspect speed breaker condition for cracks, displacement, or deterioration.",
            "Ensure height and geometry meet IRC specifications for the road classification.",
            "Maintain adequate approach marking and signage (advance warning signs at 40m for urban roads).",
            "Consider surface treatment (paint or reflectors) for nighttime visibility."
        ]
        severity = "Moderate" if present > 3 else "None"
        priority = "Medium" if present > 3 else "Low"
        notes = f"Base={base}, Present={present}, Delta={delta}."

        self._append_rec("Speed Breakers", severity, present, actions, priority, notes)

    # ---------------------------
    # Rule 8: Guardrails
    # ---------------------------
    def rule_guardrails(self):
        # Count from frame data
        counts = self._count_defect_in_frames("furniture")  # Guardrails often in furniture
        base = counts["base"]
        present = counts["present"]

        # Detect occlusion
        present_frames = self._safe_get(["present_frame_data"], []) or []
        guardrail_occluded = 0
        guardrail_damaged = 0

        for frame in present_frames:
            for det in frame.get("furniture", []) or []:
                label = str(det.get("label", "")).lower()
                if "guardrail" in label or "guard_rail" in label or "guard rail" in label:
                    if det.get("occluded", False):
                        guardrail_occluded += 1
                    if det.get("root_cause") and "damaged" in det.get("root_cause", "").lower():
                        guardrail_damaged += 1

        # Detect changes from frame changes
        changes = self._extract_frame_changes_for_type("Guardrail")
        missing_in_changes = len([c for c in changes if "missing" in c.lower()])

        total_issues = guardrail_occluded + guardrail_damaged + missing_in_changes

        if total_issues == 0:
            return

        severity = "High" if missing_in_changes > 0 or guardrail_damaged > 0 else "Medium"
        priority = "High" if severity == "High" else "Medium"

        actions = [
            "Vegetation trimming/clearance recommended to restore guardrail visibility and enable inspection.",
            "If guardrail is missing or structurally damaged, replace per IRC:96 guidelines.",
            "Realign guardrail as necessary; ensure proper alignment with roadway geometry.",
            "Inspect foundations and connections; reinforce or replace if corroded or loose."
        ]
        notes = f"Occluded: {guardrail_occluded}, Damaged: {guardrail_damaged}, Missing: {missing_in_changes}. Base={base}, Present={present}."

        self._append_rec("Guardrails", severity, total_issues, actions, priority, notes)

    # ---------------------------
    # Rule 9: Drainage & Surface
    # ---------------------------
    def rule_drainage_and_surface(self):
        gis = self._safe_get(["gis_profile"], {}) or {}
        drainage = gis.get("drainage_quality", "Unknown")
        pothole_delta = int(self._safe_get(["aggregate_comparison", "potholes", "delta"], 0) or 0)
        crack_delta = int(self._safe_get(["aggregate_comparison", "cracks", "delta"], 0) or 0)

        if drainage not in ["Poor", "Blocked"] and pothole_delta <= 3 and crack_delta <= 5:
            return

        actions = [
            f"Investigate roadside drainage condition (current rating: {drainage}). Clear blockages, regrade channels.",
            "If water ingress is evident, plan sub-surface drainage improvement prior to resurfacing.",
            "Coordinate drainage repairs with pavement rehabilitation works."
        ]
        severity = "High" if drainage in ["Poor", "Blocked"] else "Medium"
        priority = "High" if severity == "High" else "Medium"
        notes = f"Drainage={drainage}; pothole delta={pothole_delta}; crack delta={crack_delta}."

        self._append_rec("Drainage & Pavement Moisture", severity, 1, actions, priority, notes)

    # ---------------------------
    # Prioritization
    # ---------------------------
    def rule_prioritization(self) -> JSONDict:
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

        overall = "Low"
        if priorities["High"] > 0:
            overall = "High"
        elif priorities["Medium"] > 0:
            overall = "Medium"

        return {"overall_priority": overall, "counts": priorities}

    # ---------------------------
    # Main generate method
    # ---------------------------
    def generate(self) -> JSONDict:
        self.recommendations = []

        # Execute all 9 rules
        try:
            self.rule_potholes()
            self.rule_cracks()
            self.rule_faded_markings()
            self.rule_road_signs()
            self.rule_traffic_lights()
            self.rule_streetlights_and_furniture()
            self.rule_speed_breakers()
            self.rule_guardrails()
            self.rule_drainage_and_surface()
        except Exception as e:
            print(f"[ERROR] Rule execution failed: {e}")
            self.recommendations.append({
                "issue": "RuleEngineError",
                "severity": "High",
                "count": 0,
                "priority": "High",
                "suggested_actions": [],
                "notes": f"Rule engine error: {str(e)}"
            })

        priority_summary = self.rule_prioritization()

        irc_report: JSONDict = {
            "generated_on": datetime.now().isoformat(),
            "source_audit": str(self.audit_json_path),
            "gps": self.audit.get("gps", {}),
            "gis_profile": self.audit.get("gis_profile", {}),
            "pci_summary": self.audit.get("pci_data", {}),
            "aggregate_comparison": self.audit.get("aggregate_comparison", {}),
            "defect_count_summary": dict(self.defect_counters),
            "recommendations": self.recommendations,
            "priority_summary": priority_summary,
            "notes": "Comprehensive IRC-based recommendations covering all 8 defect types. References: IRC:35, IRC:67, IRC:83, IRC:96, IRC:103, IRC:116, IRC:SP:76, IRC:SP:100."
        }

        # Write to results
        out_path = Path("results") / "irc_output.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(irc_report, f, indent=2, ensure_ascii=False)

        print(f"✅ Enhanced IRC report generated: {out_path}")
        return irc_report


# Quick test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python irc_solution_generator.py <audit_json_path>")
        sys.exit(1)

    gen = EnhancedIRCSolutionGenerator(sys.argv[1])
    report = gen.generate()
    print(f"Total recommendations: {len(report['recommendations'])}")
    for rec in report["recommendations"]:
        print(f"  - {rec['issue']}: {rec['severity']} (Priority: {rec['priority']})")