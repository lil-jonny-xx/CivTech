"""
IRC (Indian Roads Congress) Solution Engine

FIX (V9.7): Added 'VRU' rule (IRC:103-2022).
"""
import json

class IRCSolutionEngine:
    """
    Generates maintenance solutions based on an audit report
    and simplified IRC (Indian Roads Congress) guidelines.
    """

    def __init__(self):
        # These are simplified rules for demonstration
        self.pothole_rules = {
            'high_change': "Urgent: Multiple new potholes detected. Immediate patching required as per IRC:SP:84-2019.",
            'low_change': "Routine: Minor increase in potholes. Schedule for routine patching program."
        }
        self.pavement_rules = {
            'severe_drop': "Critical: Significant pavement score drop (< 40). Requires structural overlay or reconstruction. (Ref: IRC-37-2018)",
            'moderate_drop': "Warning: Pavement score deteriorating. Requires sealing or corrective course. (Ref: IRC-81-1997)",
        }
        self.marking_rules = {
            'severe_drop': "High Priority: Markings score below 30 (severely faded/missing). Immediate repainting required for safety. (Ref: IRC-35-2015)",
            'moderate_drop': "Priority: Markings score below 60 (faded). Schedule for repainting within 3 months."
        }
        self.sign_rules = {
            'missing': "Urgent: Road signs missing. Replace immediately. (Ref: IRC-67-2022)"
        }
        self.traffic_light_rules = {
            'missing': "Critical: Traffic light failure or missing. Requires immediate dispatch and repair. (Ref: IRC-93-1985)"
        }
        self.furniture_rules = {
            'missing': "Low Priority: Road furniture (benches, etc.) missing or damaged. Schedule for inspection and replacement as per asset management plan. (Ref: IRC:SP:87-2019)"
        }
        # --- NEW (V9.7) ---
        self.vru_rules = {
            'increase': "Warning: Significant increase in VRU (pedestrian/cyclist) traffic detected. Review pedestrian facilities and traffic calming measures. (Ref: IRC:103-2022)"
        }
        # --- END NEW ---


    def generate_solutions(self, report_data):
        """
        Main function to generate a list of solutions.
        """
        print("\n" + "="*70)
        print(" RUNNING IRC SOLUTION ENGINE (V9.7)")
        print("="*70)
        
        solutions = []
        comp = report_data.get('aggregate_comparison', {})
        
        # 1. Analyze Potholes
        pothole_change = comp.get('potholes', {}).get('change', 0)
        if pothole_change > 5:
            solutions.append({
                'priority': 'P2 - Urgent',
                'recommendation': self.pothole_rules['high_change'],
                'reference': 'IRC:SP:84-2019'
            })
        elif pothole_change > 0:
            solutions.append({
                'priority': 'P4 - Routine',
                'recommendation': self.pothole_rules['low_change'],
                'reference': 'N/A'
            })

        # 2. Analyze Pavement
        pav_comp = comp.get('pavement_condition', {})
        pav_change = pav_comp.get('change', 0)
        pav_present_score = pav_comp.get('present', 100)
        
        if pav_change < -20:
            if pav_present_score < 40:
                solutions.append({
                    'priority': 'P1 - Critical',
                    'recommendation': self.pavement_rules['severe_drop'],
                    'reference': 'IRC-37-2018'
                })
            else:
                 solutions.append({
                    'priority': 'P3 - Warning',
                    'recommendation': self.pavement_rules['moderate_drop'],
                    'reference': 'IRC-81-1997'
                })
        
        # 3. Analyze Markings
        mark_comp = comp.get('marking_condition', {})
        mark_present_score = mark_comp.get('present', 100)

        if mark_present_score < 30:
            solutions.append({
                'priority': 'P2 - Urgent',
                'recommendation': self.marking_rules['severe_drop'],
                'reference': 'IRC-35-2015'
            })
        elif mark_present_score < 60:
            solutions.append({
                'priority': 'P3 - Priority',
                'recommendation': self.marking_rules['moderate_drop'],
                'reference': 'IRC-35-2015'
            })

        # 4. Analyze Road Signs
        sign_change = comp.get('road_signs', {}).get('change', 0)
        if sign_change < 0:
            solutions.append({
                'priority': 'P2 - Urgent',
                'recommendation': self.sign_rules['missing'],
                'reference': 'IRC-67-2022'
            })
            
        # 5. Analyze Traffic Lights
        light_change = comp.get('traffic_lights', {}).get('change', 0)
        if light_change < 0:
            solutions.append({
                'priority': 'P1 - Critical',
                'recommendation': self.traffic_light_rules['missing'],
                'reference': 'IRC-93-1985'
            })

        # 6. Analyze Road Furniture
        furn_change = comp.get('furniture', {}).get('change', 0)
        if furn_change < 0:
            solutions.append({
                'priority': 'P4 - Low Priority',
                'recommendation': self.furniture_rules['missing'],
                'reference': 'IRC:SP:87-2019'
            })

        # 7. Analyze VRUs (NEW V9.7)
        vru_change = comp.get('vru', {}).get('change', 0)
        if vru_change > 10: # Flag if more than 10 new VRUs are consistently detected
            solutions.append({
                'priority': 'P3 - Warning',
                'recommendation': self.vru_rules['increase'],
                'reference': 'IRC:103-2022'
            })

        if not solutions:
            print("   [SUCCESS] No high-priority interventions required.")
            solutions.append({
                'priority': 'P5 - Good',
                'recommendation': 'No significant deterioration detected. Continue routine monitoring.',
                'reference': 'N/A'
            })
        else:
            print(f"   [SUCCESS] Generated {len(solutions)} recommendations.")

        # Sort by priority
        solutions.sort(key=lambda x: x['priority'])
        return solutions