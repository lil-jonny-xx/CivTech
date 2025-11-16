"""
LaTeX PDF Report Generator

FIXES (V9.7.2):
- CRITICAL FIX: Removed aggressive colon-escaping (.replace(':', ':\\'))
  that caused 'Undefined control sequence' crash.
- FORMAT FIX: Narrowed longtable columns to fix 'Overfull \hbox' warning.
- Added 'VRUs' to the summary table.

This is an updated file. Save as: latex_report_generator.py
"""
import json
from pathlib import Path

class LatexReportGenerator:
    """Generates a .tex file from audit data."""

    def __init__(self, report_data, irc_solutions):
        self.report = report_data
        self.solutions = irc_solutions
        self.tex_lines = []

    def _write_header(self):
        """Writes the LaTeX preamble."""
        self.tex_lines = [
            "\\documentclass[11pt, a4paper]{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{longtable}",
            "\\usepackage{xcolor}",
            "\\usepackage{hyperref}",
            "\\hypersetup{colorlinks=true, urlcolor=blue, linkcolor=black}",
            "",
            "\\title{Road Safety Audit Report V9.7}",
            "\\author{Automated Analysis System}",
            f"\\date{{{self.report.get('audit_date', '')}}}",
            "",
            "\\begin{document}",
            "",
            "\\maketitle",
        ]

    def _write_summary(self):
        """Writes the main summary tables."""
        self.tex_lines.append("\\section{Executive Summary}")
        self.tex_lines.append("This report details the findings of an automated comparative road audit.")
        
        comp = self.report.get('aggregate_comparison', {})
        self.tex_lines.extend([
            "\\subsection{Aggregate Comparison}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "\\textbf{Metric} & \\textbf{Base} & \\textbf{Present} & \\textbf{Change} \\\\",
            "\\midrule",
        ])
        
        # Helper to format table rows
        def add_row(metric, key, is_score=False):
            data = comp.get(key, {})
            base = data.get('base', 0)
            present = data.get('present', 0)
            change = data.get('change', 0)
            if is_score:
                self.tex_lines.append(f"{metric} & {base:.1f} & {present:.1f} & {change:+.1f} \\\\")
            else:
                self.tex_lines.append(f"{metric} & {base} & {present} & {change:+} \\\\")

        add_row("Potholes", 'potholes')
        add_row("Cracks", 'cracks')
        add_row("Road Signs", 'road_signs')
        add_row("Traffic Lights", 'traffic_lights')
        add_row("Road Furniture", 'furniture')
        add_row("VRUs", 'vru') # <-- NEW V9.7
        add_row("Pavement Score", 'pavement_condition', is_score=True)
        add_row("Marking Score", 'marking_condition', is_score=True)

        self.tex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\subsection{Issue Summary by Severity}",
            "\\begin{tabular}{lr}",
            "\\toprule",
            "\\textbf{Severity} & \\textbf{Total Issues} \\\\",
            "\\midrule",
        ])
        
        issues = self.report.get('issue_summary', {}).get('by_severity', {})
        self.tex_lines.append(f"\\textcolor{{red!70!black}}{{High}} & {issues.get('high', 0)} \\\\")
        self.tex_lines.append(f"\\textcolor{{orange!90!black}}{{Medium}} & {issues.get('medium', 0)} \\\\")
        self.tex_lines.append(f"\\textcolor{{yellow!80!black}}{{Low}} & {issues.get('low', 0)} \\\\")
        self.tex_lines.append(f"\\midrule")
        self.tex_lines.append(f"\\textbf{{Total}} & \\textbf{{{self.report.get('issue_summary', {}).get('total_issues', 0)}}} \\\\")
        self.tex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ])

    def _write_irc_solutions(self):
        """Writes the IRC solutions table."""
        self.tex_lines.append("\\section{Recommended Interventions (IRC Standards)}")
        
        # --- FORMATTING FIX: Column 2 width changed from 0.55 to 0.50 ---
        self.tex_lines.extend([
            "\\begin{longtable}{p{0.15\\textwidth} p{0.50\\textwidth} p{0.25\\textwidth}}",
            "\\toprule",
            "\\textbf{Priority} & \\textbf{Recommendation} & \\textbf{Reference} \\\\",
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            "\\textbf{Priority} & \\textbf{Recommendation} & \\textbf{Reference} \\\\",
            "\\midrule",
            "\\endhead",
            "\\bottomrule",
            "\\endfoot",
        ])
        
        for s in self.solutions:
            priority = s.get('priority', 'N/A')
            if 'P1' in priority or 'Critical' in priority:
                priority = f"\\textcolor{{red!70!black}}{{{priority}}}"
            elif 'P2' in priority or 'Urgent' in priority:
                priority = f"\\textcolor{{red!80!black}}{{{priority}}}"
            elif 'P3' in priority or 'Warning' in priority:
                priority = f"\\textcolor{{orange!90!black}}{{{priority}}}"
            
            # --- CRASH FIX: Removed .replace(':', ':\\') from 'rec' variable ---
            rec = s.get('recommendation', 'N/A').replace('_', '\\_')
            ref = s.get('reference', 'N/A').replace('_', '\\_')
            
            self.tex_lines.append(f"{priority} & {rec} & {ref} \\\\")
            
        self.tex_lines.append("\\end{longtable}")
        self.tex_lines.append("")


    def _write_visuals(self):
        """Writes the section for visual comparisons."""
        self.tex_lines.append("\\section{Visual Evidence (Top Changes)}")
        self.tex_lines.append(
            "Visual comparisons for the top 10 most significant changes detected. "
            "Base video on the left, present video on the right."
        )
        self.tex_lines.append("")

        comp_dir = Path("results/comparisons")
        if comp_dir.exists():
            images = sorted(list(comp_dir.glob("*.jpg")))
            if not images:
                self.tex_lines.append("No comparison images found or generated.")
                return

            for img_path in images:
                # Use relative path and escape spaces
                rel_path = f"results/comparisons/{img_path.name}"
                rel_path = rel_path.replace(" ", "\\ ")
                
                # Clean up caption
                caption = img_path.stem.replace('_', ' ').replace(' ', ' ')
                
                self.tex_lines.extend([
                    "\\begin{figure}[h!]",
                    "  \\centering",
                    f"  \\includegraphics[width=0.9\\textwidth]{{{rel_path}}}",
                    f"  \\caption{{Visual comparison for {caption}}}",
                    "\\end{figure}",
                    "\\clearpage",
                ])
        else:
            self.tex_lines.append("Comparison image directory not found.")
            
    def _write_footer(self):
        """Closes the document."""
        self.tex_lines.append("\\end{document}")

    def generate_tex(self, output_path="final_audit_report.tex"):
        """Generates the full .tex file."""
        print(f"\n   Generating LaTeX report: {output_path}")
        self.tex_lines = [] # Clear previous lines
        self._write_header()
        self._write_summary()
        self._write_irc_solutions()
        self._write_visuals()
        self._write_footer()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.tex_lines))
            print(f"   [SUCCESS] LaTeX report saved: {output_path}")
        except Exception as e:
            print(f"   [ERROR] Error saving LaTeX report: {e}")