"""
latex_report_generator.py - GENERATES TEX FILE ONLY
No PDF compilation. Just creates the .tex file for reference.
PDF is generated separately using fpdf2 from IRC JSON.
"""

import json
from pathlib import Path
from datetime import datetime


class LatexReportGenerator:
    """
    Generates LaTeX (.tex) files from audit and IRC data
    Does NOT compile to PDF - PDF is generated via fpdf2 instead
    """
    
    def __init__(self, audit_json, irc_json=None):
        self.audit_json = Path(audit_json)
        
        if not self.audit_json.exists():
            raise FileNotFoundError(f"Audit JSON missing: {audit_json}")
        
        # Load audit data
        with open(self.audit_json, "r", encoding="utf-8") as f:
            self.audit = json.load(f)
        
        # Load IRC data
        if irc_json and Path(irc_json).exists():
            with open(irc_json, "r", encoding="utf-8") as f:
                self.irc = json.load(f)
        else:
            self.irc = {"recommendations": [], "priority_summary": {}}
        
        self.output_dir = Path("results").absolute()
        self.output_dir.mkdir(exist_ok=True)
        
        self.tex_path = self.output_dir / "report.tex"
    
    def _escape(self, text):
        """Escape LaTeX special characters"""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        replacements = [
            ("\\", "\\textbackslash{}"),
            ("{", "\\{"),
            ("}", "\\}"),
            ("$", "\\$"),
            ("&", "\\&"),
            ("%", "\\%"),
            ("_", "\\_"),
            ("#", "\\#"),
            ("~", "\\textasciitilde{}"),
            ("^", "\\textasciicircum{}"),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def build_tex(self):
        """Build LaTeX document"""
        
        gps = self.audit.get("gps", {})
        pci = self.audit.get("pci_data", {})
        gis = self.audit.get("gis_profile", {})
        agg = self.audit.get("aggregate_comparison", {})
        frames_analyzed = self.audit.get("frames_analyzed", {})
        irc_recs = self.irc.get("recommendations", [])
        priority_summary = self.irc.get("priority_summary", {})
        
        report_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{fontspec}
\usepackage{xunicode}
\usepackage{xltxtra}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{float}
\usepackage{hyperref}
\usepackage{fancyhdr}

\geometry{margin=0.9in}

\setmainfont{Calibri}
\setsansfont{Calibri}
\setmonofont{Courier New}

\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}

\pagestyle{fancy}
\fancyhf{}
\rhead{Road Safety Audit}
\lhead{Report}
\cfoot{\thepage}

\begin{document}

\title{\textbf{Road Safety Audit Report}}
\author{Automated Road Safety Audit System}
\date{""" + report_date + r"""}
\maketitle

\tableofcontents
\newpage

"""
        
        # Section 1: GPS Location
        tex += r"""\section{GPS Location \& Coordinates}
"""
        lat = gps.get("latitude", "N/A")
        lon = gps.get("longitude", "N/A")
        tex += f"""\\textbf{{Latitude:}} {self._escape(str(lat))} \\\\[0.3cm]
\\textbf{{Longitude:}} {self._escape(str(lon))} \\\\[0.5cm]

"""
        
        # Section 2: GIS Profile
        tex += r"""\section{GIS Context \& Environmental Profile}
"""
        if gis:
            tex += r"""\begin{longtable}{|l|p{10cm}|}
\hline
\textbf{Attribute} & \textbf{Value} \\ \hline
\endhead
"""
            for k, v in gis.items():
                key_name = self._escape(k.replace("_", " ").title())
                value = self._escape(str(v))
                tex += f"{key_name} & {value} \\\\ \\hline\n"
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No GIS profile data available.}\\\\[0.5cm]\n\n"
        
        # Section 3: Frames Analyzed
        tex += r"""\section{Video Analysis Summary}
"""
        base_frames = frames_analyzed.get("base", 0)
        present_frames = frames_analyzed.get("present", 0)
        tex += f"""\\begin{{tabular}}{{|l|r|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Count}} \\\\
\\hline
Base Video Frames Analyzed & {base_frames} \\\\
Present Video Frames Analyzed & {present_frames} \\\\
\\hline
\\end{{tabular}}\\\\[0.5cm]

"""
        
        # Section 4: Pavement Condition Index
        tex += r"""\section{Pavement Condition Index (PCI)}
"""
        base_pci = pci.get("base", {})
        pres_pci = pci.get("present", {})
        delta_pci = pci.get("delta", 0)
        
        base_score = base_pci.get("score", "-")
        base_rating = self._escape(str(base_pci.get("rating", "-")))
        pres_score = pres_pci.get("score", "-")
        pres_rating = self._escape(str(pres_pci.get("rating", "-")))
        
        tex += f"""\\begin{{longtable}}{{|l|c|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Base Video}} & \\textbf{{Present Video}} \\\\
\\hline
\\endhead
PCI Score & {base_score} & {pres_score} \\\\
Rating & {base_rating} & {pres_rating} \\\\
\\hline
\\end{{longtable}}

\\textbf{{PCI Change (Delta):}} {delta_pci:+d} points\\\\[0.5cm]

"""
        
        # Section 5: Defect Comparison
        tex += r"""\section{Aggregate Defect Comparison (All 8 Types)}
"""
        if agg:
            tex += r"""\begin{longtable}{|l|c|c|c|}
\hline
\textbf{Defect Type} & \textbf{Base} & \textbf{Present} & \textbf{Delta} \\ \hline
\endhead
"""
            for defect_name, counts in agg.items():
                label = self._escape(defect_name.replace("_", " ").title())
                base_val = counts.get("base", 0)
                pres_val = counts.get("present", 0)
                delta_val = counts.get("delta", 0)
                tex += f"{label} & {base_val} & {pres_val} & {delta_val:+d} \\\\ \\hline\n"
            tex += "\\end{longtable}\\\\[0.5cm]\n\n"
        else:
            tex += "\\textit{No defect comparison data available.}\\\\[0.5cm]\n\n"
        
        # Section 6: IRC Recommendations
        tex += r"""\section{IRC-Based Maintenance Recommendations (All 8 Defects)}
"""
        if irc_recs:
            tex += f"""Total Recommendations: {len(irc_recs)}\\\\[0.3cm]

"""
            for idx, rec in enumerate(irc_recs, 1):
                issue = self._escape(rec.get("issue", "Unknown"))
                severity = self._escape(rec.get("severity", "N/A"))
                priority = self._escape(rec.get("priority", "N/A"))
                count = rec.get("count", 0)
                notes = self._escape(rec.get("notes", ""))
                actions = rec.get("suggested_actions", [])
                
                tex += f"""\\subsection{{{idx}. {issue}}}
\\textbf{{Severity:}} {severity} \\\\
\\textbf{{Priority:}} {priority} \\\\
\\textbf{{Count:}} {count} \\\\
\\textbf{{Notes:}} {notes}\\\\[0.2cm]

\\textbf{{Suggested Actions:}}
\\begin{{enumerate}}
"""
                for action in actions:
                    action_escaped = self._escape(action)
                    tex += f"\\item {action_escaped}\n"
                
                tex += "\\end{enumerate}\n\n"
        else:
            tex += "\\textit{No IRC recommendations available.}\\\\[0.5cm]\n\n"
        
        # Section 7: Priority Summary
        tex += r"""\section{Maintenance Priority Summary}
"""
        if priority_summary:
            counts = priority_summary.get("counts", {})
            overall = self._escape(str(priority_summary.get("overall_priority", "N/A")))
            high_count = counts.get("High", 0)
            med_count = counts.get("Medium", 0)
            low_count = counts.get("Low", 0)
            
            tex += f"""\\begin{{tabular}}{{|l|r|}}
\\hline
\\textbf{{Priority Level}} & \\textbf{{Count}} \\\\
\\hline
Overall Priority & {overall} \\\\
High Priority Items & {high_count} \\\\
Medium Priority Items & {med_count} \\\\
Low Priority Items & {low_count} \\\\
\\hline
\\textbf{{TOTAL}} & \\textbf{{{high_count + med_count + low_count}}} \\\\
\\hline
\\end{{tabular}}\\\\[0.5cm]

"""
        else:
            tex += "\\textit{No priority summary available.}\\\\[0.5cm]\n\n"
        
        # Footer
        tex += r"""\section*{Report Information}
\textit{This report was automatically generated by the Road Safety Audit System.}\\
\textit{For detailed analysis, refer to the JSON outputs in the Downloads section.}\\[0.3cm]
\textit{Report Generated:} """ + report_date + r"""

\end{document}
"""
        
        return tex
    
    def generate(self):
        """Generate TEX file only (no PDF compilation)"""
        
        print("\n[TEX] ============ LATEX FILE GENERATION START ============")
        
        # Build and write LaTeX
        print("[TEX] Building LaTeX document...")
        tex_content = self.build_tex()
        
        try:
            self.tex_path.write_text(tex_content, encoding="utf-8")
            file_size = self.tex_path.stat().st_size
            print(f"[TEX] ✓ LaTeX file written: {self.tex_path.name} ({file_size:,} bytes)")
            print("[TEX] ============ LATEX FILE GENERATION COMPLETE ============\n")
            return (str(self.tex_path), None)  # Return (tex_path, None) - no PDF
        except Exception as e:
            print(f"[TEX] ✗ Failed to write LaTeX file: {e}")
            return (None, None)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python latex_report_generator.py <audit_json> [irc_json]")
        print("Example: python latex_report_generator.py results/audit_output.json results/irc_output.json")
        sys.exit(1)
    
    audit_path = sys.argv[1]
    irc_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        gen = LatexReportGenerator(audit_path, irc_path)
        tex, pdf = gen.generate()
        
        if tex:
            print(f"✓ LaTeX file generated: {tex}")
        else:
            print("✗ Failed to generate LaTeX file")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)