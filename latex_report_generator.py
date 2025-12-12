"""
latex_report_generator.py - LUALATEX OPTIMIZED VERSION
Works perfectly with LuaLaTeX (better Unicode support than pdflatex)

Key improvements:
- Optimized for lualatex specifically
- Better error detection and reporting
- Handles UTF-8 properly
- Uses fontspec for modern fonts
- Proper table formatting for large content
- Retry logic with cleanup between runs
"""

import json
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime


class LatexReportGenerator:
    """
    Generates comprehensive PDF reports using LuaLaTeX
    Works on Windows, macOS, and Linux
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
        self.pdf_path = self.output_dir / "report.pdf"
        self.log_path = self.output_dir / "report.log"
        
        # Intermediate files created by lualatex
        self.aux_path = self.output_dir / "report.aux"
        self.out_path = self.output_dir / "report.out"
    
    def _cleanup_files(self):
        """Remove old PDF and intermediate files that might cause issues"""
        files_to_remove = [self.pdf_path, self.aux_path, self.out_path, self.log_path]
        
        for f in files_to_remove:
            if f.exists():
                try:
                    f.unlink()
                except Exception as e:
                    print(f"[PDF] Warning: Could not delete {f.name}: {e}")
        
        # Give Windows filesystem time to release files
        time.sleep(0.3)
    
    def _escape(self, text):
        """Escape LaTeX special characters while preserving readability"""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        # Order matters! Backslash must be first
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
    
    def _format_list(self, items):
        """Format a list of strings for LaTeX"""
        if not items:
            return ""
        
        formatted = []
        for item in items:
            escaped = self._escape(str(item))
            formatted.append(f"\\item {escaped}")
        
        return "\n".join(formatted)
    
    def build_tex(self):
        """Build comprehensive LaTeX document"""
        
        gps = self.audit.get("gps", {})
        pci = self.audit.get("pci_data", {})
        gis = self.audit.get("gis_profile", {})
        agg = self.audit.get("aggregate_comparison", {})
        frames_analyzed = self.audit.get("frames_analyzed", {})
        irc_recs = self.irc.get("recommendations", [])
        priority_summary = self.irc.get("priority_summary", {})
        
        report_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        # Start LaTeX document with lualatex-friendly preamble
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
        tex += r"""\section{Aggregate Defect Comparison}
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
        tex += r"""\section{IRC-Based Maintenance Recommendations}
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
\textit{For detailed analysis, refer to the audit JSON outputs.}\\[0.3cm]
\textit{Report Generated:} """ + report_date + r"""

\end{document}
"""
        
        return tex
    
    def generate(self):
        """Generate LaTeX and compile to PDF using LuaLaTeX"""
        
        print("\n[PDF] ============ LUALATEX PDF GENERATION START ============")
        
        # Step 1: Cleanup
        print("[PDF] Cleaning up previous files...")
        self._cleanup_files()
        
        # Step 2: Build and write LaTeX
        print("[PDF] Building LaTeX document...")
        tex_content = self.build_tex()
        
        try:
            self.tex_path.write_text(tex_content, encoding="utf-8")
            print(f"[PDF] ✓ LaTeX file written: {self.tex_path.name}")
        except Exception as e:
            print(f"[PDF] ✗ Failed to write LaTeX file: {e}")
            return (str(self.tex_path), None)
        
        # Step 3: Compile with lualatex
        print("[PDF] Starting LuaLaTeX compilation...")
        print(f"[PDF] Working directory: {self.output_dir}")
        
        try:
            # Use lualatex with proper flags
            cmd = [
                "lualatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-shell-escape",
                str(self.tex_path)
            ]
            
            print(f"[PDF] Command: {' '.join(cmd)}")
            
            # Run lualatex (usually needs 1-2 passes for TOC)
            for run_num in range(1, 3):
                print(f"\n[PDF] === LuaLaTeX Run {run_num}/2 ===")
                
                result = subprocess.run(
                    cmd,
                    cwd=str(self.output_dir),
                    capture_output=True,
                    text=True,
                    timeout=90,
                    shell=False  # False for list format
                )
                
                output_text = result.stdout + result.stderr
                
                # Check for critical errors
                has_error = "! " in output_text or "Fatal" in output_text
                
                if has_error:
                    print(f"[PDF] ⚠ Run {run_num}: Issues detected")
                    # Show error context
                    for line in output_text.split('\n'):
                        if "!" in line or "Error" in line or "error" in line:
                            print(f"      {line[:120]}")
                else:
                    print(f"[PDF] ✓ Run {run_num}: Completed successfully")
                
                # Always print return code for debugging
                if result.returncode != 0:
                    print(f"[PDF] Return code: {result.returncode}")
        
        except subprocess.TimeoutExpired:
            print("[PDF] ✗ TIMEOUT: LuaLaTeX took too long (>90 sec)")
            return (str(self.tex_path), None)
        
        except FileNotFoundError:
            print("[PDF] ✗ ERROR: lualatex not found in PATH")
            print("[PDF] Install TeX Live or MiKTeX with lualatex")
            return (str(self.tex_path), None)
        
        except Exception as e:
            print(f"[PDF] ✗ ERROR: {type(e).__name__}: {e}")
            return (str(self.tex_path), None)
        
        # Step 4: Verify PDF generation
        print("\n[PDF] Verifying PDF creation...")
        time.sleep(0.5)
        
        if self.pdf_path.exists():
            pdf_size = self.pdf_path.stat().st_size
            
            if pdf_size > 1000:  # At least 1KB
                print(f"[PDF] ✓✓✓ SUCCESS! PDF CREATED ({pdf_size:,} bytes)")
                print(f"[PDF] Location: {self.pdf_path}")
                print("[PDF] ============ PDF GENERATION COMPLETE ============\n")
                return (str(self.tex_path), str(self.pdf_path))
            else:
                print(f"[PDF] ✗ PDF too small ({pdf_size} bytes) - likely empty")
                return (str(self.tex_path), None)
        else:
            print(f"[PDF] ✗ PDF not created at expected location")
            print(f"[PDF] Expected: {self.pdf_path}")
            
            # Debug: show generated files
            print("[PDF] Files generated:")
            for f in self.output_dir.glob("report.*"):
                size = f.stat().st_size
                print(f"[PDF]   {f.name}: {size:,} bytes")
            
            return (str(self.tex_path), None)


# =========================================================================
# TEST SCRIPT
# =========================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python latex_report_generator.py <audit_json> [irc_json]")
        print("Example: python latex_report_generator.py results/audit_output.json results/irc_output.json")
        sys.exit(1)
    
    audit_path = sys.argv[1]
    irc_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        print(f"Loading audit from: {audit_path}")
        if irc_path:
            print(f"Loading IRC from: {irc_path}")
        
        gen = LatexReportGenerator(audit_path, irc_path)
        tex, pdf = gen.generate()
        
        print(f"\n✓ Complete!")
        print(f"  LaTeX: {tex}")
        print(f"  PDF:   {pdf if pdf else 'NOT GENERATED'}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)