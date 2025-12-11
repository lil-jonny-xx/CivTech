"""
latex_report_generator.py — FIXED VERSION WITH PROPER REGENERATION
-------------------------------------------------------------------
Ensures fresh PDF generation every time with proper cleanup.
"""

import json
from pathlib import Path
import subprocess
from datetime import datetime
import glob
import shutil


class LatexReportGenerator:
    def __init__(self, audit_json, irc_json=None):
        self.audit_json = Path(audit_json)
        if not self.audit_json.exists():
            raise FileNotFoundError(f"Audit JSON missing: {audit_json}")

        if irc_json and Path(irc_json).exists():
            self.irc_json = Path(irc_json)
        else:
            self.irc_json = None

        # Load audit JSON
        with open(self.audit_json, "r", encoding="utf-8") as f:
            self.audit = json.load(f)

        # Load IRC JSON
        if self.irc_json:
            with open(self.irc_json, "r", encoding="utf-8") as f:
                self.irc = json.load(f)
        else:
            self.irc = {"recommendations": [], "priority_summary": {}}

        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)

        # Use timestamp to ensure uniqueness
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tex_path = self.output_dir / "report.tex"
        self.pdf_path = self.output_dir / "report.pdf"

    def _cleanup_old_files(self):
        """Remove old LaTeX auxiliary files and PDF - with verification"""
        patterns = ["report.*"]
        failed_deletes = []
        
        for pattern in patterns:
            for f in self.output_dir.glob(pattern):
                if not f.is_file():
                    continue
                
                # Try multiple times (Windows file locks)
                deleted = False
                for attempt in range(3):
                    try:
                        f.unlink()
                        deleted = True
                        break
                    except Exception as e:
                        if attempt == 2:
                            failed_deletes.append((str(f), str(e)))
                        import time
                        time.sleep(0.1)
        
        if failed_deletes:
            print("[LaTeX][WARN] Could not delete some files:")
            for fname, error in failed_deletes:
                print(f"  - {fname}: {error}")
            print("[LaTeX] This may cause PDF regeneration issues.")

    def _escape(self, text):
        """Properly escape LaTeX special characters"""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        # Replace in specific order to avoid double-escaping
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

    def _latex_image_block(self):
        """Embed comparison images"""
        out = ""
        comparison_dir = Path("results/comparisons")
        
        if not comparison_dir.exists():
            return "\\textit{No comparison images found.}\n"
        
        images = sorted(comparison_dir.glob("comp_*.jpg"))

        if not images:
            return "\\textit{No comparison images found.}\n"

        # Limit to 30 images to keep PDF reasonable
        for img in images[:30]:
            # Convert to relative path from results/ directory and normalize
            try:
                # Get path relative to results directory
                img_rel = img.relative_to(self.output_dir.parent)
                # Use forward slashes for LaTeX (works on Windows and Unix)
                img_path = str(img_rel).replace("\\", "/")
            except ValueError:
                # Fallback if relative path fails
                img_path = str(img).replace("\\", "/")
            
            out += (
                "\\begin{figure}[h!]\n"
                "\\centering\n"
                f"\\includegraphics[width=0.92\\textwidth]{{{img_path}}}\n"
                "\\caption{Before/After Comparison}\n"
                "\\end{figure}\n"
                "\\clearpage\n\n"
            )

        return out

    def build_tex(self):
        """Build complete LaTeX document"""
        gps = self.audit.get("gps", {})
        pci = self.audit.get("pci_data", {})
        gis = self.audit.get("gis_profile", {})
        agg = self.audit.get("aggregate_comparison", {})
        irc_recs = self.irc.get("recommendations", [])
        priority_summary = self.irc.get("priority_summary", {})
        
        # Get current timestamp
        report_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")

        # Document preamble
        tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{float}
\geometry{margin=0.8in}

\begin{document}

\title{\textbf{Road Safety Audit Report}}
\author{Automated Road Audit System}
\date{""" + report_date + r"""}
\maketitle

\tableofcontents
\clearpage

"""

        # GPS Location Section
        tex += r"""
\section{GPS Location}
\begin{tabular}{ll}
\textbf{Latitude:} & """ + self._escape(str(gps.get("latitude", "N/A"))) + r""" \\
\textbf{Longitude:} & """ + self._escape(str(gps.get("longitude", "N/A"))) + r""" \\
\end{tabular}

"""

        # GIS Profile Section
        tex += r"""
\section{GIS Profile}
"""
        if gis:
            tex += r"""\begin{longtable}{|l|p{10cm}|}
\hline
\textbf{Attribute} & \textbf{Value} \\ \hline
\endhead
"""
            for k, v in gis.items():
                tex += f"{self._escape(k.replace('_', ' ').title())} & {self._escape(str(v))} \\\\ \\hline\n"
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No GIS profile data available.}\n\n"

        # PCI Section
        tex += r"""
\section{Pavement Condition Index (PCI)}
"""
        base_pci = pci.get("base", {})
        pres_pci = pci.get("present", {})
        delta_pci = pci.get("delta", 0)
        
        tex += r"""\begin{tabular}{|l|l|}
\hline
\textbf{Metric} & \textbf{Value} \\ \hline
Base PCI Score & """ + str(base_pci.get("score", "-")) + r""" \\
Base Rating & """ + self._escape(str(base_pci.get("rating", "-"))) + r""" \\ \hline
Present PCI Score & """ + str(pres_pci.get("score", "-")) + r""" \\
Present Rating & """ + self._escape(str(pres_pci.get("rating", "-"))) + r""" \\ \hline
\textbf{Delta (Change)} & \textbf{""" + str(delta_pci) + r"""} \\ \hline
\end{tabular}

"""

        # Aggregate Comparison
        tex += r"""
\section{Aggregate Defect Comparison}
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
                tex += f"{label} & {base_val} & {pres_val} & {delta_val} \\\\ \\hline\n"
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No aggregate comparison data available.}\n\n"

        # IRC Recommendations
        tex += r"""
\section{IRC Maintenance Recommendations}
"""
        if irc_recs:
            tex += r"""\begin{longtable}{|p{3cm}|p{2cm}|p{1.8cm}|p{7cm}|}
\hline
\textbf{Issue} & \textbf{Severity} & \textbf{Priority} & \textbf{Suggested Actions} \\ \hline
\endhead
"""
            for rec in irc_recs:
                issue = self._escape(rec.get("issue", ""))
                severity = self._escape(rec.get("severity", ""))
                priority = self._escape(rec.get("priority", ""))
                
                actions = rec.get("suggested_actions", [])
                actions_text = " \\newline ".join([self._escape(a) for a in actions])
                
                tex += f"{issue} & {severity} & {priority} & {actions_text} \\\\ \\hline\n"
            
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No IRC recommendations available.}\n\n"

        # Priority Summary
        tex += r"""
\section{Maintenance Priority Summary}
"""
        if priority_summary:
            counts = priority_summary.get("counts", {})
            tex += r"""\begin{tabular}{|l|c|}
\hline
\textbf{Priority Level} & \textbf{Count} \\ \hline
Overall Priority & """ + self._escape(str(priority_summary.get("overall_priority", "N/A"))) + r""" \\ \hline
High Priority Items & """ + str(counts.get("High", 0)) + r""" \\ \hline
Medium Priority Items & """ + str(counts.get("Medium", 0)) + r""" \\ \hline
Low Priority Items & """ + str(counts.get("Low", 0)) + r""" \\ \hline
\end{tabular}

"""
        else:
            tex += "\\textit{No priority summary available.}\n\n"

        # Frame-level changes
        tex += r"""
\section{Frame-Level Deterioration Summary}
"""
        changes = self.audit.get("frame_level_changes", [])
        if changes:
            # Limit to first 100 changes for readability
            tex += r"""\begin{longtable}{|c|c|p{9cm}|}
\hline
\textbf{Frame} & \textbf{Time (s)} & \textbf{Changes Detected} \\ \hline
\endhead
"""
            for change in changes[:100]:
                frame_id = change.get("frame_id", "-")
                timestamp = round(change.get("timestamp_seconds", 0), 2)
                
                change_items = change.get("changes", [])
                change_desc = " \\newline ".join([
                    f"{self._escape(c.get('element', ''))}: {self._escape(c.get('type', ''))}"
                    for c in change_items
                ])
                
                tex += f"{frame_id} & {timestamp} & {change_desc} \\\\ \\hline\n"
            
            tex += "\\end{longtable}\n\n"
        else:
            tex += "\\textit{No frame-level deterioration detected.}\n\n"

        # Comparison images
        tex += r"""
\section{Before/After Comparison Images}
"""
        tex += self._latex_image_block()

        # End document
        tex += r"""
\end{document}
"""

        return tex

    def generate(self):
        """Generate LaTeX and compile to PDF"""
        
        # Clean up old files first
        self._cleanup_old_files()
        
        # Build LaTeX content
        print("[LaTeX] Building document...")
        tex_content = self.build_tex()

        # Write .tex file
        with open(self.tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        
        print(f"[LaTeX] Written to {self.tex_path}")

        # Try to compile with pdflatex
        try:
            print("[LaTeX] Compiling PDF (this may take a moment)...")
            
            # Run pdflatex twice for proper cross-references
            for run in [1, 2]:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", str(self.tex_path.name)],
                    cwd=str(self.output_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                    check=False
                )
                
                if run == 1:
                    print("[LaTeX] First pass complete")
                else:
                    print("[LaTeX] Second pass complete")
            
            # Check if PDF was created
            if self.pdf_path.exists():
                print(f"[LaTeX] ✅ PDF generated: {self.pdf_path}")
                return (str(self.tex_path), str(self.pdf_path))
            else:
                print("[LaTeX] ⚠️ PDF not created. Check LaTeX logs.")
                
                # Try to show error from log
                log_file = self.output_dir / "report.log"
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        log_lines = f.readlines()
                        # Find error lines
                        for i, line in enumerate(log_lines):
                            if "! " in line or "Error" in line:
                                print(f"[LaTeX Error] {line.strip()}")
                                if i + 1 < len(log_lines):
                                    print(f"             {log_lines[i+1].strip()}")
                
                return (str(self.tex_path), None)
                
        except FileNotFoundError:
            print("[LaTeX] ⚠️ pdflatex not found. Install TeX Live or MikTeX.")
            print("[LaTeX] .tex file created, but PDF compilation skipped.")
            return (str(self.tex_path), None)
            
        except subprocess.TimeoutExpired:
            print("[LaTeX] ⚠️ Compilation timeout. Document may be too large.")
            return (str(self.tex_path), None)
            
        except Exception as e:
            print(f"[LaTeX] ⚠️ Compilation error: {e}")
            return (str(self.tex_path), None)


# CLI test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python latex_report_generator.py <audit_json> [irc_json]")
        sys.exit(1)
    
    audit_path = sys.argv[1]
    irc_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    gen = LatexReportGenerator(audit_path, irc_path)
    tex, pdf = gen.generate()
    
    print(f"\nGenerated files:")
    print(f"  LaTeX: {tex}")
    print(f"  PDF:   {pdf or 'Not generated'}")