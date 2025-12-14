"""
fpdf_irc_report_generator.py - FIXED VERSION
- Handles Unicode characters properly
- Uses DejaVu font instead of Helvetica
- Generates PDF directly from IRC JSON
"""

import json
from pathlib import Path
from datetime import datetime
from fpdf import FPDF


class IRCToPDFGenerator:
    """
    Generates professional PDF reports directly from IRC JSON
    Using fpdf2 library with Unicode font support
    """
    
    def __init__(self, irc_json, audit_json=None):
        self.irc_json = Path(irc_json)
        self.audit_json = Path(audit_json) if audit_json else None
        
        if not self.irc_json.exists():
            raise FileNotFoundError(f"IRC JSON missing: {irc_json}")
        
        # Load IRC data
        with open(self.irc_json, "r", encoding="utf-8") as f:
            self.irc = json.load(f)
        
        # Load audit data if provided
        if self.audit_json and self.audit_json.exists():
            with open(self.audit_json, "r", encoding="utf-8") as f:
                self.audit = json.load(f)
        else:
            self.audit = {}
        
        self.output_dir = Path("results").absolute()
        self.output_dir.mkdir(exist_ok=True)
        
        self.pdf_path = self.output_dir / "report.pdf"
    
    def _sanitize_text(self, text):
        """Remove problematic Unicode characters"""
        if not isinstance(text, str):
            text = str(text)
        
        # Replace problematic characters
        text = text.replace("•", "-")
        text = text.replace("→", "->")
        text = text.replace("°", "deg")
        text = text.replace("±", "+-")
        text = text.replace("µ", "u")
        
        return text
    
    def generate(self):
            """Generate PDF from IRC JSON"""
            
            print("\n[PDF] ============ FPDF2 PDF GENERATION START ============")
            print("[PDF] Generating PDF directly from IRC JSON...")
            
            try:
                # Create PDF object with Unicode font
                pdf = FPDF(orientation="P", unit="mm", format="A4")
                pdf.set_auto_page_break(auto=True, margin=10)
                
                # ===========================================================
                # CRITICAL FIX: Register ALL font styles separately
                # Ensure these filenames match EXACTLY what is in your folder
                # ===========================================================
                
                # 1. Normal (Regular)
                pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
                
                # 2. Bold (Link "B" to your Bold file)
                pdf.add_font("DejaVu", "B", "DejaVuSansCondensed-Bold.ttf", uni=True)
                
                # 3. Italic (Link "I" to your Oblique/Italic file)
                pdf.add_font("DejaVu", "I", "DejaVuSansCondensed-Oblique.ttf", uni=True)
                
                # 4. Bold + Italic (Link "BI" to your BoldOblique file)
                pdf.add_font("DejaVu", "BI", "DejaVuSansCondensed-BoldOblique.ttf", uni=True)
                
                # Now set the default font
                pdf.set_font("DejaVu", "", 11)
                
                pdf.add_page()
                
                # Set fonts - use DejaVu for Unicode support
                # This "B" will now look for the file registered as "B" above
                pdf.set_font("DejaVu", "B", 20)
                pdf.cell(0, 15, "Road Safety Audit Report", ln=True, align="C")
                
                pdf.set_font("DejaVu", "I", 10)
                report_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")
                pdf.cell(0, 5, f"Generated: {report_date}", ln=True, align="C")
                pdf.ln(5)
                
                # Get data from IRC report
                gps = self.irc.get("gps", {})
                pci = self.irc.get("pci_summary", {})
                gis = self.irc.get("gis_profile", {})
                agg = self.irc.get("aggregate_comparison", {})
                recs = self.irc.get("recommendations", [])
                priority = self.irc.get("priority_summary", {})
                
                # Section 1: GPS
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 8, "1. GPS Location", ln=True)
                pdf.set_font("DejaVu", "", 10)
                lat = self._sanitize_text(str(gps.get("latitude", "N/A")))
                lon = self._sanitize_text(str(gps.get("longitude", "N/A")))
                pdf.cell(0, 6, f"Latitude: {lat}", ln=True)
                pdf.cell(0, 6, f"Longitude: {lon}", ln=True)
                pdf.ln(3)
                
                # Section 2: GIS Profile
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 8, "2. GIS Context & Environmental Profile", ln=True)
                pdf.set_font("DejaVu", "", 9)
                
                if gis:
                    for key, value in gis.items():
                        key_display = key.replace("_", " ").title()
                        value_str = self._sanitize_text(str(value))[:60]
                        pdf.cell(0, 5, f"  - {key_display}: {value_str}", ln=True)
                else:
                    pdf.cell(0, 5, "  No GIS data available", ln=True)
                pdf.ln(3)
                
                # Section 3: PCI
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 8, "3. Pavement Condition Index (PCI)", ln=True)
                pdf.set_font("DejaVu", "", 10)
                
                base_pci = pci.get("base", {})
                pres_pci = pci.get("present", {})
                delta = pci.get("delta", 0)
                
                pdf.cell(0, 6, f"Base PCI Score: {base_pci.get('score', 'N/A')} ({base_pci.get('rating', 'N/A')})", ln=True)
                pdf.cell(0, 6, f"Present PCI Score: {pres_pci.get('score', 'N/A')} ({pres_pci.get('rating', 'N/A')})", ln=True)
                pdf.cell(0, 6, f"PCI Change (Delta): {delta:+d} points", ln=True)
                pdf.ln(3)
                
                # Section 4: Defect Comparison
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 8, "4. Aggregate Defect Comparison (All 8 Types)", ln=True)
                pdf.set_font("DejaVu", "", 8)
                
                if agg:
                    # Table header
                    pdf.cell(45, 5, "Defect Type", border=1)
                    pdf.cell(18, 5, "Base", border=1, align="C")
                    pdf.cell(18, 5, "Present", border=1, align="C")
                    pdf.cell(18, 5, "Delta", border=1, align="C", ln=True)
                    
                    for defect_name, counts in agg.items():
                        label = self._sanitize_text(defect_name.replace("_", " ").title()[:30])
                        base = counts.get("base", 0)
                        pres = counts.get("present", 0)
                        delta = counts.get("delta", 0)
                        
                        pdf.cell(45, 5, label, border=1)
                        pdf.cell(18, 5, str(base), border=1, align="C")
                        pdf.cell(18, 5, str(pres), border=1, align="C")
                        pdf.cell(18, 5, f"{delta:+d}", border=1, align="C", ln=True)
                else:
                    pdf.cell(0, 5, "  No defect data available", ln=True)
                
                pdf.ln(3)
                
                # Section 5: Priority Summary
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 8, "5. Maintenance Priority Summary", ln=True)
                pdf.set_font("DejaVu", "", 10)
                
                overall_priority = priority.get("overall_priority", "N/A")
                counts = priority.get("counts", {})
                
                pdf.cell(0, 6, f"Overall Priority: {overall_priority}", ln=True)
                pdf.cell(0, 6, f"High Priority Items: {counts.get('High', 0)}", ln=True)
                pdf.cell(0, 6, f"Medium Priority Items: {counts.get('Medium', 0)}", ln=True)
                pdf.cell(0, 6, f"Low Priority Items: {counts.get('Low', 0)}", ln=True)
                pdf.ln(3)
                
                # Section 6: Recommendations
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 8, "6. IRC Maintenance Recommendations (All 8 Defects)", ln=True)
                pdf.set_font("DejaVu", "", 8)
                
                if recs:
                    pdf.cell(0, 5, f"Total Recommendations: {len(recs)}", ln=True)
                    pdf.ln(2)
                    
                    for idx, rec in enumerate(recs, 1):
                        # Issue title
                        pdf.set_font("DejaVu", "B", 9)
                        issue_name = self._sanitize_text(rec.get('issue', 'Unknown'))
                        pdf.cell(0, 5, f"{idx}. {issue_name}", ln=True)
                        
                        # Details
                        pdf.set_font("DejaVu", "", 7)
                        severity = self._sanitize_text(rec.get("severity", "N/A"))
                        priority_level = self._sanitize_text(rec.get("priority", "N/A"))
                        count = rec.get("count", 0)
                        
                        pdf.cell(0, 4, f"  Severity: {severity} | Priority: {priority_level} | Count: {count}", ln=True)
                        
                        # Notes (truncated)
                        notes = self._sanitize_text(rec.get("notes", ""))[:90]
                        if notes:
                            pdf.cell(0, 4, f"  Notes: {notes}", ln=True)
                        
                        # Actions - use dashes instead of bullets
                        actions = rec.get("suggested_actions", [])
                        for action in actions[:2]:  # Limit to 2 actions per item
                            action_text = self._sanitize_text(action)[:75]
                            pdf.cell(0, 4, f"    - {action_text}", ln=True)
                        
                        pdf.ln(1)
                        
                        # Add page break if needed
                        if idx % 5 == 0 and idx < len(recs):
                            pdf.add_page()
                            pdf.set_font("DejaVu", "B", 12)
                            pdf.cell(0, 8, "Recommendations (continued)", ln=True)
                            pdf.set_font("DejaVu", "", 8)
                else:
                    pdf.cell(0, 5, "  No recommendations available", ln=True)
                
                # Add new page for notes
                pdf.add_page()
                pdf.set_font("DejaVu", "B", 12)
                pdf.cell(0, 8, "Notes", ln=True)
                pdf.set_font("DejaVu", "", 10)
                
                notes_text = self._sanitize_text(self.irc.get("notes", ""))
                if notes_text:
                    pdf.multi_cell(0, 5, notes_text)
                else:
                    pdf.cell(0, 5, "Recommendations reference IRC guidelines.", ln=True)
                
                # Save PDF
                self.pdf_path.write_bytes(b"")  # Ensure file is writable
                pdf.output(str(self.pdf_path))
                
                pdf_size = self.pdf_path.stat().st_size
                print(f"[PDF] ✓ PDF generated: {self.pdf_path.name} ({pdf_size:,} bytes)")
                print("[PDF] ============ FPDF2 PDF GENERATION COMPLETE ============\n")
                
                return str(self.pdf_path)
            
            except Exception as e:
                print(f"[PDF] ✗ Error generating PDF: {e}")
                import traceback
                traceback.print_exc()
                return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fpdf_irc_report_generator.py <irc_json> [audit_json]")
        sys.exit(1)
    
    irc_path = sys.argv[1]
    audit_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        gen = IRCToPDFGenerator(irc_path, audit_path)
        pdf_file = gen.generate()
        
        if pdf_file:
            print(f"✓ PDF generated: {pdf_file}")
        else:
            print("✗ Failed to generate PDF")
            sys.exit(1)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)