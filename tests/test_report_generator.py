from models.report_generator import PDFReportGenerator

def test_pdf_report_generator():
    gen = PDFReportGenerator("Threat Trends")
    assert "Threat Trends" in gen.build_report()
