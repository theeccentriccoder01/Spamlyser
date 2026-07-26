class PDFReportGenerator:
    def __init__(self, data):
        self.data = data
    def build_report(self):
        return f"Encrypted PDF Report Content: {self.data}"
