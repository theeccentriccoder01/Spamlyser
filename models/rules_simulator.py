import re

class ThreatRulesSimulator:
    def __init__(self, rule_pattern: str):
        self.pattern = re.compile(rule_pattern, re.IGNORECASE)
        
    def simulate(self, mock_texts: list[str]) -> list[bool]:
        return [bool(self.pattern.search(text)) for text in mock_texts]
