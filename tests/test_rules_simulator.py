from models.rules_simulator import ThreatRulesSimulator

def test_threat_rules_simulator():
    sim = ThreatRulesSimulator("win free cash")
    results = sim.simulate(["click here to win free cash!", "normal sms message"])
    assert results == [True, False]
