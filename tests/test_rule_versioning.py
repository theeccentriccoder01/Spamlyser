import pytest
from models.rule_version_store import RuleVersionStore
from models.custom_rules_manager import (
    save_custom_rules_versioned,
    rollback_custom_rules,
    get_rule_version_store,
    load_custom_rules
)

def test_rule_version_store():
    store = RuleVersionStore(max_history=5)
    r1 = {"allowlist": ["example.com"], "blocklist": ["phish.*"]}
    r2 = {"allowlist": ["example.com", "trusted.org"], "blocklist": ["phish.*", "crypto.*"]}

    v1 = store.commit_version(r1, author="admin", comment="Initial rules")
    v2 = store.commit_version(r2, author="security_team", comment="Added crypto block")

    assert v1 == 1
    assert v2 == 2
    assert len(store.list_history()) == 2

    rolled_back = store.rollback(v1)
    assert rolled_back == r1

def test_custom_rules_versioned_workflow(tmp_path, monkeypatch):
    test_file = str(tmp_path / "custom_rules.json")
    monkeypatch.setattr("models.custom_rules_manager._rules_file_path", lambda: test_file)

    rules1 = {"allowlist": ["safe.com"], "blocklist": ["bad.*"]}
    rules2 = {"allowlist": ["safe.com", "good.org"], "blocklist": ["bad.*", "malware.*"]}

    assert save_custom_rules_versioned(rules1, author="user1", comment="v1") is True
    assert save_custom_rules_versioned(rules2, author="user1", comment="v2") is True

    store = get_rule_version_store()
    history = store.list_history()
    assert len(history) >= 2

    assert rollback_custom_rules(1) is True
    current = load_custom_rules()
    assert current["allowlist"] == ["safe.com"]
