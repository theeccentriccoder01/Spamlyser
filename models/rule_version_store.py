"""
Versioned Custom Rules Store with snapshot tracking, rollback capability, and metadata audit.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class RuleVersionStore:
    """
    Manages historical snapshots of custom rules to enable audit trails and instant rollback.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []

    def commit_version(self, rules: Dict[str, List[Any]], author: str = "system", comment: str = "Rule update") -> int:
        """
        Record a new snapshot of custom rules and return the version ID.
        """
        version_id = len(self._history) + 1
        snapshot = {
            "version_id": version_id,
            "timestamp": time.time(),
            "author": author,
            "comment": comment,
            "rules": json.loads(json.dumps(rules))
        }
        self._history.append(snapshot)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        logger.info(f"Committed rule version {version_id} by {author}")
        return version_id

    def get_version(self, version_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific historical version snapshot.
        """
        for item in self._history:
            if item["version_id"] == version_id:
                return item
        return None

    def list_history(self) -> List[Dict[str, Any]]:
        """
        List all version metadata (excluding full rule payload).
        """
        return [
            {
                "version_id": item["version_id"],
                "timestamp": item["timestamp"],
                "author": item["author"],
                "comment": item["comment"]
            }
            for item in self._history
        ]

    def rollback(self, version_id: int) -> Optional[Dict[str, List[Any]]]:
        """
        Rollback rules to a specified version ID.
        """
        target = self.get_version(version_id)
        if target:
            logger.info(f"Rolled back to rule version {version_id}")
            return json.loads(json.dumps(target["rules"]))
        return None
