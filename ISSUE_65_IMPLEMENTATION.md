# Issue #65 - Unify Test Imports and Infrastructure

## Summary

Standardized test infrastructure across all test files with unified import patterns and centralized pytest configuration.

## Files Added/Modified

### NEW FILES
1. **tests/__init__.py** - Make tests a proper Python package
2. **pyproject.toml** - Centralized pytest configuration

### MODIFIED FILES
- Test files: Import paths now consistent

---

## Key Changes

### 1. tests/__init__.py
Makes the tests directory a proper Python package, enabling:
- Proper import of test modules
- Shared fixtures across all tests
- Better IDE support

### 2. pyproject.toml
Centralized pytest configuration with:
- Test discovery rules
- Coverage settings
- Marker definitions
- Tool configurations (black, ruff, mypy)

### 3. Unified Import Patterns

**Before (Fragile):**
```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from threat_analyzer import analyze_threat
```

**After (Clean):**
```python
from models.threat_analyzer import analyze_threat
```

---

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_threat_analyzer.py -v
```

### Run with coverage:
```bash
pytest --cov=models --cov=config tests/
```

### Run specific marker:
```bash
pytest -m unit
```

---

## Test Organization

```
tests/
├── __init__.py                    # Package init
├── conftest.py                    # Shared fixtures
├── test_batch_processor.py        # Batch processing tests
├── test_config.py                 # Config module tests
├── test_ensemble_get_model_prediction.py
├── test_export_feature.py
├── test_page_functions.py
├── test_risk_indicators.py
├── test_threat_analyzer.py
└── test_word_analyzer_is_hammy.py
```

---

## Configuration in pyproject.toml

### Pytest Settings
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --strict-markers --tb=short"
```

### Markers
```toml
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests",
]
```

### Coverage
```toml
[tool.coverage.run]
source = ["models", "config"]
branch = true
```

---

## Best Practices

1. **Imports** - Always use `from models.module import ...`
2. **Conftest** - Put shared fixtures in `tests/conftest.py`
3. **Markers** - Use `@pytest.mark.unit` for test categorization
4. **Naming** - Test files as `test_*.py`, functions as `test_*`

---

**Issue #65 Complete!** ✅
