# Issue #61 - Centralized Configuration System with Environment Support

## Overview
This fix implements a centralized configuration management system for Spamlyser Pro, eliminating hardcoded paths and values throughout the codebase.

## Files Added/Modified

### NEW FILES
1. **config.py** - Central configuration module
2. **.env.example** - Environment variables template
3. **.streamlit/secrets.toml.example** - Streamlit secrets template
4. **tests/test_config.py** - 8 unit tests for config module

### FILES TO MODIFY
1. **models/model_init.py** - Use config for model paths
2. **models/feedback_handler.py** - Use config for feedback paths
3. **.gitignore** - Add .env and secrets patterns
4. **models/__init__.py** - Add __all__ export

---

## Step-by-Step Implementation

### Step 1: Copy New Files

```powershell
# Copy config module
Copy-Item config.py C:\Elusoc\Spamlyser\config.py

# Copy env template
Copy-Item .env.example C:\Elusoc\Spamlyser\.env.example

# Copy streamlit secrets template
Copy-Item .streamlit_secrets.toml.example C:\Elusoc\Spamlyser\.streamlit\secrets.toml.example

# Copy tests
Copy-Item test_config.py C:\Elusoc\Spamlyser\tests\test_config.py
```

### Step 2: Update .gitignore

Add these lines to `.gitignore`:

```
# Environment variables
.env
.env.local
.env.*.local

# Streamlit secrets
.streamlit/secrets.toml

# Cache directories
.cache/
__pycache__/
*.pyc

# IDE
.vscode/
.idea/
*.swp
```

### Step 3: Update models/model_init.py

**OLD (Hardcoded):**
```python
model_name = "distilbert-base-uncased"
cache_dir = Path.home() / ".cache" / "huggingface"
```

**NEW (Using Config):**
```python
from config import Config

model_name = Config.get_verification_model_name()
cache_dir = Config.ensure_model_cache_dir()
```

**Complete updated section:**
```python
"""
Model initialization and verification module

This module handles the initialization and verification of ML models with
comprehensive error handling to prevent app crashes when models are missing.
"""

from pathlib import Path
from typing import Dict, Tuple

from config import Config  # ADD THIS IMPORT

# ... rest of file ...

def verify_model_availability() -> Tuple[bool, str, list]:
    """
    Verify that required ML frameworks and models are available.

    Returns:
        tuple: (success: bool, error_message: str, warnings: list)
    """
    warnings = []

    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append(
                "⚠️ CUDA not available. Using CPU with dynamic quantization (int8) for balanced performance."
            )
    except ImportError as e:
        return False, f"❌ PyTorch not installed: {str(e)}", warnings

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # USE CONFIG HERE
        model_name = Config.get_verification_model_name()
        cache_dir = Config.ensure_model_cache_dir()
        
        # Test load with cache directory from config
        AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        if Config.get_use_quantization():
            warnings.append("✅ Dynamic quantization enabled for optimized performance")
        
        return True, "", warnings
        
    except Exception as e:
        return False, f"❌ Model loading failed: {str(e)}", warnings
```

### Step 4: Update models/feedback_handler.py

**OLD (Hardcoded):**
```python
def __init__(self, feedback_file: str = "feedback_data.json"):
    self.db_path = os.path.splitext(feedback_file)[0] + ".db"
```

**NEW (Using Config):**
```python
from config import Config

class FeedbackHandler:
    """Handles user feedback operations using SQLite for concurrent write safety."""

    def __init__(self, feedback_file: str = None):
        if feedback_file is None:
            feedback_file = str(Config.get_feedback_json_path())
        
        self.db_path = Config.get_feedback_db_path()
        Config.ensure_feedback_dir()  # Create directory if needed
        self._init_db()
        self._migrate_from_json(feedback_file)
```

### Step 5: Update models/__init__.py

Add this to the file:

```python
"""
Models module for Spamlyser Pro.

Exports all model-related classes and functions.
"""

from models.ensemble_classifier_method import EnsembleSpamClassifier
from models.feedback_handler import FeedbackHandler
from models.threat_analyzer import ThreatAnalyzer
from models.word_analyzer import WordAnalyzer

__all__ = [
    "EnsembleSpamClassifier",
    "FeedbackHandler",
    "ThreatAnalyzer",
    "WordAnalyzer",
]
```

### Step 6: Create .env File (Local Development)

Copy the template and fill in values:

```bash
cp .env.example .env
```

Then edit `.env`:

```
VERIFICATION_MODEL_NAME=distilbert-base-uncased
MODEL_CACHE_DIR=~/.cache/huggingface
FEEDBACK_DB_PATH=feedback_data.db
DEBUG_MODE=true
LOG_LEVEL=debug
FEATURE_WORD_ANALYSIS=true
FEATURE_THREAT_ANALYSIS=true
FEATURE_PDF_EXPORT=true
```

### Step 7: Install python-dotenv (if not already installed)

```bash
pip install python-dotenv
```

Or add to requirements.txt:
```
python-dotenv>=0.19.0
```

### Step 8: Run Tests

```bash
# Run config tests
pytest tests/test_config.py -v

# Run all tests
pytest tests/ -v
```

---

## Usage in Code

### In Models

```python
from config import Config

# Get configuration values
cache_dir = Config.ensure_model_cache_dir()
batch_size = Config.get_batch_size()
use_quantization = Config.get_use_quantization()
```

### In Streamlit App

```python
import streamlit as st
from config import Config

# Get app configuration
st.set_page_config(page_title=Config.get_app_name())

# Get data paths
feedback_db = Config.get_feedback_db_path()
csv_dir = Config.get_csv_dir()

# Check feature flags
if Config.is_feature_enabled("pdf_export"):
    # Show PDF export button
    pass
```

### In Scripts

```python
from config import Config

# Validate configuration on startup
is_valid, errors, warnings = Config.validate_configuration()

if not is_valid:
    print("Configuration errors found:")
    for error in errors:
        print(f"  - {error}")
    exit(1)

# Print configuration summary
Config.print_configuration_summary()
```

---

## Environment Variables Reference

### Essential
- `VERIFICATION_MODEL_NAME` - Model to use (default: distilbert-base-uncased)
- `MODEL_CACHE_DIR` - Where to cache models (default: ~/.cache/huggingface)
- `FEEDBACK_DB_PATH` - Feedback database path (default: feedback_data.db)

### Optional
- `HF_TOKEN` - HuggingFace API token
- `GITHUB_TOKEN` - GitHub API token
- `DEBUG_MODE` - Enable debug logging (default: false)
- `LOG_LEVEL` - Logging level (default: info)

### Feature Flags
- `FEATURE_WORD_ANALYSIS` - Enable word analysis (default: true)
- `FEATURE_THREAT_ANALYSIS` - Enable threat analysis (default: true)
- `FEATURE_FEEDBACK_SYSTEM` - Enable feedback (default: true)
- `FEATURE_PDF_EXPORT` - Enable PDF export (default: true)

See `.env.example` for full list!

---

## Testing

### Run All Config Tests
```bash
pytest tests/test_config.py -v
```

### Run Specific Test
```bash
pytest tests/test_config.py::TestModelConfiguration::test_get_verification_model_name_default -v
```

### Test Coverage
```bash
pytest tests/test_config.py --cov=config
```

---

## Benefits

✅ **No more hardcoded paths** - All configuration via environment variables
✅ **Easier deployment** - Configure for different environments
✅ **Better testing** - Mock environment variables in tests
✅ **Single source of truth** - All config in one place
✅ **Type-safe** - Static methods with clear return types
✅ **Well-documented** - Each setting has docstring with default
✅ **Extensible** - Easy to add new configuration options
✅ **Validated** - `validate_configuration()` checks for errors

---

## Migration Checklist

- [ ] Copy `config.py` to project root
- [ ] Copy `.env.example` to project root
- [ ] Create `.env` file (copy from example, fill in values)
- [ ] Copy `.streamlit/secrets.toml.example` to `.streamlit/`
- [ ] Update `.gitignore` with .env patterns
- [ ] Update `models/model_init.py` to use Config
- [ ] Update `models/feedback_handler.py` to use Config
- [ ] Update `models/__init__.py` with __all__
- [ ] Copy `tests/test_config.py` to tests folder
- [ ] Run: `pip install python-dotenv`
- [ ] Run: `pytest tests/test_config.py -v`
- [ ] Commit all changes
- [ ] Create PR

---

## Commit Message Template

```
feat: implement centralized configuration system (#61)

- Add config.py module with environment variable support
- Add .env.example and .streamlit/secrets.toml.example templates
- Update models/model_init.py to use Config class
- Update models/feedback_handler.py to use Config class
- Add 8 comprehensive unit tests for config module
- Update .gitignore for .env and secrets patterns
- Add __all__ to models/__init__.py

Benefits:
- Eliminates hardcoded paths throughout codebase
- Enables easy configuration for different environments
- Improves testing with mockable configuration
- Provides validated centralized configuration
```

---

## Troubleshooting

### ModuleNotFoundError: No module named 'dotenv'
```bash
pip install python-dotenv
```

### Configuration errors during startup
Run:
```bash
python -c "from config import Config; Config.print_configuration_summary()"
```

### Model cache directory not writable
Check permissions:
```bash
ls -la ~/.cache/huggingface
chmod 755 ~/.cache/huggingface
```

### .env file not loading
Ensure it's in the project root:
```bash
ls -la .env
```

---

**Issue #61 is now complete!** ✅
