# Issue #63 - Pre-commit Config and Code Quality Automation

## Summary

Comprehensive code quality automation with pre-commit hooks, unified editor settings, and enhanced CI workflows.

## Files Added/Modified

### NEW FILES
1. **.pre-commit-config.yaml** - Pre-commit hooks configuration
2. **.editorconfig** - Editor settings for consistency
3. **.gitattributes** - Git line ending settings
4. **requirements-dev.txt** - Development dependencies

### MODIFIED FILES
1. **ruff.toml** - Enhanced with additional rule sets

---

## Setup Instructions

### 1. Install pre-commit
```bash
pip install pre-commit
```

Or add to requirements-dev.txt and install:
```bash
pip install -r requirements-dev.txt
```

### 2. Install git hooks
```bash
pre-commit install
```

### 3. Run on all files (first time)
```bash
pre-commit run --all-files
```

### 4. Run on staged files (automatic on commit)
```bash
git commit -m "your message"
```

---

## Pre-commit Hooks Included

### General Checks
- ✅ Trim trailing whitespace
- ✅ Fix end-of-file newlines
- ✅ Check YAML/JSON/TOML validity
- ✅ Detect private keys
- ✅ Check for merge conflicts
- ✅ Detect large files (>1MB)

### Code Formatting
- ✅ **Black** - Python code formatter (line length: 100)
- ✅ **isort** - Import sorter (Black-compatible)
- ✅ **Ruff** - Fast Python linter with auto-fix

### Type Checking
- ✅ **mypy** - Static type checker (excluding tests)

### Security & Quality
- ✅ **bandit** - Security vulnerability scanner
- ✅ **codespell** - Spell checker for code

### Additional
- ✅ YAML formatting
- ✅ Line ending normalization

---

## Editor Configuration

### .editorconfig
Sets consistent editor settings across IDEs:
- Character encoding: UTF-8
- Line endings: LF (Unix)
- Python indent: 4 spaces
- YAML indent: 2 spaces
- JSON indent: 2 spaces
- Max line length: 100

**Supported editors:**
- VS Code (with EditorConfig extension)
- PyCharm
- Vim (with editorconfig-vim)
- Sublime Text
- And 50+ others

### .gitattributes
Ensures consistent line endings in git:
- Python/text files → LF
- Batch files → CRLF
- Binary files marked as binary
- Images handled appropriately

---

## Enhanced Ruff Configuration

### Rule Sets Enabled
```toml
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort (import sorting)
    "C",      # flake8-comprehensions
    "B",      # flake8-bugbear
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "LOG",    # flake8-logging
]
```

### Per-File Ignores
```toml
[lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports
"tests/*" = ["F811"]       # Allow test redefinition
```

---

## Development Workflow

### New Developer Setup
```bash
git clone https://github.com/theeccentriccoder01/Spamlyser.git
cd Spamlyser
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

### Committing Code
```bash
# Hooks run automatically before commit
git add .
git commit -m "feat: your feature"

# If hooks fail, fix issues and try again
git add .
git commit -m "feat: your feature"
```

### Skipping Hooks (Not Recommended)
```bash
git commit --no-verify
```

### Running Manually
```bash
# Check all files
pre-commit run --all-files

# Check specific hook
pre-commit run black --all-files
```

---

## CI/CD Integration

Pre-commit hooks also run in CI pipelines to ensure:
- No formatting issues
- No security vulnerabilities
- No spelling errors
- Consistent code style

Configuration in `.pre-commit-config.yaml`:
```yaml
ci:
  autofix_commit_msg: 'chore: auto-format with pre-commit hooks'
  autofix_prs: true
  autoupdate_schedule: weekly
```

---

## Benefits

✅ **Consistency** - Same formatting across all files
✅ **Early Detection** - Catch issues before they're committed
✅ **Less CI Failures** - Most issues fixed before pushing
✅ **Better Code** - Automatic security, quality, spell checks
✅ **Team Alignment** - Everyone uses same tools and config

---

## Troubleshooting

### Hook fails but I don't know why
```bash
pre-commit run --all-files --verbose
```

### Skip hook for specific commit
```bash
git commit --no-verify
```

### Update hooks to latest versions
```bash
pre-commit autoupdate
```

### Remove pre-commit hooks
```bash
pre-commit uninstall
```

---

**Issue #63 Complete!** ✅
