# Changelog

All notable changes to **Spamlyser Pro** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SECURITY.md` with responsible disclosure policy and deployment best
  practices.
- `CHANGELOG.md` following Keep a Changelog format.
- `.dockerignore` to reduce Docker build context and image size.
- Comprehensive `ModelDriftTracker` with windowed history, severity
  classification, and trend detection.
- Expanded test suite for `ModelDriftTracker` covering edge cases,
  severity thresholds, and trend analysis.

### Fixed

- Removed duplicate mock setup blocks in `tests/conftest.py`.
- Fixed corrupted null bytes at end of `.gitignore`.

### Changed

- Improved `Dockerfile` with non-root user, health check, and OCI labels.
- Expanded `.gitignore` with coverage, distribution, cache, and log
  patterns.

---

## [1.0.0] — 2025-01-01

> Baseline release capturing the existing feature set.

### Added

- **Ensemble SMS Spam Classifier** combining DistilBERT, BERT, RoBERTa,
  and ALBERT with 5 ensemble strategies (Majority Voting, Weighted Average,
  Confidence-Weighted, Adaptive Threshold, Meta-Ensemble).
- **Single-message and batch CSV analysis** with real-time progress
  tracking and ETA.
- **Threat analyzer** with phishing, smishing, and social engineering
  detection across 10+ threat categories.
- **Model explainability** via LIME feature attribution and a lightweight
  word-level highlighter (`SimpleExplainer`).
- **Custom rules engine** supporting allowlist, blocklist (keyword & regex),
  and compound AND/OR/NOT rules.
- **Confidence calibration** using Temperature Scaling and Platt Scaling
  with ECE metrics and reliability diagrams.
- **Sender reputation tracking** with historical spam ratio scoring.
- **Export system** supporting PDF (encrypted), CSV (sanitized against
  CWE-1236), and JSON formats.
- **Webhook notifications** with retry queue for real-time spam alerts.
- **Analytics dashboards**: session analytics, anomaly detection,
  benchmark runner, and trend analytics.
- **What-If Playground** for interactive sensitivity testing.
- **Theme customizer** with dark/light mode and accessibility enhancements.
- **Language detection** for multilingual SMS input routing.
- **Streamlit multi-page app** with 7 dedicated page modules.
- **CI/CD pipeline**: GitHub Actions for testing (Python 3.11–3.13),
  linting (Ruff + codespell), security auditing (pip-audit + bandit),
  and auto-deployment to Hugging Face Spaces.
- **Docker support** with `Dockerfile` for containerized deployment.
- **Developer tooling**: pre-commit hooks, Ruff formatter, EditorConfig,
  Prettier for non-Python files.
- **Documentation**: `ARCHITECTURE.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `MODEL_CARD.md`, and 5 docs under `docs/`.
