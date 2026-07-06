# Explainability System

Spamlyser Pro provides two explainability backends to help users understand why a message was classified as spam or ham:

## 1. ModelExplainer (LIME Backend)
This backend uses **LIME** (Local Interpretable Model-agnostic Explanations) to perturb the input text and run predictions against the active ensemble models. It fits a local surrogate model to estimate the feature importance of individual words.

- **Requirements**: `lime` package.
- **Output**: Visualized HTML overlays showing positive (spammy) and negative (hammy) word weights.

## 2. SimpleExplainer (Keyword Backend)
A lightweight, dependency-free backend that acts as a fallback when `lime` is not installed or when low latency is required.

- **Mechanism**: Scans text for pre-defined lists of `SPAM_KEYWORDS` and `HAM_KEYWORDS`.
- **Weights**:
  - Spam keywords match with positive weights (e.g. `+0.5` per occurrence).
  - Ham keywords match with negative weights (e.g. `-0.5` per occurrence).

## Threat Specific Explanations
Both explainers support threat-specific evaluations using `get_threat_explanation(text, threat_type)` which focuses weights only on the words indicating the identified threat class (e.g., Phishing or Scam).

## Configuration and fallback
If LIME cannot be imported:
1. The app gracefully falls back to `SimpleExplainer`.
2. UI displays clean keyword matches instead of LIME's perturbed local approximation.

## Running Tests
Run tests for both backends:
```bash
pytest tests/test_simple_explainer.py -v
```
