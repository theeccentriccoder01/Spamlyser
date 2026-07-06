# Custom Rules

Spamlyser Pro supports user-defined rules that override the ML model predictions.
Rules are evaluated **before** the ensemble models and take priority over them.

## Rule types

### Allowlist (trusted senders / domains)

Plain-text entries. If the entry string appears anywhere in the message (case-insensitive), the message is classified as **HAM** regardless of what the models think.

**Typical uses:**
- Your company's own domain (`my-company.com`)
- A known safe sender's name or email fragment
- An internal system keyword that your models occasionally misclassify

### Blocklist (spam patterns)

Each entry is interpreted as a Python regular expression (with `re.IGNORECASE`).
If the pattern matches anywhere in the message, it is classified as **SPAM**.

**Typical uses:**
- Known scam phrase literals: `click here to claim your prize`
- Regex for phone number patterns used by a spammer: `\+44\s?7\d{9}`
- Keyword with word-boundary anchors: `\bfree-money\b`

> **Tip:** Invalid regex patterns are silently skipped during matching but
> generate a warning in the application log at save time.  Fix them before
> saving for reliable results.

## Evaluation order

```
Message arrives
      │
      ▼
 Check allowlist (order: top to bottom)
      │ match → return HAM immediately
      │
      ▼ no match
 Check blocklist (order: top to bottom)
      │ match → return SPAM immediately
      │
      ▼ no match
 Pass to ML ensemble models
```

## Storage

Rules are stored as JSON in the path configured by `SPAMLYSER_CUSTOM_RULES`
(defaults to `<data_dir>/custom_rules.json`).  Every save creates a timestamped
backup so you can roll back accidental changes.

```json
{
  "allowlist": [
    "my-company.com",
    "internal-alerts"
  ],
  "blocklist": [
    "\\bwin-free-100k\\b",
    "click-now-scam",
    "\\+44\\s?7\\d{9}"
  ]
}
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SPAMLYSER_CUSTOM_RULES` | `<data_dir>/custom_rules.json` | Absolute path to the rules file |
| `SPAMLYSER_MAX_CUSTOM_RULES_PER_LIST` | `500` | Soft cap on entries per list (UI warning only) |

## Running the tests

```bash
pytest tests/test_custom_rules.py -v
```

Tests are fully isolated — they redirect the rules file to a temporary
directory via the `SPAMLYSER_CUSTOM_RULES` environment variable so production
data is never touched.
