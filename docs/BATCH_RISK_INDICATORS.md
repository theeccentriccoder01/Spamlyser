# Batch Risk Indicators

Spamlyser's batch processor adds lightweight risk indicators to every analysed
message so CSV, JSON, and PDF exports can highlight why a message may deserve
closer review.

## Current signals

| Indicator | Trigger |
|---|---|
| `urls` | URL schemes or common web domains such as `.com`, `.net`, or `.org` |
| `urgency` | Pressure language such as `urgent`, `act now`, or `limited time` |
| `money` | Prize, cash, currency, and reward language |
| `personal_info` | Requests for passwords, accounts, logins, SSNs, or credit cards |
| `all_caps` | Any original-cased token longer than two characters written in uppercase |
| `suspicious_chars` | More than 10 percent non-alphanumeric, non-whitespace characters |

## Implementation note

Keyword-style checks are case-insensitive because the message is lowercased
before matching. `all_caps` intentionally checks the original message text
instead, because lowercasing first would erase the very signal it is meant to
detect.

## Test coverage

The regression coverage lives in `tests/test_risk_indicators.py` and exercises
empty messages, whitespace-only rows, suspicious punctuation, clean text, and
uppercase pressure language.
