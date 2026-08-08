# Security Policy

## Supported Versions

We actively patch security vulnerabilities for the following versions of
Spamlyser Pro. If you are running an older release, please upgrade before
reporting.

| Version   | Supported          |
| --------- | ------------------ |
| latest    | :white_check_mark: |
| < latest  | :x:                |

## Reporting a Vulnerability

> **Please do NOT open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability, we appreciate your help in
disclosing it responsibly. Please follow these steps:

1. **Email**: Send a detailed report to the project maintainer at
   **[theeccentriccoder01](https://github.com/theeccentriccoder01)** via
   GitHub private messaging or the email listed in their profile.
2. **GitHub Security Advisories**: Alternatively, use the
   [GitHub Security Advisory](https://github.com/theeccentriccoder01/Spamlyser/security/advisories/new)
   feature to privately report the vulnerability.
3. **Include**:
   - A description of the vulnerability and its potential impact.
   - Detailed steps to reproduce the issue (proof of concept appreciated).
   - The affected component(s) — e.g., `models/`, `app.py`, webhook
     handler, CSV export, etc.
   - Your suggested fix, if any.

### What to Expect

| Stage                        | Timeline        |
| ---------------------------- | --------------- |
| Acknowledgement of report    | Within 48 hours |
| Initial assessment & triage  | Within 5 days   |
| Fix development & testing    | Within 14 days  |
| Patch release & disclosure   | Within 30 days  |

We will credit reporters in the release notes (unless you prefer to remain
anonymous).

## Security Best Practices for Deployment

When deploying Spamlyser Pro in production, we recommend:

### Environment & Secrets

- **Never commit `.env` files** — use environment variables or a secrets
  manager.
- Rotate the `SPAMLYSER_WEBHOOK_CONFIG` credentials regularly.
- Set `SPAMLYSER_ERROR_DETAIL=false` in production to avoid leaking
  stack traces.

### Network & Access

- Run the Streamlit server behind a reverse proxy (e.g., Nginx, Caddy)
  with TLS termination.
- Restrict webhook URLs to trusted external endpoints — the app does not
  currently block internal/loopback addresses.
- Enable `SPAMLYSER_CSV_SANITIZE_FORMULAS=true` (default) to prevent
  CSV formula injection (CWE-1236).

### Container Security

- Use the provided `Dockerfile` which runs as a non-root user.
- Keep the base image (`python:3.13-slim`) updated to receive OS-level
  security patches.
- Scan images with `docker scout` or `trivy` before deploying.

### Dependency Management

- The CI pipeline runs weekly `pip-audit` scans
  (`.github/workflows/security-audit.yml`).
- Enable Dependabot alerts and auto-merge for patch-level updates.
- Review `bandit` output for static analysis findings.

## Scope

The following areas are in scope for security reports:

- **Injection**: XSS via `unsafe_allow_html`, SQL injection in SQLite
  stores, CSV formula injection.
- **ReDoS**: Catastrophic backtracking in user-supplied regex rules.
- **SSRF**: Webhook URL validation bypass targeting internal services.
- **Authentication/Authorization**: Any bypass of rate limiting or
  access controls.
- **Data Exposure**: Unintended leakage of feedback data, model weights,
  or configuration secrets.
- **Dependency Vulnerabilities**: Known CVEs in pinned dependencies.

## Out of Scope

- Denial-of-service attacks against the hosted demo on Hugging Face
  Spaces (infrastructure managed by HF).
- Social engineering attacks.
- Issues in third-party dependencies that have already been reported
  upstream.

---

Thank you for helping keep Spamlyser Pro and its users safe! 🛡️
