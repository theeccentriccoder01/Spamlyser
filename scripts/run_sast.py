import os
import re
import sys


def run_sast_checks():
    print("==================================================")
    print("🛡️ RUNNING SPAMLYSER SAST SECURITY SCANNER")
    print("==================================================")

    insecure_patterns = {
        r"eval\(": "Insecure usage of eval() detected.",
        r"exec\(": "Insecure usage of exec() detected.",
        r"unsafe_allow_html=True": "Potential XSS vector: Verify inputs are html-escaped.",
    }

    issues_found = 0
    for root, _dirs, files in os.walk("."):
        if any(p in root for p in [".git", "tests", "docs", "venv", ".venv"]):
            continue
        for file in files:
            if not file.endswith(".py"):
                continue
            filepath = os.path.join(root, file)
            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                for pattern, desc in insecure_patterns.items():
                    if re.search(pattern, content):
                        print(f"⚠️ [WARNING] {filepath}: {desc}")
                        issues_found += 1
            except Exception:
                pass

    print("==================================================")
    if issues_found > 0:
        print(f"Scan complete. Found {issues_found} potential security warnings.")
    else:
        print("Scan complete. No security warnings found. Code is secure.")
    print("==================================================")
    sys.exit(0)


if __name__ == "__main__":
    run_sast_checks()
