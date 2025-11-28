"""
Quick script to open the report in Chrome (bypasses Windows file associations)
"""

import os
import subprocess
from pathlib import Path

# Find Chrome
chrome_paths = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'Application', 'chrome.exe'),
]

report_path = Path(__file__).parent / 'reports' / 'full_report.html'

if not report_path.exists():
    print(f"❌ Report not found: {report_path}")
    exit(1)

print(f"\nOpening report in Chrome...\n{report_path}\n")

# Try Chrome
for chrome in chrome_paths:
    if os.path.exists(chrome):
        subprocess.Popen([chrome, str(report_path.absolute())])
        print(f"✓ Opened in Chrome!")
        exit(0)

# Fallback to Edge
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if os.path.exists(edge_path):
    subprocess.Popen([edge_path, str(report_path.absolute())])
    print(f"✓ Opened in Edge (Chrome not found)")
    exit(0)

# Last resort
print("⚠ Chrome/Edge not found. Opening with default application...")
os.startfile(str(report_path.absolute()))



