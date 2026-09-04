import json
import os
import re

with open('ops/all_findings_detailed.json', 'r', encoding='utf-8') as f:
    findings = json.load(f)

print(f"Total findings to evaluate: {len(findings)}")

# Check git diff for files touched since 899d0e8
import subprocess
git_changed_files = subprocess.check_output(
    ["git", "diff", "--name-only", "899d0e8..HEAD"],
    text=True
).splitlines()
git_changed_set = set(git_changed_files)

evaluations = []

for f in findings:
    fid = f['id']
    severity = f['severity'].lower()
    category = f['category'].lower()
    title = f['title']
    location = f['location']
    body = f['text']
    
    parts = location.split(':')
    filepath = parts[0].strip()
    line_str = parts[1].strip() if len(parts) > 1 else None
    line_num = int(line_str) if line_str and line_str.isdigit() else None
    
    file_exists = os.path.exists(filepath)
    is_modified_in_recent_commits = filepath in git_changed_set
    
    evaluations.append({
        'id': fid,
        'severity': severity,
        'category': category,
        'title': title,
        'location': location,
        'filepath': filepath,
        'line_num': line_num,
        'file_exists': file_exists,
        'modified_since_audit': is_modified_in_recent_commits,
    })

with open('ops/audit_evaluations_prelim.json', 'w', encoding='utf-8') as out:
    json.dump(evaluations, out, indent=2)

print(f"Evaluations written. Modified files count: {len(git_changed_set)}")
