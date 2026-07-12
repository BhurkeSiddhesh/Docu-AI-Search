import json

def get_priority(labels):
    label_names = [l['name'] for l in labels]
    if 'Critical Bug' in label_names:
        return 1
    if 'Logic Enhancement' in label_names:
        return 2
    if 'Developer Experience' in label_names:
        return 3
    return 4

def has_auto_merge_ok(labels):
    return 'auto-merge-ok' in [l['name'] for l in labels]

with open('issues_utf8.json', 'r', encoding='utf-8-sig') as f:
    issues = json.load(f)

import subprocess

def has_open_pr(issue_number):
    try:
        result = subprocess.run(['gh', 'pr', 'list', '--repo', 'BhurkeSiddhesh/Docu-AI-Search', '--search', f'closes #{issue_number}', '--json', 'number'], capture_output=True, text=True, check=True)
        prs = json.loads(result.stdout)
        return len(prs) > 0
    except Exception as e:
        return False

valid_issues = []
for issue in issues:
    labels = issue.get('labels', [])
    label_names = [l['name'] for l in labels]
    
    if 'needs-human-review' in label_names or 'wont-fix' in label_names:
        continue
    
    body = issue.get('body', '')
    if '<!-- skip-autofix -->' in body:
        continue
        
    prio = get_priority(labels)
    if prio > 3:
        continue
        
    if has_open_pr(issue['number']):
        continue
        
    valid_issues.append({
        'number': issue['number'],
        'title': issue['title'],
        'prio': prio,
        'auto_merge': has_auto_merge_ok(labels),
        'body': body
    })

# Sort: priority (lower is better), then auto_merge (True is better), then number (lower is better or higher is better - let's do higher is better / more recent)
valid_issues.sort(key=lambda x: (x['prio'], not x['auto_merge'], -x['number']))

with open('triage_output.txt', 'w', encoding='utf-8') as f_out:
    for issue in valid_issues[:20]:
        f_out.write(f"#{issue['number']} - P{issue['prio']} - AutoMerge:{issue['auto_merge']} - {issue['title']}\n")
