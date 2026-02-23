import json, subprocess, sys

result = subprocess.run(
    ["gh", "pr", "list", "--state", "open", "--limit", "100",
     "--json", "number,title,headRefName,body,author"],
    capture_output=True, text=True, cwd=sys.argv[1] if len(sys.argv)>1 else "."
)
prs = sorted(json.loads(result.stdout), key=lambda x: x["number"])
for p in prs:
    print(f"PR #{p['number']}: {p['title']}")
    print(f"  Branch: {p['headRefName']}")
    print(f"  Author: {p['author']['login']}")
    print()
