import json
import sys
from pathlib import Path

file_path = Path(sys.argv[1])
with file_path.open(encoding='utf-8') as f:
    data = json.load(f)

print(f"{'#':<5} {'Branch':<40} {'Title'}")
print("-" * 80)
for pr in data:
    if pr['state'] == 'open':
        print(f"{pr['number']:<5} {pr['head']['ref']:<40} {pr['title']}")
