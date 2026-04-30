import json
import os

file_path = r'C:\Users\siddh\.gemini\antigravity\brain\efae1060-aa30-4e5e-803c-5c3570d3dbf8\.system_generated\steps\515\output.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"{'#':<5} {'Branch':<40} {'Title'}")
print("-" * 80)
for pr in data:
    if pr['state'] == 'open':
        print(f"{pr['number']:<5} {pr['head']['ref']:<40} {pr['title']}")
