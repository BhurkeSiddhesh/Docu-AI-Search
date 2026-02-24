import sys
import os

filepath = 'backend/api.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
inserted = False
for line in lines:
    new_lines.append(line)
    if 'file_path = os.path.normpath(file_path)' in line and not inserted:
        indent = line[:line.find('file_path')]
        new_lines.append(f'\n{indent}# Security: Prevent argument injection (files starting with -)\n')
        new_lines.append(f'{indent}if os.path.basename(file_path).startswith("-"):\n')
        new_lines.append(f'{indent}    logger.warning(f"Security: Blocked attempt to open file with leading dash: {{file_path}}")\n')
        new_lines.append(f'{indent}    raise HTTPException(status_code=400, detail="Invalid filename: Files starting with \'-\' are not allowed.")\n')
        inserted = True

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print("Fix applied successfully.")
