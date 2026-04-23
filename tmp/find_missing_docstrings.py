
import os
import ast
import re

def find_python_files(start_dir):
    py_files = []
    exclude_dirs = {'venv', 'venv_new', '__pycache__', 'tests', '.git', 'node_modules', '.agent'}
    for root, dirs, files in os.walk(start_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and 'tests' not in d.lower()]
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def check_docstrings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            tree = ast.parse(content)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []

    needs_docs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Ignore some internal functions if they follow double underscore pattern but not dunder
            if node.name.startswith('_') and not node.name.startswith('__'):
                # Still document if complex or if they are just regular private methods
                pass
            
            docstring = ast.get_docstring(node)
            is_class = isinstance(node, ast.ClassDef)
            
            if not docstring:
                needs_docs.append({
                    'type': 'Class' if is_class else 'Function',
                    'name': node.name,
                    'lineno': node.lineno,
                    'file': file_path,
                    'reason': 'Missing'
                })
            elif not is_class: # For functions/methods, check params/returns
                # Check for Args and Returns if it has parameters/return value
                has_params = len(node.args.args) > 0 or node.args.vararg or node.args.kwarg
                # Simple check for Returns
                has_return = any(isinstance(child, ast.Return) and child.value is not None for child in ast.walk(node))
                
                missing_args = has_params and "Args:" not in docstring
                missing_returns = has_return and "Returns:" not in docstring
                
                if missing_args or missing_returns:
                    reasons = []
                    if missing_args: reasons.append("Missing Args")
                    if missing_returns: reasons.append("Missing Returns")
                    needs_docs.append({
                        'type': 'Function',
                        'name': node.name,
                        'lineno': node.lineno,
                        'file': file_path,
                        'reason': ' / '.join(reasons)
                    })
    return needs_docs

if __name__ == "__main__":
    start_dir = os.getcwd()
    files = find_python_files(start_dir)
    all_needs = []
    for f in files:
        needs = check_docstrings(f)
        all_needs.extend(needs)
    
    # Sort by file and then line number
    all_needs.sort(key=lambda x: (x['file'], x['lineno']))
    
    for n in all_needs:
        print(f"{n['file']}:{n['lineno']} - {n['type']} {n['name']} ({n['reason']})")
