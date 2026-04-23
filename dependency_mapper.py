import ast
import os

def get_imports(filepath):
    with open(filepath, 'r') as f:
        try:
            tree = ast.parse(f.read())
        except Exception:
            return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            if node.level > 0:
                # Handle relative imports (approximate based on current file)
                module = "." * node.level + module
            imports.add(module)
    return imports

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py') and not file.startswith('dependency_mapper'):
            filepath = os.path.join(root, file)
            imports = get_imports(filepath)
            print(f"{filepath}: {imports}")
