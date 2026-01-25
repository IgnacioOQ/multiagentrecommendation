import os
import glob
import sys

# Ensure we can import md_parser
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from md_parser import MarkdownParser

def scan_content_for_type(filename):
    """Peek at file content to determine type."""
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(50)]
        content = "".join(lines)
        
        # Agent Skill heuristics
        if "Role:" in content and "Agent" in content:
            return 'agent_skill'
        if "Goal:" in content and "Provision" in content:
            return 'agent_skill'
            
        # Protocol heuristics
        if "Protocol" in content and ("# " in content or "## " in content):
            return 'protocol'
            
        # Context heuristics
        if "Background:" in content or "Context:" in content:
            return 'context'

    except Exception:
        pass
    return None

def get_file_type(filename, path):
    """Determine type based on filename patterns and content."""
    basename = os.path.basename(filename)
    
    # Guideline
    if basename in ['README.md', 'MD_CONVENTIONS.md', 'AGENTS.md']:
        return 'guideline'
    if 'DEFINITIONS' in basename or 'REQUIREMENTS' in basename:
        return 'guideline'
    
    # Log
    if 'LOG' in basename:
        return 'log'
        
    # Recurring
    if 'HOUSEKEEPING' in basename:
        return 'recurring'
        
    # Agent Skill
    if 'AGENT' in basename or 'ASSISTANT' in basename:
        if 'AGENTS.md' not in basename and 'LOG' not in basename: # Exclude AGENTS.md and logs
            return 'agent_skill'
            
    # Plan
    if 'PLAN' in basename.upper() or 'SETUP' in basename.upper():
        return 'plan'
        
    # Protocol
    if 'PROTOCOL' in basename:
        return 'protocol'
        
    # Context
    if basename in ['DAG_Example.md', 'HYPOTHESIS.md']:
        return 'context'
    
    # Fallback: Content Scan
    content_type = scan_content_for_type(path)
    if content_type:
        return content_type
        
    return None

def apply_types_to_tree(node, root_type):
    """Recursively apply types to nodes."""
    # Apply root type to root node (or level 1 header if root is synthetic)
    # The parser returns a synthetic root level 0.
    # The actual "Document Title" is usually level 1.
    
    # If node is level 1 (Document Title), set the main type
    if node.level == 1:
        if 'type' not in node.metadata:
            node.metadata['type'] = root_type
            print(f"  Set root type: {root_type}")
            
    # If root type is plan or recurring, sub-tasks should be 'task'
    # Check if we should propagate 'task' type to children
    if root_type in ['plan', 'recurring']:
        # If this node is level >= 1, its children are tasks
        if node.level >= 1:
            for child in node.children:
                if 'type' not in child.metadata:
                    child.metadata['type'] = 'task'
                    # Recursively set children of tasks to tasks as well
                    apply_types_to_tree(child, root_type)
        else:
            # Synthetic root level 0 -> recurse to children
            for child in node.children:
                apply_types_to_tree(child, root_type)
    else:
        # Just recurse without forcing 'task'
        for child in node.children:
             apply_types_to_tree(child, root_type)

def main():
    parser = MarkdownParser()
    files = glob.glob('**/*.md', recursive=True)
    
    count = 0
    for f in files:
        if 'node_modules' in f or '.git' in f or 'brain' in f:
            continue
            
        ftype = get_file_type(f, f)
        if not ftype:
            print(f"Skipping {f} (no type mapping)")
            continue
            
        try:
            print(f"Processing {f} as {ftype}...")
            root = parser.parse_file(f)
            
            apply_types_to_tree(root, ftype)
            
            new_content = root.to_markdown()
            with open(f, 'w') as write_f:
                write_f.write(new_content)
                if not new_content.endswith('\n'):
                    write_f.write('\n')
            count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    print(f"Updated {count} files.")

if __name__ == "__main__":
    main()
