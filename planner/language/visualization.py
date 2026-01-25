import sys
import os
import argparse
# Add the current directory to sys.path to allow importing md_parser if run from scripts/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from md_parser import MarkdownParser
import cli_utils
from cli_utils import add_standard_arguments, validate_and_get_pairs

class TreeVisualizer:
    def __init__(self):
        pass

    def visualize(self, node, prefix="", is_last=True, is_root=True):
        if not is_root:
            # Prepare the connector
            connector = "└── " if is_last else "├── "
            
            # Prepare metadata string (e.g., [status: done])
            meta_str = ""
            if node.metadata:
                # Prioritize specific fields for display
                status = node.metadata.get('status')
                type_ = node.metadata.get('type')
                estimate = node.metadata.get('estimate')
                owner = node.metadata.get('owner')
                blocked_by = node.metadata.get('blocked_by')
                node_id = node.metadata.get('id')
                
                parts = []
                if node_id:
                    parts.append(f"id: {node_id}")
                if status:
                    parts.append(f"status: {status}")
                if type_:
                    parts.append(f"type: {type_}")
                if estimate:
                    parts.append(f"est: {estimate}")
                if owner:
                    parts.append(f"@{owner}")
                if blocked_by:
                    if isinstance(blocked_by, list):
                        blocked_str = ",".join(blocked_by)
                    else:
                        blocked_str = str(blocked_by)
                    parts.append(f"blocked: [{blocked_str}]")
                
                if parts:
                    meta_str = f"  [{', '.join(parts)}]"

            print(f"{prefix}{connector}{node.title}{meta_str}")
        
        # Prepare prefix for children
        if is_root:
            child_prefix = ""
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Recurse for children
        count = len(node.children)
        for i, child in enumerate(node.children):
            is_last_child = (i == count - 1)
            self.visualize(child, child_prefix, is_last_child, is_root=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Markdown-JSON tree.")
    add_standard_arguments(parser, multi_file=False)
    parser.add_argument('args', nargs='*', help='Input file (and optional output file)')

    args = parser.parse_args()
    
    try:
        pairs = validate_and_get_pairs(args, args.args, tool_name="visualization.py", allow_single_file_stdio=True)
        
        md_parser = MarkdownParser()
        visualizer = TreeVisualizer()
        
        for input_path, output_path in pairs:
            root = md_parser.parse_file(input_path)
            
            # Visualization usually prints to stdout. Capturing it for file output requires redirecting stdout 
            # or modifying visualize method. modifying visualize method is better or using context manager.
            # Visualizer.visualize prints directly. I should probably redirect stdout if output_path is set.
            
            if output_path:
                with open(output_path, 'w') as f:
                    # Redirect stdout
                    original_stdout = sys.stdout
                    sys.stdout = f
                    try:
                        print(f"\nTree for: {os.path.basename(input_path)}\n")
                        visualizer.visualize(root)
                        print("\n")
                    finally:
                        sys.stdout = original_stdout
            else:
                print(f"\nTree for: {os.path.basename(input_path)}\n")
                visualizer.visualize(root)
                print("\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
