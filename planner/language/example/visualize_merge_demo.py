import sys
import os

# Add parent directory (language/) to sys.path to access the tools
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # language/
sys.path.append(parent_dir)

from visualization import TreeVisualizer
from md_parser import MarkdownParser
from operations import merge_trees

def visualize_file(path, title):
    print(f"\n{'='*10} {title} {'='*10}\n")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    parser = MarkdownParser()
    try:
        root = parser.parse_file(path)
        visualizer = TreeVisualizer()
        visualizer.visualize(root)
    except Exception as e:
        print(f"Error parsing {path}: {e}")

def main():
    general_path = os.path.join(current_dir, "GENERAL_PLAN.md")
    specific_path = os.path.join(current_dir, "SPECIFIC_PLAN.md")
    merged_path = os.path.join(current_dir, "MERGED_PLAN.md")

    # 1. Visualize Individual Plans
    visualize_file(general_path, "General Plan")
    visualize_file(specific_path, "Specific Plan")

    # 2. Merge
    print(f"\n{'='*10} MERGING {'='*10}\n")
    target_node = "Phase 2: Core Development"
    print(f"Merging 'Specific Plan' into '{target_node}' of 'General Plan'...")
    
    try:
        merge_trees(general_path, specific_path, target_node, output_file=merged_path)
    except Exception as e:
        print(f"Merge failed: {e}")
        # If operations uses sys.exit, we might not catch it here, but let's try.
        # Actually operations.py calls sys.exit(1) on error. 
        return

    # 3. Visualize Merged Result
    visualize_file(merged_path, "Merged Plan")

if __name__ == "__main__":
    main()
