#!/usr/bin/env python3
import sys
import os

# 1. Setup path to find 'language' module
# Current file: .../manager/planner/visualize_plan.py
current_file_path = os.path.abspath(__file__)
planner_dir = os.path.dirname(current_file_path)      # .../manager/planner
manager_dir = os.path.dirname(planner_dir)            # .../manager
root_dir = os.path.dirname(manager_dir)               # .../central_planner
# current_dir: .../manager/planner
current_file_path = os.path.abspath(__file__)
planner_dir = os.path.dirname(current_file_path)      # .../manager/planner
language_dir = os.path.join(planner_dir, 'language')

# Add language_dir to sys.path so we can import md_parser, visualization, etc.
if language_dir not in sys.path:
    sys.path.append(language_dir)

def main():
    try:
        from md_parser import MarkdownParser
        from visualization import TreeVisualizer
    except ImportError as e:
        print(f"Error: Could not import language tools. Checked path: {language_dir}")
        print(f"Detail: {e}")
        sys.exit(1)

    # 2. Define target file
    target_file = os.path.join(planner_dir, "MASTER_PLAN.md")
    
    if not os.path.exists(target_file):
        print(f"Error: MASTER_PLAN.md not found at {target_file}")
        sys.exit(1)

    # 3. Print nice header
    print(f"\n{'='*60}")
    print(f"  MASTER PLAN VISUALIZATION")
    print(f"  Source: {target_file}")
    print(f"{'='*60}\n")

    # 4. Parse and Visualize
    try:
        parser = MarkdownParser()
        visualizer = TreeVisualizer()
        
        root = parser.parse_file(target_file)
        visualizer.visualize(root)
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"Error processing plan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
