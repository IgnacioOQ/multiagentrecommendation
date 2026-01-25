import sys
import os

# Set up paths to import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from md_parser import MarkdownParser
from visualization import TreeVisualizer
from visualize_dag import MermaidVisualizer, DependencyReportVisualizer

def run_tree_demo():
    print("\n" + "="*20 + " TREE VISUALIZATION DEMO " + "="*20)
    print("Visualizing hierarchical structure of 'MERGED_PLAN.md'...\n")
    
    file_path = os.path.join(current_dir, "MERGED_PLAN.md")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    parser = MarkdownParser()
    root = parser.parse_file(file_path)
    
    visualizer = TreeVisualizer()
    visualizer.visualize(root)

def run_dag_demo():
    print("\n" + "="*20 + " DAG VISUALIZATION DEMO (Text Report) " + "="*20)
    print("Visualizing dependencies (DAG) of 'DAG_Example.md' in human-readable format...\n")
    
    file_path = os.path.join(current_dir, "DAG_Example.md")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    parser = MarkdownParser()
    root = parser.parse_file(file_path)
    
    visualizer = DependencyReportVisualizer()
    visualizer.generate(root)

if __name__ == "__main__":
    run_tree_demo()
    run_dag_demo()
    print("\n" + "="*60 + "\n")
