import sys
import os
import re

# Add the current directory to sys.path to allow importing md_parser if run from scripts/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from md_parser import MarkdownParser

def sanitize_id(text):
    """Sanitizes a string to be a valid Mermaid ID."""
    # Keep only alphanumerics and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', text)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized

class MermaidVisualizer:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.subgraphs = []

    def process_node(self, node, parent_id=None):
        # Determine Node ID (prioritize global ID, fallback to sanitized title)
        raw_id = node.metadata.get('id')
        if not raw_id:
            # Fallback to sanitized title + unique hash if needed (simple unique here)
            # To ensure uniqueness without global ID, we might need path.
            # For simplicity, let's use sanitized title but this risks collisions.
            # Improved: use object hash or parent-prefixed path? 
            # Let's use sanitized title for readability, warn if no ID.
            node_id = sanitize_id(node.title)
        else:
            node_id = sanitize_id(raw_id)
        
        # Store for lookup
        self.nodes.append((node_id, node.title, node))

        # Hierarchy Edge (if we aren't using subgraphs, we might use edges)
        # Decision: Use Subgraphs for hierarchy if it has children, else it's a node.
        # But Mermaid subgraphs are tricky if edges go *to* the subgraph.
        # Let's try: Nested Subgraphs.
        
        is_subgraph = len(node.children) > 0
        
        if is_subgraph:
            # It's a container
            block = f'subgraph {node_id} ["{node.title}"]\n'
            block += f'  direction TB\n' 
            # Recurse
            children_blocks = []
            for child in node.children:
                children_blocks.append(self.process_node(child, node_id))
            
            block += "".join(children_blocks)
            block += 'end\n'
            return block
        else:
            # It's a leaf node
            status = node.metadata.get('status', '')
            label = f"{node.title}\\n[{status}]" if status else node.title
            return f'  {node_id}["{label}"]\n'

    def extract_dependency_edges(self, node):
        edges = []
        node_id = self.get_node_id(node)
        
        blocked_by = node.metadata.get('blocked_by')
        if blocked_by:
            if not isinstance(blocked_by, list):
                blocked_by = [blocked_by] # Normalize to list
            
            for dep in blocked_by:
                # dep is a string ID. We assume it matches the sanitized global ID 
                # or we just use it raw?
                # The user puts "id: component.backend" in metadata.
                # In blocked_by, they put "component.backend".
                # So we should sanitize both identically.
                dep_id = sanitize_id(dep)
                
                # Edge: Dependency -> Node
                edges.append(f'{dep_id} --> {node_id}')
        
        for child in node.children:
            edges.extend(self.extract_dependency_edges(child))
            
        return edges

    def get_node_id(self, node):
        raw_id = node.metadata.get('id')
        if raw_id:
            return sanitize_id(raw_id)
        return sanitize_id(node.title)

    def generate(self, root_node):
        print("graph TD")
        
        # 1. Generate Structure (Nodes & Subgraphs)
        # We process children of root directly so we don't make a huge "Root" box unless needed.
        # But root usually represents the Project.
        structure = self.process_node(root_node)
        print(structure)
        
        # 2. Generate Dependency Edges
        # We need to traverse again or collect them during pass 1. 
        # Traversing again is cleaner.
        edges = self.extract_dependency_edges(root_node)
        
        if edges:
            print("\n  %% Dependencies")
            for edge in edges:
                print(f"  {edge}")

import argparse

class DependencyReportVisualizer:
    def __init__(self):
        self.node_map = {} # id -> node
        self.reverse_deps = {} # id -> list of ids that rely on this id (successors)

    def collect_nodes(self, node):
        node_id = self.get_node_id(node)
        self.node_map[node_id] = node
        
        # Determine dependencies (predecessors)
        blocked_by = node.metadata.get('blocked_by')
        if blocked_by:
            if not isinstance(blocked_by, list):
                blocked_by = [blocked_by]
            
            for dep in blocked_by:
                dep_id = sanitize_id(dep)
                # Register successor (dep blocks node, so node is successor of dep)
                if dep_id not in self.reverse_deps:
                    self.reverse_deps[dep_id] = []
                self.reverse_deps[dep_id].append(node_id)

        for child in node.children:
            self.collect_nodes(child)

    def get_node_id(self, node):
        raw_id = node.metadata.get('id')
        if raw_id:
            return sanitize_id(raw_id)
        return sanitize_id(node.title)

    def generate(self, root_node):
        self.collect_nodes(root_node)
        
        print("Dependency Report")
        print("=================")
        
        # Sort nodes by title for consistent output, or traversing?
        # Let's verify all nodes that have edges
        
        for node_id, node in self.node_map.items():
            predecessors = node.metadata.get('blocked_by', [])
            if not isinstance(predecessors, list) and predecessors:
                 predecessors = [predecessors]
            
            successors = self.reverse_deps.get(node_id, [])
            
            # Sanitize predecessors for matching
            sanitized_preds = [sanitize_id(p) for p in predecessors] if predecessors else []

            if not predecessors and not successors:
                continue # Skip isolated nodes in this view? Or show all?
                # User asked for DAG visualization, usually implies showing connections.
                # Let's skip strictly isolated nodes to reduce noise, or maybe show them as "Independent"
                # Let's show only nodes with connections for clarity.
                pass

            if predecessors or successors:
                status = node.metadata.get('status', 'n/a')
                print(f"\n[{node.title}] (ID: {node_id}) [Status: {status}]")
                
                if predecessors:
                    print(f"  Wait for:")
                    for pred_id in sanitized_preds:
                        pred_node = self.node_map.get(pred_id)
                        title = pred_node.title if pred_node else "Unknown"
                        print(f"    - {title} (ID: {pred_id})")
                
                if successors:
                    print(f"  Blocks:")
                    for succ_id in successors:
                        succ_node = self.node_map.get(succ_id)
                        title = succ_node.title if succ_node else "Unknown"
                        print(f"    - {title} (ID: {succ_id})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DAG from Markdown file.")
    parser.add_argument("file", help="Path to .md file")
    parser.add_argument("--format", choices=["mermaid", "text"], default="mermaid", help="Output format")
    
    args = parser.parse_args()

    file_path = args.file
    md_parser = MarkdownParser()
    
    try:
        root = md_parser.parse_file(file_path)
        
        if args.format == "text":
            visualizer = DependencyReportVisualizer()
            visualizer.generate(root)
        else:
            visualizer = MermaidVisualizer()
            visualizer.generate(root)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
