#!/usr/bin/env python3
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup path to find 'language' module
current_file_path = os.path.abspath(__file__)
planner_dir = os.path.dirname(current_file_path)
manager_dir = os.path.dirname(planner_dir)
root_dir = os.path.dirname(manager_dir)
# Setup path to find 'language' module
current_file_path = os.path.abspath(__file__)
planner_dir = os.path.dirname(current_file_path)
language_dir = os.path.join(planner_dir, 'language')

if language_dir not in sys.path:
    sys.path.append(language_dir)

try:
    from md_parser import MarkdownParser
except ImportError as e:
    print(f"Error: Could not import md_parser from {language_dir}. {e}")
    sys.exit(1)

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    '''
    if not nx.is_tree(G):
        # Fallback for non-trees (though our main structure is a tree)
        # We'll use a breadth-first search to simulate levels
        return nx.spring_layout(G)

    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
    '''
    Recursive helper function for hierarchy_pos.
    '''
    if pos is None:
        pos = {root:(xcenter,vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
        
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)  

    if len(children)!=0:
        dx = width/len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                pos=pos, parent = root)
    return pos

def build_graph(node, graph=None, parent_id=None):
    if graph is None:
        graph = nx.DiGraph()
    
    # Use ID if available, otherwise Title, otherwise hash
    # But for visual labels we want the Title.
    # We'll use the unique python object id or the 'id' metadata as the node key
    # to avoid name collisions, but store the title as a property.
    
    node_id = node.metadata.get('id', node.title) if node.metadata else node.title
    # Fallback to avoid duplicates if titles are same
    if node_id in graph:
        node_id = f"{node_id}_{id(node)}"

    # Determine status color
    status = node.metadata.get('status', 'todo') if node.metadata else 'todo'
    color_map = {
        'done': '#2ecc71',      # Green
        'active': '#3498db',    # Blue
        'in-progress': '#3498db',
        'blocked': '#e74c3c',   # Red
        'todo': '#bdc3c7',      # Light Gray
        'draft': '#95a5a6'      # Gray
    }
    color = color_map.get(status, '#bdc3c7')
    
    # Determine Label (wrap text)
    label = node.title
    if len(label) > 20:
        label = label[:18] + "..."

    graph.add_node(node_id, label=label, color=color, full_title=node.title)
    
    if parent_id:
        graph.add_edge(parent_id, node_id, edge_type='hierarchy')
        
    # Check for Explicit Dependencies (blocked_by) to add dashed edges
    # Note: This is tricky because we need to know the IDs of other nodes.
    # We might need a second pass or global lookup.
    # For now, we'll store the blocked_by list in the node data and add edges later if nodes exist.
    blocked_by = node.metadata.get('blocked_by', []) if node.metadata else []
    if blocked_by:
         nx.set_node_attributes(graph, {node_id: {'blocked_by': blocked_by}})

    for child in node.children:
        build_graph(child, graph, node_id)
        
    return graph

def add_dependency_edges(graph):
    # Iterate over all nodes, check 'blocked_by' attribute
    # blocked_by is expected to be a list of IDs.
    # We need to find the graph-node that corresponds to that ID.
    # Since our graph keys ARE the IDs (mostly), we can try direct lookup.
    
    nodes_with_blockers = [n for n, d in graph.nodes(data=True) if 'blocked_by' in d]
    
    for u in nodes_with_blockers:
        blockers = graph.nodes[u]['blocked_by']
        for blocker_id in blockers:
            # We assume blocker_id matches the graph node key.
            # In a real system, we might need a lookup table id -> node_key
            if blocker_id in graph:
                graph.add_edge(blocker_id, u, edge_type='dependency')
            else:
                # Try simple matching
                pass
    return graph

def main():
    target_file = os.path.join(planner_dir, "MASTER_PLAN.md")
    if not os.path.exists(target_file):
        print("MASTER_PLAN.md not found")
        sys.exit(1)
        
    parser = MarkdownParser()
    root_node = parser.parse_file(target_file)
    
    G = build_graph(root_node)
    G = add_dependency_edges(G)
    
    # Layout
    # Extract root id
    root_id = list(G.nodes())[0] # Approximated, since we built depth first
    # Better: find node with in-degree 0 (hierarchical)
    roots = [n for n,d in G.in_degree() if d==0]
    true_root = roots[0] if roots else None
    
    # Create hierarchy layout only on Hierarchy edges
    # We create a view of the graph with only hierarchy edges to compute layout
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from([(u,v) for u,v,d in G.edges(data=True) if d.get('edge_type') == 'hierarchy'])
    
    if nx.is_tree(H):
        pos = hierarchy_pos(H, root=true_root)
    else:
        print("Warning: Structure is not a strict tree. Using spring layout.")
        pos = nx.spring_layout(G)
        
    # Draw
    plt.figure(figsize=(16, 10))
    
    # Draw Edges
    hierarchy_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('edge_type') == 'hierarchy']
    dependency_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('edge_type') == 'dependency']
    
    nx.draw_networkx_edges(G, pos, edgelist=hierarchy_edges, edge_color='#7f8c8d', arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=dependency_edges, edge_color='#e74c3c', style='dashed', connectionstyle='arc3,rad=0.2')
    
    # Draw Nodes
    colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    labels = nx.get_node_attributes(G, 'label')
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000, alpha=0.9, node_shape='s')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
    
    # Legend
    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Done'),
        mpatches.Patch(color='#3498db', label='In Progress'),
        mpatches.Patch(color='#bdc3c7', label='To Do'),
        mpatches.Patch(color='#e74c3c', label='Blocked / Blocked Dependency')
    ]
    plt.legend(handles=legend_patches, loc='upper right')
    
    plt.title("Master Plan Visualization", fontsize=16)
    plt.axis('off')
    
    output_file = os.path.join(planner_dir, "master_plan_viz.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Try to show if interactive
    try:
        # plt.show() 
        pass
    except:
        pass

if __name__ == "__main__":
    main()
