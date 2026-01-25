import sys
import os
import argparse

# Add the current directory to sys.path to allow importing md_parser if run from scripts/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from md_parser import MarkdownParser, Node

def find_node(node, identifier, by_id=False):
    if by_id:
        if node.metadata.get('id') == identifier:
            return node
    else:
        if node.title == identifier:
            return node
            
    for child in node.children:
        found = find_node(child, identifier, by_id)
        if found:
            return found
    return None

def merge_trees(target_file, source_file, target_identifier, output_file=None):
    parser = MarkdownParser()
    target_root = parser.parse_file(target_file)
    source_root = parser.parse_file(source_file)

    # Find insertion point
    # First try to find by ID (assuming target_identifier might be an ID)
    target_node = find_node(target_root, target_identifier, by_id=True)
    
    # If not found, try by Title
    if not target_node:
        target_node = find_node(target_root, target_identifier, by_id=False)

    if not target_node:
        print(f"Error: Target node '{target_identifier}' (ID or Title) not found in {target_file}")
        sys.exit(1)

    # In a merge, we typically append the children of the source root (since source root is usually documented title)
    # as children of the target node. Or do we append the whole source root?
    # Usually source file is like:
    # # Submodule
    # ## Task
    # And we want to insert it under # Feature A.
    # So we probably want to append source_root's children if source_root is just a container (level 0 or 1).
    # Let's assume we append the children of the source root to the target node.
    
    # Adjust levels
    # target_node level = N. Children should be N+1.
    # source children level = M.
    # We need to shift source children levels by (target_node.level + 1) - source_children[0].level (roughly)
    
    base_level = target_node.level + 1
    
    # Recursive level adjuster
    def adjust_level(node, increment):
        node.level += increment
        for child in node.children:
            adjust_level(child, increment)

    # Scan source children
    for child in source_root.children:
        # Determine increment. child.level should become base_level.
        # current child level might be 1 (if source file starts with #)
        increment = base_level - child.level
        adjust_level(child, increment)
        target_node.children.append(child)

    # Write output
    output_content = target_root.to_markdown()
    
    dest = output_file if output_file else target_file
    with open(dest, 'w') as f:
        f.write(output_content)
    
    print(f"Successfully merged {source_file} into {target_identifier} of {target_file}. Saved to {dest}.")

def extend_tree(target_file, source_file, output_file=None):
    # Extend basically means append source file children to target file root children
    # Effectively concatenation at top level
    parser = MarkdownParser()
    target_root = parser.parse_file(target_file)
    source_root = parser.parse_file(source_file)
    
    for child in source_root.children:
        target_root.children.append(child)
        # Levels should presumably stay same if both are valid docs with level 1 headers
    
    output_content = target_root.to_markdown()
    dest = output_file if output_file else target_file
    with open(dest, 'w') as f:
        f.write(output_content)
    print(f"Successfully extended {target_file} with {source_file}. Saved to {dest}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operations on Markdown-JSON trees.")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Insert source tree into target node")
    merge_parser.add_argument("target_file", help="Target Markdown file")
    merge_parser.add_argument("source_file", help="Source Markdown file to insert")
    merge_parser.add_argument("target_node", help="ID or Title of the node in target file to insert under")
    merge_parser.add_argument("--output", "-o", help="Output file (default: overwrite target)")

    # Extend command
    extend_parser = subparsers.add_parser("extend", help="Append source tree to target tree")
    extend_parser.add_argument("target_file", help="Target Markdown file")
    extend_parser.add_argument("source_file", help="Source Markdown file to append")
    extend_parser.add_argument("--output", "-o", help="Output file (default: overwrite target)")

    args = parser.parse_args()

    # Note: operations.py logic is 2-input (target, source) -> 1 output.
    # It does not fit the simple "transform list of files" pattern.
    # We keep it as is with explicit args.

    if args.command == "merge":
        merge_trees(args.target_file, args.source_file, args.target_node, args.output)
    elif args.command == "extend":
        extend_tree(args.target_file, args.source_file, args.output)
    else:
        parser.print_help()
