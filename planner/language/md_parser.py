import re
import sys
import os
import json
import argparse
# Add current directory to sys.path to ensure we can import cli_utils if running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import cli_utils
from cli_utils import add_standard_arguments, validate_and_get_pairs

class Node:
    def __init__(self, level, title, metadata=None, content=""):
        self.level = level
        self.title = title
        self.metadata = metadata if metadata else {}
        self.content = content
        self.children = []

    def to_dict(self):
        """Convert Node tree to JSON-serializable dictionary."""
        return {
            "level": self.level,
            "title": self.title,
            "metadata": self.metadata,
            "content": self.content,
            "children": [child.to_dict() for child in self.children]
        }

    @classmethod
    def from_dict(cls, data):
        """Create Node tree from JSON-serializable dictionary (inverse of to_dict)."""
        node = cls(
            level=data.get("level", 0),
            title=data.get("title", ""),
            metadata=data.get("metadata", {}),
            content=data.get("content", "")
        )
        for child_data in data.get("children", []):
            node.children.append(cls.from_dict(child_data))
        return node

    def to_markdown(self):
        md_lines = []
        if self.level > 0:
            md_lines.append(f"{'#' * self.level} {self.title}")
            for key, value in self.metadata.items():
                if isinstance(value, list):
                    val_str = f"[{', '.join(str(v) for v in value)}]"
                    md_lines.append(f"- {key}: {val_str}")
                elif isinstance(value, dict):
                    val_str = json.dumps(value)
                    md_lines.append(f"- {key}: {val_str}")
                else:
                    md_lines.append(f"- {key}: {value}")
            if self.metadata:
                md_lines.append("<!-- content -->")  # Separator after metadata
        
        if self.content:
            md_lines.append(self.content)
            # Ensure proper spacing if content doesn't end with newlines
            if not self.content.endswith('\n'):
                md_lines.append("")
        elif self.level > 0:
             # If no content but valid node, ensure at least one newline separator for clarity
             md_lines.append("")

        for child in self.children:
            md_lines.append(child.to_markdown())
            
        return "\n".join(md_lines)

# Separator constant - used to distinguish metadata from content
CONTENT_SEPARATOR = "<!-- content -->"
CONTENT_SEPARATOR_PATTERN = re.compile(r'^\s*<!--\s*content\s*-->\s*$')

class MarkdownParser:
    def __init__(self):
        # Regex for headers: # Title
        self.header_pattern = re.compile(r'^(#+)\s+(.*)')
        # Regex for metadata lines: - key: value
        self.metadata_pattern = re.compile(r'^\s*-\s*([a-zA-Z0-9_]+):\s*(.*)')
        # Regex for content separator: <!-- content -->
        self.separator_pattern = CONTENT_SEPARATOR_PATTERN

    def parse_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        return self.parse_lines(lines)

    def parse_lines(self, lines):
        root = Node(0, "Root")
        node_stack = [root] # Stack to track hierarchy

        i = 0
        while i < len(lines):
            line = lines[i]
            header_match = self.header_pattern.match(line)

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                new_node = Node(level, title)

                # Look ahead for metadata (immediately following header)
                i += 1
                while i < len(lines):
                    meta_line = lines[i]
                    # Check for content separator - this definitively ends metadata block
                    if self.separator_pattern.match(meta_line):
                        i += 1  # Skip the separator line
                        break
                    
                    # Check for metadata line
                    meta_match = self.metadata_pattern.match(meta_line)
                    
                    if meta_match:
                        key = meta_match.group(1).strip()
                        value_str = meta_match.group(2).strip()
                        
                        # Basic list parsing: [a, b, c]
                        if value_str.startswith('[') and value_str.endswith(']'):
                            inner = value_str[1:-1]
                            if inner.strip():
                                value = [x.strip() for x in inner.split(',')]
                            else:
                                value = []
                        elif value_str.startswith('{') and value_str.endswith('}'):
                            try:
                                value = json.loads(value_str)
                            except json.JSONDecodeError:
                                # If invalid JSON, treat as string
                                value = value_str
                        else:
                            value = value_str
                            
                        new_node.metadata[key] = value
                        i += 1
                    elif meta_line.strip() == "":
                        # Blank line handling for backward compatibility:
                        # If we've found metadata, blank line ends the block (old separator)
                        # If no metadata yet, skip blank line and continue looking
                        if new_node.metadata:
                            i += 1  # Skip the blank line
                            break   # End metadata block
                        else:
                            i += 1
                            continue
                    else:
                        # Non-metadata line, stop looking for metadata
                        break
                
                # Capture content until next header
                content_lines = []
                while i < len(lines):
                    if self.header_pattern.match(lines[i]):
                        break # Next header found
                    content_lines.append(lines[i])
                    i += 1
                
                new_node.content = "".join(content_lines).strip()

                # Place node in hierarchy
                # Pop until we find a parent with level < new_node.level
                while node_stack[-1].level >= level:
                    node_stack.pop()
                
                node_stack[-1].children.append(new_node)
                node_stack.append(new_node)
            else:
                # Content before first header?
                # For now just ignore or append to root content if needed
                i += 1
        
        # Check if we should unwrap the root
        # If there is exactly one top-level child, and the Root node itself has no content or metadata,
        # return the child directly.
        if len(root.children) == 1 and not root.content and not root.metadata:
             return root.children[0]
             
        return root

    def validate(self, node):
        errors = []
        # Example validation: check if status is valid enum if present
        if 'status' in node.metadata:
            valid_statuses = ['todo', 'in-progress', 'done', 'blocked', 'proposed', 'active', 'draft', 'pending'] # Expanded list based on usage
            if node.metadata['status'] not in valid_statuses:
                errors.append(f"Invalid status '{node.metadata['status']}' in node '{node.title}' (Allowed: {valid_statuses})")
        
        if 'type' in node.metadata:
            valid_types = ['plan', 'task', 'recurring', 'agent_skill', 'protocol', 'guideline', 'log', 'context']
            if node.metadata['type'] not in valid_types:
                errors.append(f"Invalid type '{node.metadata['type']}' in node '{node.title}' (Allowed: {valid_types})")

        for child in node.children:
            errors.extend(self.validate(child))
        return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and validate Markdown files with Metadata.")
    add_standard_arguments(parser, multi_file=False)
    # Also add positional args which are handled by validat_and_get_pairs
    parser.add_argument('args', nargs='*', help='Input file (and optional output file)')

    args = parser.parse_args()
    
    try:
        # allow_single_file_stdio=True because md_parser can print to stdout
        pairs = validate_and_get_pairs(args, args.args, tool_name="md_parser.py", allow_single_file_stdio=True)
        
        parser_obj = MarkdownParser()
        
        for input_path, output_path in pairs:
            root_node = parser_obj.parse_file(input_path)
            errors = parser_obj.validate(root_node)
            
            if errors:
                print(f"Validation Errors in {input_path}:", file=sys.stderr)
                for err in errors:
                    print(f"- {err}", file=sys.stderr)
                sys.exit(1)
            else:
                result = json.dumps(root_node.to_dict(), indent=2)
                if output_path:
                    with open(output_path, 'w') as f:
                        f.write(result)
                else:
                    print(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
