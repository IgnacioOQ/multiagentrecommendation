import sys
import os
import re
import argparse
import cli_utils
from cli_utils import add_standard_arguments, validate_and_get_pairs

def is_valid_metadata_line(line):
    """
    Checks if a line is a valid metadata line based on user criteria:
    - Starts with '- ' (list item)
    - Key-value pair with ':' separator
    - Key is entirely lowercase
    - Key has no spaces (words separated by dash or underscore)
    - Further colons are part of value
    """
    stripped = line.strip()
    if not stripped.startswith("- "):
        return False
    
    # Remove list marker
    content = stripped[2:]
    
    if ":" not in content:
        return False
        
    parts = content.split(":", 1)
    key = parts[0]
    
    # Key validation
    # "entirely lowercase" and "no spaces"
    if any(c.isupper() or c.isspace() for c in key):
        return False
        
    return True

def is_content_separator(line):
    """Checks if a line is the content separator <!-- content -->."""
    return bool(re.match(r'^\s*<!--\s*content\s*-->\s*$', line))

def get_metadata_block(lines, start_index):
    """
    Returns the metadata lines and the parsed dict from a block starting at start_index.
    Stops at the first non-metadata line. Also captures trailing separator.
    """
    meta_lines = []
    parsed_data = {}
    current_index = start_index
    
    while current_index < len(lines):
        line = lines[current_index]
        if is_valid_metadata_line(line):
            meta_lines.append(line)
            
            # Parse key-value
            stripped = line.strip()[2:]
            key, value = stripped.split(":", 1)
            parsed_data[key] = value.strip()
            
            current_index += 1
        elif is_content_separator(line):
            # Include separator in meta_lines so it gets removed too
            meta_lines.append(line)
            current_index += 1
            break
        else:
            break
            
    return meta_lines, parsed_data, current_index

def find_next_header(lines, start_index, min_level):
    """
    Finds the index of the next header with level <= min_level.
    Returns len(lines) if not found.
    """
    header_pattern = re.compile(r'^(#+)\s+')
    
    for i in range(start_index, len(lines)):
        match = header_pattern.match(lines[i])
        if match:
            level = len(match.group(1))
            if level <= min_level:
                return i
    return len(lines)

def process_file(file_path, remove_incomplete_content, remove_incomplete_sections):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False
    
    new_lines = []
    header_pattern = re.compile(r'^(#+)\s+')
    
    i = 0
    changes_made = False
    
    while i < len(lines):
        line = lines[i]
        header_match = header_pattern.match(line)
        
        if header_match:
            header_level = len(header_match.group(1))
            
            # Look ahead for metadata
            meta_lines, meta_data, meta_end_index = get_metadata_block(lines, i + 1)
            
            # "undoes what migrate.py does" -> we remove metadata if found.
            # But wait, we only remove metadata blocks that match our strict criteria? Yes.
            
            status = meta_data.get('status')
            
            # Determine if we need to remove content or section
            # Condition: has metadata 'status:' different than 'done'
            is_incomplete = False
            if 'status' in meta_data:
                if status != 'done':
                    is_incomplete = True
            
            if remove_incomplete_sections and is_incomplete:
                # Remove section: Remove header + metadata + content
                # Content ends at next header of level <= header_level
                
                # Check where the content ends
                # Content starts after metadata
                next_header_index = find_next_header(lines, meta_end_index, header_level)
                
                # Skip everything from i to next_header_index
                i = next_header_index
                changes_made = True
                continue
                
            elif remove_incomplete_content and is_incomplete:
                # Remove content: Keep header, Remove metadata, Remove content
                # Keep header
                new_lines.append(line)
                
                # Content starts after metadata
                next_header_index = find_next_header(lines, meta_end_index, header_level)
                
                # Skip metadata and content
                i = next_header_index
                changes_made = True
                continue
            
            else:
                # valid metadata block found -> remove it (standard behavior)
                # If no metadata found (meta_lines empty), we just keep going.
                
                new_lines.append(line)
                if meta_lines:
                    # Skip metadata lines
                    i = meta_end_index
                    changes_made = True
                else:
                    i += 1
        else:
            new_lines.append(line)
            i += 1
            
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Processed {file_path}")
        return True
    else:
        # print(f"No changes for {file_path}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove metadata from markdown files (counterpart to migrate.py).")
    add_standard_arguments(parser, multi_file=True)
    
    parser.add_argument('--remove-incomplete-content', action='store_true',
                        help="Remove content of sections with status != 'done' (keep header).")
    parser.add_argument('--remove-incomplete-sections', action='store_true',
                        help="Remove sections (header and content) with status != 'done'.")
                        
    parser.add_argument('args', nargs='*', help='Input file(s)')
    
    args = parser.parse_args()
    
    try:
        pairs = validate_and_get_pairs(args, args.args, tool_name="remove_meta.py", allow_single_file_stdio=False)
        
        for input_path, output_path in pairs:
            # If output_path != input_path, copy first
            if input_path != output_path:
                import shutil
                shutil.copy2(input_path, output_path)
                target = output_path
            else:
                target = input_path
            
            process_file(target, args.remove_incomplete_content, args.remove_incomplete_sections)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
