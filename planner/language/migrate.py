import sys
import os
import re
import argparse
from datetime import date
import unicodedata

# Simple migration script
# Logic:
# 1. Read file
# 2. Identify headers
# 3. Check if header has METADATA block immediately after
# 4. If not, insert default METADATA block
# 5. Write back

ALLOWED_FIELDS = {
    'status', 'type', 'owner', 'estimate', 'blocked_by', 
    'priority', 'id', 'last_checked', 'context_dependencies'
}

# Content separator constant
CONTENT_SEPARATOR = "<!-- content -->"

def has_meta_block(lines, header_index):
    # check next few lines for METADATA-like content
    if header_index + 1 >= len(lines):
        return False
    
    # Check immediate next line (standard/strict) or allow 1 empty line?
    # Our convention says "immediately following", so let's check index+1
    next_line = lines[header_index + 1].strip()
    
    # Regex to capture key: "- key: value" or "key: value"
    match = re.match(r'^-?\s*(\w+):', next_line)
    if match:
        key = match.group(1)
        if key in ALLOWED_FIELDS:
            return True
        
    return False

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '_', value)

def migrate_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    
    header_pattern = re.compile(r'^(#+)\s+(.*)')
    header_stack = [] # List of (level, slug) tuples
    
    i = 0
    while i < len(lines):
        line = lines[i]
        header_match = header_pattern.match(line)
        
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2)
            current_slug = slugify(title)
            
            # Update stack
            # Pop headers that are at the same level or deeper (higher number of # is deeper, but sibling is same level)
            # Standard markdown: ## is level 2. 
            # If we are at level 2, we pop anything level 2 or greater from the stack top.
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            
            header_stack.append((level, current_slug))
            
        new_lines.append(line)
        
        if header_match:
            # Check if existing METADATA
            if not has_meta_block(lines, i):
                # Insert default METADATA
                # Infer status if possible, otherwise default to "active" or "documented"
                # For AGENTS files, "active" seems appropriate.
                # Convention update: add a blank line after metadata

                # Update logic: Generate ID and last_checked
                
                # Reconstruct full ID from stack
                full_id = ".".join([item[1] for item in header_stack])
                current_date = date.today().isoformat()
                
                default_context = '{ "conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md" }'
                default_meta = f"- id: {full_id}\n- status: active\n- type: context\n- context_dependencies: {default_context}\n- last_checked: {current_date}\n{CONTENT_SEPARATOR}\n"
                
                # If there's a newline after header, we can insert before it or after it?
                # Usually we want Header\nMETADATA.
                # If the line ends with \n, we just append to new_lines.
                
                # But wait, we just appended 'line' (the header) to new_lines.
                # Now we append the METADATA.
                new_lines.append(default_meta)
                
                # Also, we might want to add 'owner' or 'updated' if we could guess, but let's keep it simple.
            else:
                # Header exists and has metadata block. Check for blank line after metadata.
                # has_meta_block returned True.
                # Re-find the metadata block end index
                j = i + 1
                meta_pattern = re.compile(r'^-?\s*(\w+):')
                found_meta = False
                while j < len(lines):
                    match = meta_pattern.match(lines[j].strip())
                    is_valid_meta = False
                    if match:
                        key = match.group(1)
                        if key in ALLOWED_FIELDS:
                            is_valid_meta = True
                    
                    if is_valid_meta:
                        found_meta = True
                        j += 1
                    elif found_meta and (lines[j].strip() == "" or re.match(r'^\s*<!--\s*content\s*-->\s*$', lines[j])):
                         # Already has blank line or separator
                         break
                    else:
                         # Found end of metadata, and lines[j] is NOT blank
                         # Insert blank line
                         if found_meta: # Only if we actually traversed some metadata
                             new_lines.extend(lines[i+1:j])
                             new_lines.append(f"{CONTENT_SEPARATOR}\n")
                             # We want to continue outer loop from j.
                             # Outer loop does i+=1 at end.
                             # So set i = j - 1
                             i = j - 1
                             break
                         break
        
        i += 1
    
    # Write back only if changed
    if new_lines != lines:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Migrated {file_path}")
        return True
    else:
        # print(f"No changes needed for {file_path}")
        return False

import cli_utils
from cli_utils import add_standard_arguments, validate_and_get_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate markdown files to Protocol format.")
    add_standard_arguments(parser, multi_file=True)
    parser.add_argument('args', nargs='*', help='Input file(s)')
    
    args = parser.parse_args()
    
    try:
        # migrate.py updates files. 
        # allow_single_file_stdio=False because it does not support stdout mode easily (it modifies in place).
        pairs = validate_and_get_pairs(args, args.args, tool_name="migrate.py", allow_single_file_stdio=False)
        
        for input_path, output_path in pairs:
            # If output_path != input_path, we should copy first then migrate? 
            # OR read input, migrate in memory, write to output.
            # migrate_file function currently does read -> check -> write. 
            # It takes file_path. It reads from file_path and writes to file_path.
            # We need to adapt it or copy the file first.
            
            if input_path != output_path:
                import shutil
                shutil.copy2(input_path, output_path)
                target = output_path
            else:
                target = input_path
                
            migrate_file(target)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
