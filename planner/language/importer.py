import sys
import os
import subprocess
import shutil
import argparse

# Try importing optional libraries
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

def convert_docx_to_md(file_path):
    """
    Converts DOCX to Markdown. 
    1. Tries python-docx for structure preservation.
    2. Fallback to MacOS textutil if available (converts to text/html then we clean up).
    """
    if HAS_DOCX:
        doc = docx.Document(file_path)
        md_lines = []
        for para in doc.paragraphs:
            style = para.style.name
            text = para.text.strip()
            if not text:
                continue
                
            if 'Heading 1' in style:
                md_lines.append(f"# {text}")
            elif 'Heading 2' in style:
                md_lines.append(f"## {text}")
            elif 'Heading 3' in style:
                md_lines.append(f"### {text}")
            else:
                md_lines.append(text)
            
            md_lines.append("") # space after paragraphs
            
        return "\n".join(md_lines)
        
    # MacOS Fallback: textutil
    if sys.platform == 'darwin' and shutil.which('textutil'):
        print(f"python-docx not found. Using MacOS textutil for {file_path}. Structure might be lost.")
        try:
            cmd = ['textutil', '-convert', 'txt', '-stdout', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Add a root header since textutil flattens it
            filename = os.path.basename(file_path)
            return f"# Imported {filename}\n\n" + result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error using textutil: {e}")
            return None

    print("Error: Library 'python-docx' not installed and 'textutil' unavailable.")
    print("Install it via: pip install python-docx")
    return None

def convert_pdf_to_md(file_path):
    """
    Converts PDF to Markdown (Text Extraction).
    """
    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(file_path)
            text = []
            filename = os.path.basename(file_path)
            text.append(f"# Imported {filename}")
            
            for page in reader.pages:
                text.append(page.extract_text())
                
            return "\n\n".join(text)
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return None
            
    # Fallback? Maybe `pdftotext` CLI
    if shutil.which('pdftotext'):
         print(f"pypdf not found. Using pdftotext CLI for {file_path}.")
         try:
            cmd = ['pdftotext', '-layout', file_path, '-']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            filename = os.path.basename(file_path)
            return f"# Imported {filename}\n\n" + result.stdout
         except subprocess.CalledProcessError as e:
             print(f"Error using pdftotext: {e}")
             return None

    print("Error: Library 'pypdf' not installed and 'pdftotext' unavailable.")
    print("Install it via: pip install pypdf")
    return None

def import_file(file_path, output_path=None):
    ext = os.path.splitext(file_path)[1].lower()
    
    md_content = None
    if ext == '.docx':
        md_content = convert_docx_to_md(file_path)
    elif ext == '.doc':
        # textutil handles .doc too
        if sys.platform == 'darwin' and shutil.which('textutil'):
             try:
                cmd = ['textutil', '-convert', 'txt', '-stdout', file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                filename = os.path.basename(file_path)
                md_content = f"# Imported {filename}\n\n" + result.stdout
             except Exception as e:
                 print(f"Error converting .doc: {e}")
        else:
            print("Cannot convert .doc files without MacOS textutil or external tools.")
    elif ext == '.pdf':
        md_content = convert_pdf_to_md(file_path)
    else:
        print(f"Unsupported format: {ext}")
        return

    if md_content:
        # Save as .md
        if output_path:
            new_path = output_path
        else:
            new_path = os.path.splitext(file_path)[0] + ".md"
            
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Converted to {new_path}")
        
        # Run migration to add metadata
        # We need to import migrate entry point
        # Assuming migrate.py is in the same folder
        import migrate
        print("Running migration on new file...")
        migrate.migrate_file(new_path)

import cli_utils
from cli_utils import add_standard_arguments, validate_and_get_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import legacy documents to Markdown.")
    add_standard_arguments(parser, multi_file=True)
    parser.add_argument('args', nargs='*', help='Input file(s)')
    
    args = parser.parse_args()
    
    try:
        # importer.py takes input and writes output.
        # It's not strictly "update in place".
        # If no output specified, it defaults to changing extension to .md.
        # So we should probably allow single file positional, but treat it as "derive output".
        # But cli_utils logic for single positional in "update" mode fails.
        # But importer is arguably "conversion".
        # However, the user said "For python programs that handle multiple input files...".
        # importer handles multiple input files.
        # So it should follow the strict -I rule.
        # If -I is used, we derive output name.
        # If -i/-o used, we use explicit output.
        
        pairs = validate_and_get_pairs(args, args.args, tool_name="importer.py", allow_single_file_stdio=False)
        
        for input_path, output_path in pairs:
            # import_file function currently takes just file_path and writes to file_path.with_suffix('.md')
            # changing it to support explicit output path is needed.
            
            # Logic inside loop:
            # If output_path == input_path (from -I or in-place logic), we actually want derived path for converter.
            # Wait, validate_and_get_pairs returns (path, path) for in-line.
            # For importer, input != output usually (docx vs md).
            # So if input == output, it means we should derive filename? 
            # Or is it an error to overwrite docx with md?
            # Usually strict update-in-place means overwrite. But here types differ.
            # Let's assume if input==output (from -I), we use default derivation logic.
            
            # We need to modify import_file to accept optional output path.
            # Or handle it here.
            
            if input_path == output_path:
                 # Default behavior invocation
                 import_file(input_path) 
                 # But import_file writes to derived name.
            else:
                 # Explicit output invocation
                 # We need to modify import_file or implement logic here.
                 # Let's look at import_file signature. It takes file_path.
                 # It calls convert_docx_to_md, gets content, writes to file.
                 # We can modify import_file to take output_path.
                 import_file(input_path, output_path=output_path)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
