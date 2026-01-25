import sys
import os
import argparse
import tempfile
import shutil
import subprocess

# Add language directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
language_dir = os.path.join(grandparent_dir, 'language')
# Add language directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
language_dir = os.path.join(current_dir, 'language')
sys.path.append(language_dir)

try:
    from md_parser import MarkdownParser, Node
    import migrate
except ImportError as e:
    print(f"Error importing language tools: {e}")
    sys.exit(1)

def transform_github_url_to_folder_name(url):
    # Extract the last part of the URL (e.g., central_planner from https://github.com/user/central_planner)
    name = url.rstrip('/').split('/')[-1]
    if name.endswith('.git'):
        name = name[:-4]
    return name

def adjust_level(node, increment):
    node.level += increment
    for child in node.children:
        adjust_level(child, increment)

def build_master_plan(repo_url, master_plan_path):
    repo_name = transform_github_url_to_folder_name(repo_url)
    print(f"Processing Repository: {repo_name} ({repo_url})")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning to temporary directory: {temp_dir}")
        try:
            subprocess.run(['git', 'clone', repo_url, temp_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"Error: Failed to clone {repo_url}")
            return

        # 1. Audit and Migrate files in temp_dir
        md_files = []
        for root, dirs, files in os.walk(temp_dir):
            if '.git' in dirs:
                dirs.remove('.git') # Don't traverse .git
            
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.join(root, file)
                    # Run migration to ensure updated metadata
                    migrate.migrate_file(full_path)
                    md_files.append(full_path)
        
        print(f"Found and audited {len(md_files)} Markdown files.")

        # 2. Prepare Master Plan
        parser = MarkdownParser()
        
        if os.path.exists(master_plan_path):
            print(f"Reading existing Master Plan: {master_plan_path}")
            master_root = parser.parse_file(master_plan_path)
        else:
            print(f"Creating new Master Plan: {master_plan_path}")
            master_root = Node(0, "Root")
            master_root.children.append(Node(1, "Master Plan", {"status": "active"}))
        
        # 3. Find or Create Repository Node
        # We expect a structure like:
        # # Master Plan
        # ## Repo Name
        
        # Find the main "Master Plan" node (Level 1)
        main_node = None
        for child in master_root.children:
            if child.level == 1:
                main_node = child
                break
        
        if not main_node:
            main_node = Node(1, "Master Plan", {"status": "active"})
            master_root.children.append(main_node)
        
        # Find Repo Node under Main Node
        repo_node = None
        for child in main_node.children:
            if child.title == repo_name:
                repo_node = child
                break
        
        if repo_node:
            print(f"Updating existing section for '{repo_name}'. Clearing previous content.")
            repo_node.children = [] # Clear existing children for idempotency
            repo_node.metadata['updated'] = 'true'
        else:
            print(f"Creating new section for '{repo_name}'")
            repo_node = Node(2, repo_name, {"status": "active", "type": "repository", "source": repo_url})
            main_node.children.append(repo_node)

        # 4. Merging Files
        # We append all files found in the repo under the repo_node
        # To avoid chaos, let's just append them.
        # Future improvement: Reconstruct directory structure?
        # For now, flattened list of files or structure preserved?
        # The prompt says: "merge them into a MASTER_PLAN.md files which contains a general all encompasing abstraction"
        # Flattening is simplest for now.
        
        # Base level for children of repo_node (Level 2) is 3.
        target_base_level = repo_node.level + 1

        for md_file in md_files:
            try:
                source_root = parser.parse_file(md_file)
                
                # Each file usually has specific headers.
                # We append source_root.children to repo_node
                for child in source_root.children:
                    # Calculate level shift
                    # child.level should become target_base_level ?
                    # If child.level is 1 (# Title), it becomes 3 (### Title)
                    increment = target_base_level - child.level
                    adjust_level(child, increment)
                    
                    # Add filename context if useful? 
                    # Maybe add a "source_file" metadata?
                    rel_path = os.path.relpath(md_file, temp_dir)
                    child.metadata['file_path'] = rel_path
                    
                    repo_node.children.append(child)
                    
            except Exception as e:
                print(f"Warning: Failed to parse {md_file}: {e}")

        # 5. Write back
        new_content = master_root.to_markdown()
        with open(master_plan_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Successfully built Master Plan at {master_plan_path}")

def update_all(repolist_path, master_plan_path):
    if not os.path.exists(repolist_path):
        print(f"Error: Repository list not found at {repolist_path}")
        return

    with open(repolist_path, 'r') as f:
        repos = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not repos:
        print("No repositories found in list.")
        return

    print(f"Found {len(repos)} repositories to process.")
    for repo_url in repos:
        print(f"\n--- Syncing {repo_url} ---")
        build_master_plan(repo_url, master_plan_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and Merge remote repository markdown files into Master Plan.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="GitHub Repository URL to process")
    group.add_argument("--all", action="store_true", help="Update all repositories listed in manager/repolist.txt")
    
    parser.add_argument("--master-plan", default="MASTER_PLAN.md", help="Path to Master Plan file (default: MASTER_PLAN.md)")
    
    args = parser.parse_args()
    
    if args.all:
        repolist = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repolist.txt')
        update_all(repolist, args.master_plan)
    else:
        build_master_plan(args.repo, args.master_plan)
