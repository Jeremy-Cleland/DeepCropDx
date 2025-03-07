# Project Code Overview (Complete Version)

*Generated on 2025-03-06 14:38:08*

## Project Structure

```
scripts
├── extract_trial_data.py
├── model_snapshot.md
└── project_docs
    ├── project_code.md
    └── project_code_full.md
```

## Table of Contents

- [extract_trial_data.py](#extract_trial_data-py)

## Code Files

### extract_trial_data.py

```python
#!/usr/bin/env python3
import os
import json
import argparse
import glob
from pathlib import Path
import sys


def get_models_dir():
    """Get the models directory from the scripts folder"""
    # Get the directory where this script is located (scripts folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)

    # Path to models directory
    models_dir = os.path.join(project_root, "models")

    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        sys.exit(1)

    return models_dir


def build_directory_tree(models_dir):
    """Create a markdown-formatted directory tree of the models directory"""
    tree_content = "# Complete Models Directory Structure\n\n```\nmodels\n"

    # Get all top-level items, sorted
    top_items = sorted(os.listdir(models_dir))

    # Process each top-level item
    for i, item in enumerate(top_items):
        item_path = os.path.join(models_dir, item)
        is_last = i == len(top_items) - 1

        # Skip hidden files
        if item.startswith("."):
            continue

        # Handle model_registry.json separately
        if item == "model_registry.json":
            tree_content += "├── model_registry.json\n"
            continue

        # Handle optuna_logs directory
        elif item == "optuna_logs":
            tree_content += "├── optuna_logs\n"
            log_files = sorted(os.listdir(item_path))
            for j, log_file in enumerate(log_files):
                if j == len(log_files) - 1:
                    tree_content += "│   └── " + log_file + "\n"
                else:
                    tree_content += "│   ├── " + log_file + "\n"

        # Handle optuna_study directory
        elif item == "optuna_study":
            tree_content += "├── optuna_study\n"
            study_files = sorted(os.listdir(item_path))
            for j, study_file in enumerate(study_files):
                if j == len(study_files) - 1:
                    tree_content += "│   └── " + study_file + "\n"
                else:
                    tree_content += "│   ├── " + study_file + "\n"

        # Handle processed_data directory - special formatting
        elif item == "processed_data":
            tree_content += "├── processed_data\n"
            model_dirs = sorted(os.listdir(item_path))
            for j, model_dir in enumerate(model_dirs):
                model_path = os.path.join(item_path, model_dir)
                if not os.path.isdir(model_path):
                    continue

                # Add model directory
                if j == len(model_dirs) - 1:
                    tree_content += "│   └── " + model_dir + "\n"
                    # Files in last model dir
                    model_files = sorted(os.listdir(model_path))
                    for k, file in enumerate(model_files):
                        if k == len(model_files) - 1:
                            tree_content += "        └── " + file + "\n"
                        else:
                            tree_content += "        ├── " + file + "\n"
                else:
                    tree_content += "│   ├── " + model_dir + "\n"
                    # Files in non-last model dir
                    model_files = sorted(os.listdir(model_path))
                    for k, file in enumerate(model_files):
                        if k == len(model_files) - 1:
                            tree_content += "│   │   └── " + file + "\n"
                        else:
                            tree_content += "│   │   ├── " + file + "\n"

        # Handle trial directories
        elif item.startswith("trial_"):
            if is_last:
                tree_content += "└── " + item + "\n"
                tree_content += "    └── ...\n"
            else:
                tree_content += "├── " + item + "\n"
                tree_content += "│   └── ...\n"

        # Other items
        else:
            if is_last:
                tree_content += "└── " + item + "\n"
            else:
                tree_content += "├── " + item + "\n"

    tree_content += "```\n\n"
    return tree_content


def build_trial_directory_tree(models_dir, trial_num):
    """Create a markdown-formatted directory tree for a specific trial"""
    trial_dir = f"trial_{trial_num}"
    trial_path = os.path.join(models_dir, trial_dir)

    if not os.path.exists(trial_path):
        return f"Error: Trial directory '{trial_dir}' not found in {models_dir}", []

    tree_content = f"# Trial {trial_num} Directory Structure\n\n```\n{trial_dir}\n"
    empty_dirs = []

    # Keep track of directories to add and files for each directory
    dir_structure = {}
    dir_empty = {}

    # Walk the directory tree
    for root, dirs, files in sorted(os.walk(trial_path)):
        rel_path = os.path.relpath(root, trial_path)

        # Skip the root level (we've already added it)
        if rel_path == ".":
            continue

        # Keep track of empty dirs
        is_empty = len(dirs) == 0 and len(files) == 0
        if is_empty:
            empty_dirs.append(os.path.relpath(root, models_dir))
            dir_empty[rel_path] = True
        else:
            dir_empty[rel_path] = False

        # Store dir structure
        dir_structure[rel_path] = sorted(files)

    # Function to build tree recursively
    def build_subtree(path, prefix=""):
        components = path.split(os.sep)
        current_depth = len(components)

        if path == ".":
            result = []
            # Get top-level directories
            top_dirs = [d for d in dir_structure.keys() if len(d.split(os.sep)) == 1]

            for i, dir_name in enumerate(sorted(top_dirs)):
                is_last = i == len(top_dirs) - 1
                empty_marker = " [EMPTY]" if dir_empty.get(dir_name, False) else ""

                if is_last:
                    result.append(f"├── {dir_name}{empty_marker}")
                    # Process files
                    files = dir_structure.get(dir_name, [])
                    for j, file in enumerate(files):
                        if j == len(files) - 1:
                            result.append(f"│   └── {file}")
                        else:
                            result.append(f"│   ├── {file}")

                    # Process subdirectories
                    subdirs = [
                        d
                        for d in dir_structure.keys()
                        if d.startswith(dir_name + os.sep)
                        and len(d.split(os.sep)) == len(dir_name.split(os.sep)) + 1
                    ]

                    for j, subdir in enumerate(sorted(subdirs)):
                        subdir_name = subdir.split(os.sep)[-1]
                        is_last_subdir = j == len(subdirs) - 1
                        empty_marker = (
                            " [EMPTY]" if dir_empty.get(subdir, False) else ""
                        )

                        if is_last_subdir:
                            result.append(f"│   ├── {subdir_name}{empty_marker}")
                        else:
                            result.append(f"│   ├── {subdir_name}{empty_marker}")

                        # Process files in subdir
                        files = dir_structure.get(subdir, [])
                        for k, file in enumerate(files):
                            if k == len(files) - 1:
                                result.append(f"│   │   └── {file}")
                            else:
                                result.append(f"│   │   ├── {file}")
                else:
                    result.append(f"├── {dir_name}{empty_marker}")
                    # Process files
                    files = dir_structure.get(dir_name, [])
                    for j, file in enumerate(files):
                        if j == len(files) - 1:
                            result.append(f"│   └── {file}")
                        else:
                            result.append(f"│   ├── {file}")

            return result

    # Build tree structure manually based on directory listing
    all_paths = sorted(dir_structure.keys())

    # Manual construction of trial directory tree
    current_level = []
    previous_path = None

    for root, dirs, files in sorted(os.walk(trial_path)):
        rel_path = os.path.relpath(root, trial_path)
        if rel_path == ".":
            continue

        # Calculate level
        level = rel_path.count(os.sep)
        indent = "│   " * level

        # Get directory name
        dir_name = os.path.basename(root)

        # Check if empty
        is_empty = len(dirs) == 0 and len(files) == 0
        empty_marker = " [EMPTY]" if is_empty else ""

        # Add directory to tree
        tree_content += f"{indent[:-4]}├── {dir_name}{empty_marker}\n"

        # Add files
        for i, file in enumerate(sorted(files)):
            is_last_file = i == len(files) - 1 and len(dirs) == 0
            if is_last_file:
                tree_content += f"{indent}└── {file}\n"
            else:
                tree_content += f"{indent}├── {file}\n"

    tree_content += "```\n\n"
    return tree_content, empty_dirs


def create_markdown_note(models_dir, trial_num=None):
    """Create a markdown note with contents of specified files"""
    markdown_content = ""
    empty_dirs = []

    # Add directory tree
    markdown_content += build_directory_tree(models_dir)

    # If a trial is specified, also add its detailed tree
    if trial_num is not None:
        trial_tree, trial_empty_dirs = build_trial_directory_tree(models_dir, trial_num)
        if trial_tree.startswith("Error:"):
            return trial_tree
        markdown_content += trial_tree
        empty_dirs.extend(trial_empty_dirs)

    # Files to always include regardless of trial
    base_files = [
        os.path.join(models_dir, "model_registry.json"),
        *glob.glob(os.path.join(models_dir, "optuna_study", "best_params_summary.txt")),
        *glob.glob(os.path.join(models_dir, "optuna_study", "study_results.json")),
        *glob.glob(
            os.path.join(models_dir, "processed_data", "*", "class_mapping.txt")
        ),
        *glob.glob(os.path.join(models_dir, "optuna_logs", "*.log")),
    ]

    # Filter out non-existent files
    base_files = [f for f in base_files if os.path.exists(f)]

    # If a trial is specified, add its logs and other content
    if trial_num is not None:
        trial_dir = os.path.join(models_dir, f"trial_{trial_num}")
        trial_files = glob.glob(os.path.join(trial_dir, "**", "*.log"), recursive=True)
        files_to_process = base_files + trial_files
    else:
        # Just get all log files if no trial specified
        all_logs = glob.glob(os.path.join(models_dir, "**", "*.log"), recursive=True)
        files_to_process = base_files + all_logs

    # Remove duplicates and sort
    files_to_process = sorted(set(files_to_process))

    # Extensions to exclude content from
    exclude_extensions = [".csv", ".pth", ".db"]

    # Add file contents section
    markdown_content += "# File Contents\n\n"

    # Process each file
    for file_path in files_to_process:
        rel_path = os.path.relpath(file_path, models_dir)

        # Add file header
        markdown_content += f"## {rel_path}\n\n"

        # Skip content extraction for specified file types
        if any(file_path.endswith(ext) for ext in exclude_extensions):
            markdown_content += "*File content not included due to file type*\n\n"
            continue

        try:
            # Check if it's a JSON file
            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                markdown_content += "```json\n"
                markdown_content += json.dumps(data, indent=2)
                markdown_content += "\n```\n\n"
            else:
                # Regular text file
                with open(file_path, "r") as f:
                    content = f.read()
                markdown_content += "```\n"
                markdown_content += content
                markdown_content += "\n```\n\n"
        except Exception as e:
            markdown_content += f"*Error reading file: {str(e)}*\n\n"

    # Add empty directories section
    if empty_dirs:
        markdown_content += "# Empty Directories Summary\n\n"
        for dir_path in sorted(empty_dirs):
            markdown_content += f"- `{dir_path}`\n"
        markdown_content += "\n"

    return markdown_content


def main():
    parser = argparse.ArgumentParser(description="Extract model data to markdown")
    parser.add_argument(
        "--trial", type=int, help="Trial number to focus on (e.g., 0 for trial_0)"
    )
    parser.add_argument(
        "--output", type=str, default="model_data.md", help="Output markdown file name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output file (defaults to working directory)",
    )

    args = parser.parse_args()

    # Get models directory
    models_dir = get_models_dir()

    # Generate markdown content
    markdown_content = create_markdown_note(models_dir, args.trial)

    # Determine output path
    if args.output_dir:
        output_dir = args.output_dir
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, args.output)

    # Write output file
    with open(output_path, "w") as f:
        f.write(markdown_content)

    print(f"Markdown file created: {output_path}")


if __name__ == "__main__":
    main()

```

---

## Summary

Total Python files: 1
