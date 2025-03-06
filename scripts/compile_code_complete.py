#!/usr/bin/env python3
"""
Script to compile all Python files in a project into a single markdown document.
This is the FULL version that preserves all docstrings and comments.
This is useful for sharing code with LLMs for detailed analysis.
"""

import os
import argparse
import sys
from datetime import datetime

# Add the root directory to the system path to allow importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def find_python_files(
    directory, ignore_dirs=None, ignore_files=None, ignore_paths=None
):
    """
    Find all Python files in the given directory and its subdirectories.

    Args:
        directory (str): The directory to scan for Python files
        ignore_dirs (list): List of directory names to ignore
        ignore_files (list): List of file names to ignore
        ignore_paths (list): List of directory paths to ignore

    Returns:
        list: Sorted list of Python file paths
    """
    if ignore_dirs is None:
        ignore_dirs = [
            ".git",
            ".github",
            "__pycache__",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
            "node_modules",
        ]

    if ignore_files is None:
        ignore_files = [
            "compile_code_complete.py",
            "compile_code_compact.py",
            "extract_model_snapshot.py",
            "reset_project.py",
            "api_client_example.py",
            "cleanup.py",
            "run_app.py",
        ]

    if ignore_paths is None:
        ignore_paths = ["src/app", "project_docs"]

    # Convert ignore_paths to absolute paths for proper comparison
    base_dir = os.path.abspath(directory)
    absolute_ignore_paths = [
        os.path.join(base_dir, p) if not os.path.isabs(p) else p for p in ignore_paths
    ]

    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        # Check if current directory should be skipped
        skip_dir = False
        for path in absolute_ignore_paths:
            if os.path.abspath(root).startswith(path):
                skip_dir = True
                break

        if skip_dir:
            dirs[:] = []  # Clear dirs to prevent further traversal
            continue

        for file in files:
            if file.endswith(".py") and file not in ignore_files:
                python_files.append(os.path.join(root, file))

    return sorted(python_files)


def get_relative_path(file_path, base_dir):
    """
    Get the path relative to the base directory.

    Args:
        file_path (str): The absolute file path
        base_dir (str): The base directory path

    Returns:
        str: The relative path
    """
    return os.path.relpath(file_path, base_dir)


def is_minimal_file(code):
    """
    Check if a file is minimal (like empty __init__.py)

    Args:
        code (str): The file content

    Returns:
        bool: True if the file is minimal, False otherwise
    """
    # Remove docstrings and comments to check actual code content
    import re

    stripped_code = re.sub(
        r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", code
    )  # Remove docstrings
    stripped_code = re.sub(
        r"#.*$", "", stripped_code, flags=re.MULTILINE
    )  # Remove comments
    stripped_code = stripped_code.strip()
    return len(stripped_code) < 50  # Arbitrary threshold


def generate_directory_tree(
    directory,
    prefix="",
    ignore_dirs=None,
    ignore_files=None,
    ignore_paths=None,
    max_depth=None,
    trial_limit=2,
):
    """
    Recursively generate a tree representation of the directory structure

    Args:
        directory (str): The directory to generate the tree for
        prefix (str): Prefix for the current line (used in recursion)
        ignore_dirs (list): List of directory names to ignore
        ignore_files (list): List of filenames to ignore
        ignore_paths (list): List of directory paths to ignore
        max_depth (int): Maximum depth to traverse
        trial_limit (int): Maximum number of trial directories to display

    Returns:
        list: Lines representing the directory tree
    """
    if ignore_dirs is None:
        ignore_dirs = [
            ".git",
            ".github",
            "__pycache__",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
            "node_modules",
        ]

    if ignore_files is None:
        ignore_files = [
            "compile_code_complete.py",
            "compile_code_compact.py",
            "extract_model_snapshot.py",
            "reset_project.py",
            "api_client_example.py",
            ".DS_Store",
            ".gitattributes",
            ".gitignore",
            ".pylintrc",
            "cleanup.py",
            "run_app.py",
        ]

    if ignore_paths is None:
        ignore_paths = []

    # Convert ignore_paths to absolute paths for proper comparison
    base_dir = os.path.abspath(directory)
    absolute_ignore_paths = [
        os.path.join(base_dir, p) if not os.path.isabs(p) else p for p in ignore_paths
    ]

    # Check if current directory should be skipped based on absolute path
    if any(
        os.path.abspath(directory).startswith(path) for path in absolute_ignore_paths
    ):
        return []

    # Check if we've reached max depth
    current_depth = len(prefix.split("│")) - 1
    if max_depth is not None and current_depth >= max_depth:
        return ["├── ..."]

    lines = []

    try:
        dir_contents = sorted(os.listdir(directory))
    except (PermissionError, FileNotFoundError):
        return []

    # Filter out ignored directories and files
    filtered_contents = [
        item
        for item in dir_contents
        if not (
            (os.path.isdir(os.path.join(directory, item)) and item in ignore_dirs)
            or item in ignore_files
        )
    ]

    # Handle trial directories limitation
    trial_dirs = []
    non_trial_items = []

    for item in filtered_contents:
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path) and item.startswith("trial_"):
            trial_dirs.append(item)
        else:
            non_trial_items.append(item)

    # If we have more trial directories than the limit, only include the first few
    if len(trial_dirs) > trial_limit:
        # Sort to ensure we get trial_0, trial_1, etc.
        sorted_trials = sorted(trial_dirs)
        # Take the first trial_limit trials
        limited_trials = sorted_trials[:trial_limit]
        display_items = non_trial_items + limited_trials
        has_extra_trials = True
    else:
        display_items = non_trial_items + trial_dirs
        has_extra_trials = False

    # Count directories and files for processing
    dirs_and_files = []
    for item in display_items:
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            dirs_and_files.append((item, True))  # is a directory
        else:
            dirs_and_files.append((item, False))  # is a file

    # Process each item
    for i, (item, is_dir) in enumerate(dirs_and_files):
        is_last_item = i == len(dirs_and_files) - 1 and not has_extra_trials

        if is_last_item:
            branch = "└── "
            new_prefix = prefix + "    "
        else:
            branch = "├── "
            new_prefix = prefix + "│   "

        full_path = os.path.join(directory, item)

        # Add the current item to the tree
        lines.append(f"{prefix}{branch}{item}")

        # Recursively process directories
        if is_dir:
            try:
                dir_items = os.listdir(full_path)
                if len(dir_items) > 50:
                    lines.append(f"{new_prefix}├── [{len(dir_items)} items...]")
                else:
                    lines.extend(
                        generate_directory_tree(
                            full_path,
                            new_prefix,
                            ignore_dirs,
                            ignore_files,
                            ignore_paths,
                            max_depth,
                            trial_limit,
                        )
                    )
            except (PermissionError, FileNotFoundError):
                lines.append(f"{new_prefix}├── [access error]")

    # Add ellipsis for additional trial directories
    if has_extra_trials:
        if dirs_and_files:
            branch = "└── "
        else:
            branch = "├── "
        lines.append(
            f"{prefix}{branch}[... {len(trial_dirs) - trial_limit} more trials ...]"
        )

    return lines


def generate_file_tree(python_files, base_dir, include_project_structure=True):
    """
    Generate a visual tree representation of the project structure

    Args:
        python_files (list): List of Python files
        base_dir (str): Base directory
        include_project_structure (bool): Whether to include the full project structure

    Returns:
        str: String representation of the tree
    """
    tree_lines = []
    tree_lines.append("## Project Structure")
    tree_lines.append("```")

    # Add root directory
    root_name = os.path.basename(base_dir)
    tree_lines.append(root_name)

    # Generate the directory tree
    if include_project_structure:
        # Additional files to ignore in tree display
        ignore_files = [
            "compile_code_complete.py",
            "compile_code_compact.py",
            "extract_model_snapshot.py",
            "reset_project.py",
            "api_client_example.py",
            ".DS_Store",
            ".gitattributes",
            ".gitignore",
            ".pylintrc",
            "cleanup.py",
            "run_app.py",
        ]

        # Use a reasonable max_depth to avoid overly detailed trees
        ignore_paths = ["project_docs", "src/app"]
        directory_tree = generate_directory_tree(
            base_dir,
            prefix="",
            ignore_dirs=None,
            ignore_files=ignore_files,
            ignore_paths=ignore_paths,
            max_depth=6,  # Reasonable depth to prevent overly large tree
            trial_limit=2,  # Show only 2 trial directories
        )
        tree_lines.extend(directory_tree)

    tree_lines.append("```")
    return "\n".join(tree_lines)


def create_markdown(
    python_files,
    base_dir,
    output_file,
    skip_minimal=False,
    include_project_structure=True,
):
    """
    Create a markdown document from the Python files.

    Args:
        python_files (list): List of Python file paths
        base_dir (str): Base directory of the project
        output_file (str): Output markdown file path
        skip_minimal (bool): Whether to skip minimal files
        include_project_structure (bool): Whether to include project structure
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Project Code Overview (Complete Version)\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        # Add file tree with project structure
        f.write(
            generate_file_tree(python_files, base_dir, include_project_structure)
            + "\n\n"
        )

        # Filter minimal files if requested
        file_list = []
        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as code_file:
                    code = code_file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as code_file:
                        code = code_file.read()
                except:
                    code = "# Error reading file"

            # Check if file is minimal and should be skipped
            if skip_minimal and is_minimal_file(code):
                continue

            file_list.append((file_path, code))

        # Generate table of contents
        f.write("## Table of Contents\n\n")
        for file_path, _ in file_list:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")
            f.write(f"- [{rel_path}](#{anchor})\n")

        f.write("\n## Code Files\n\n")

        # Write each file with its content
        for file_path, code in file_list:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")

            f.write(f"### {rel_path}\n\n")
            f.write("```python\n")
            f.write(code)
            f.write("\n```\n\n")
            f.write("---\n\n")

        # Add summary information
        f.write(f"## Summary\n\n")
        f.write(f"Total Python files: {len(file_list)}\n\n")


def main():
    """Main entry point of the script"""
    parser = argparse.ArgumentParser(
        description="Compile Python files into a complete markdown document with all docstrings and comments. Useful for LLM analysis."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__))),
        help="Directory to scan for Python files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "reports",
            "project_code_full.md",
        ),
        help="Output markdown file",
    )
    parser.add_argument(
        "--ignore",
        "-i",
        type=str,
        nargs="+",
        default=[
            ".git",
            ".github",
            "__pycache__",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
            "node_modules",
        ],
        help="Directories to ignore",
    )
    parser.add_argument(
        "--ignore-files",
        type=str,
        nargs="+",
        default=[
            "compile_code_complete.py",
            "compile_code_compact.py",
            "extract_model_snapshot.py",
            "reset_project.py",
            "api_client_example.py",
            "cleanup.py",
            "run_app.py",
        ],
        help="Specific files to ignore",
    )
    parser.add_argument(
        "--ignore-paths",
        type=str,
        nargs="+",
        default=["src/app", "project_docs"],
        help="Directory paths to ignore (relative to base directory)",
    )
    parser.add_argument(
        "--skip-minimal",
        action="store_true",
        help="Skip minimal files like empty __init__.py",
    )
    parser.add_argument(
        "--no-project-structure",
        action="store_true",
        help="Exclude project structure from the output",
    )

    args = parser.parse_args()

    base_dir = os.path.abspath(args.directory)
    python_files = find_python_files(
        base_dir, args.ignore, args.ignore_files, args.ignore_paths
    )

    if not python_files:
        print(f"No Python files found in {base_dir}")
        return

    # Create the reports directory if it doesn't exist
    reports_dir = os.path.dirname(args.output)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created reports directory: {reports_dir}")

    create_markdown(
        python_files,
        base_dir,
        args.output,
        skip_minimal=args.skip_minimal,
        include_project_structure=not args.no_project_structure,
    )

    print(f"Complete markdown document created at {args.output}")
    print(f"Found {len(python_files)} Python files")


if __name__ == "__main__":
    main()
