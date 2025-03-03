#!/usr/bin/env python3
"""
Script to compile all Python files in a project into a single markdown document.
This is useful for sharing code with LLMs for analysis.
"""

import os
import argparse
from datetime import datetime


def find_python_files(directory, ignore_dirs=None):
    """Find all Python files in the given directory and its subdirectories."""
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
        ]

    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return sorted(python_files)


def get_relative_path(file_path, base_dir):
    """Get the path relative to the base directory."""
    return os.path.relpath(file_path, base_dir)


def create_markdown(python_files, base_dir, output_file):
    """Create a markdown document from the Python files."""
    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Project Code Overview\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("## Table of Contents\n\n")

        # Generate table of contents
        for file_path in python_files:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")
            f.write(f"- [{rel_path}](#{anchor})\n")

        f.write("\n## Code Files\n\n")

        # Write each file with its content
        for file_path in python_files:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")

            f.write(f"### {rel_path}\n\n")
            f.write("```python\n")

            try:
                with open(file_path, "r", encoding="utf-8") as code_file:
                    f.write(code_file.read())
            except UnicodeDecodeError:
                try:
                    # Try with a different encoding if UTF-8 fails
                    with open(file_path, "r", encoding="latin-1") as code_file:
                        f.write(code_file.read())
                except Exception as e:
                    f.write(f"# Error reading file: {str(e)}\n")
            except Exception as e:
                f.write(f"# Error reading file: {str(e)}\n")

            f.write("```\n\n")
            f.write("---\n\n")

        # Add summary information
        f.write(f"## Summary\n\n")
        f.write(f"Total Python files: {len(python_files)}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compile Python files into a markdown document."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Directory to scan for Python files (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="project_code.md",
        help="Output markdown file (default: project_code.md)",
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
        ],
        help="Directories to ignore (default: .git, __pycache__, venv, etc.)",
    )

    args = parser.parse_args()

    base_dir = os.path.abspath(args.directory)
    python_files = find_python_files(base_dir, args.ignore)

    if not python_files:
        print(f"No Python files found in {base_dir}")
        return

    create_markdown(python_files, base_dir, args.output)
    print(f"Markdown document created at {args.output}")
    print(f"Found {len(python_files)} Python files")


if __name__ == "__main__":
    main()
