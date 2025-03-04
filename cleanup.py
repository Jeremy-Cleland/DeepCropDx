#!/usr/bin/env python3
"""
Cleanup script to remove all generated model files, logs, and other artifacts,
including __pycache__ directories, for starting with a clean slate.
"""

import os
import glob
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Clean up the project directory")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove all generated files (models, reports, logs, etc.)",
    )
    parser.add_argument(
        "--models", action="store_true", help="Remove only model files and directories"
    )
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Remove only report files and directories",
    )
    parser.add_argument("--logs", action="store_true", help="Remove only log files")
    parser.add_argument(
        "--pycache", action="store_true", help="Remove only __pycache__ directories"
    )
    parser.add_argument(
        "--keep-registry",
        action="store_true",
        help="Keep the model registry file in the models directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing anything",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt (USE WITH CAUTION)",
    )
    return parser.parse_args()


def is_safe_to_delete(path, root_dir):
    """Check if it's safe to delete this path based on some rules."""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root_dir)

    # Ensure the path is within our project root
    if not abs_path.startswith(abs_root):
        return False

    # Don't delete source code (except __pycache__ folders)
    if os.path.join(abs_root, "src") in abs_path:
        return True if "__pycache__" in abs_path else False

    # Avoid deleting raw data; allow deletion only for processed data
    if os.path.join(abs_root, "data") in abs_path:
        return (
            True
            if ("/processed" in abs_path or "/processed_data" in abs_path)
            else False
        )

    # Otherwise, it's safe to delete models, reports, logs, etc.
    return True


def remove_directory_contents(directory, dry_run=False):
    """Remove all contents of a directory without removing the directory itself."""
    if not os.path.exists(directory):
        return

    contents = glob.glob(os.path.join(directory, "*"))
    for item in contents:
        if os.path.isdir(item):
            if dry_run:
                print(f"Would remove directory: {item}")
            else:
                shutil.rmtree(item)
                print(f"Removed directory: {item}")
        else:
            if dry_run:
                print(f"Would remove file: {item}")
            else:
                os.remove(item)
                print(f"Removed file: {item}")


def remove_model_files(project_root, dry_run=False, keep_registry=False):
    """Remove model files and directories."""
    models_dir = os.path.join(project_root, "models")
    registry_path = os.path.join(models_dir, "model_registry.json")

    if keep_registry and os.path.exists(registry_path):
        # Backup the registry file content
        with open(registry_path, "r") as f:
            registry_data = f.read()
        # Remove all contents in the models directory
        remove_directory_contents(models_dir, dry_run)
        # Restore the registry file
        if dry_run:
            print(f"Would keep registry file: {registry_path}")
        else:
            with open(registry_path, "w") as f:
                f.write(registry_data)
            print(f"Kept registry file: {registry_path}")
    else:
        remove_directory_contents(models_dir, dry_run)


def remove_report_files(project_root, dry_run=False):
    """Remove report files and directories."""
    reports_dir = os.path.join(project_root, "reports")
    remove_directory_contents(reports_dir, dry_run)


def remove_log_files(project_root, dry_run=False):
    """Remove log files from the logs directory and within model directories."""
    # Remove log files in the logs directory
    logs_dir = os.path.join(project_root, "logs")
    if os.path.exists(logs_dir):
        remove_directory_contents(logs_dir, dry_run)

    # Remove log files within model directories (if any)
    for log_file in glob.glob(
        os.path.join(project_root, "models", "**", "logs", "*.log"), recursive=True
    ):
        if is_safe_to_delete(log_file, project_root):
            if dry_run:
                print(f"Would remove log file: {log_file}")
            else:
                os.remove(log_file)
                print(f"Removed log file: {log_file}")


def remove_pycache_dirs(project_root, dry_run=False):
    """Remove all __pycache__ directories recursively in the project."""
    for pycache_dir in glob.glob(
        os.path.join(project_root, "**", "__pycache__"), recursive=True
    ):
        if is_safe_to_delete(pycache_dir, project_root):
            if dry_run:
                print(f"Would remove __pycache__ directory: {pycache_dir}")
            else:
                shutil.rmtree(pycache_dir)
                print(f"Removed __pycache__ directory: {pycache_dir}")


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))

    # If no specific flags are set, default to --all
    if not (args.all or args.models or args.reports or args.logs or args.pycache):
        args.all = True

    if not args.confirm and not args.dry_run:
        print("\n⚠️  WARNING: This will PERMANENTLY DELETE files! ⚠️\n")
        print("This includes:")
        if args.all or args.models:
            print("- All trained models (with option to keep the registry)")
        if args.all or args.reports:
            print("- All evaluation reports and visualizations")
        if args.all or args.logs:
            print("- All log files")
        if args.all or args.pycache:
            print("- All __pycache__ directories")

        confirmation = input(
            "\nAre you sure you want to proceed? (type 'yes' to confirm): "
        )
        if confirmation.lower() != "yes":
            print("Operation cancelled.")
            return

    if args.dry_run:
        print("DRY RUN - no files will be deleted")

    # Process based on provided arguments
    if args.all or args.models:
        remove_model_files(project_root, args.dry_run, args.keep_registry)

    if args.all or args.reports:
        remove_report_files(project_root, args.dry_run)

    if args.all or args.logs:
        remove_log_files(project_root, args.dry_run)

    if args.all or args.pycache:
        remove_pycache_dirs(project_root, args.dry_run)

    if args.dry_run:
        print("\nDRY RUN COMPLETE - No files were actually deleted")
    else:
        print("\nCleanup complete!")
        print("You can now start with a fresh training run.")


if __name__ == "__main__":
    main()
