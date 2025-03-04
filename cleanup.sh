#!/bin/bash

# Script to clean up all generated files for a fresh start

echo "Crop Disease Detection - Cleanup Script"
echo "========================================"
echo ""

# Define the root directory as the script's location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Confirm before deletion unless running with --confirm or --dry-run flag
if [ "$1" != "--confirm" ] && [ "$1" != "--dry-run" ] && [ "$1" != "--pycache" ]; then
    echo "⚠️  WARNING: This will PERMANENTLY DELETE files! ⚠️"
    echo ""
    echo "This includes:"
    echo "- All trained models"
    echo "- All evaluation reports and visualizations"
    echo "- All log files and __pycache__ directories"
    echo ""
    read -p "Are you sure you want to proceed? (type 'yes' to confirm): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

# Define a function to remove contents of a directory
remove_dir_contents() {
    if [ -d "$1" ]; then
        if [ "$2" == "--dry-run" ]; then
            echo "Would remove all contents of: $1"
        else
            echo "Removing contents of: $1"
            rm -rf "$1"/*
        fi
    fi
}

# Check if this is a dry run
if [ "$1" == "--dry-run" ]; then
    echo "DRY RUN - No files will be deleted"
    DRY_RUN="--dry-run"
else
    DRY_RUN=""
fi

# Handle different flag options
case $1 in
    --models)
        echo "Removing model files only..."
        remove_dir_contents "$ROOT_DIR/models" $DRY_RUN
        ;;
    --reports)
        echo "Removing report files only..."
        remove_dir_contents "$ROOT_DIR/reports" $DRY_RUN
        ;;
    --logs)
        echo "Removing log files only..."
        remove_dir_contents "$ROOT_DIR/logs" $DRY_RUN
        # Also delete __pycache__ directories in the logs context
        if [ "$DRY_RUN" == "--dry-run" ]; then
            echo "Would remove __pycache__ directories within logs"
        else
            find "$ROOT_DIR/logs" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
            echo "Removed __pycache__ directories within logs"
        fi
        ;;
    --pycache)
        echo "Removing all __pycache__ directories..."
        if [ "$DRY_RUN" == "--dry-run" ]; then
            echo "Would remove __pycache__ directories"
        else
            find "$ROOT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
            echo "Removed __pycache__ directories"
        fi
        ;;
    *)
        # Default: Remove all generated files
        echo "Removing all generated files..."
        # Clean models directory
        remove_dir_contents "$ROOT_DIR/models" $DRY_RUN
        
        # Clean reports directory
        remove_dir_contents "$ROOT_DIR/reports" $DRY_RUN
        
        # Clean logs directory
        remove_dir_contents "$ROOT_DIR/logs" $DRY_RUN
        
        # Remove __pycache__ directories across the entire project
        if [ "$DRY_RUN" == "--dry-run" ]; then
            echo "Would remove __pycache__ directories"
        else
            find "$ROOT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
            echo "Removed __pycache__ directories"
        fi
        ;;
esac

if [ "$DRY_RUN" == "--dry-run" ]; then
    echo ""
    echo "DRY RUN COMPLETE - No files were actually deleted"
else
    echo ""
    echo "Cleanup complete!"
    echo "You can now start with a fresh training run."
fi
