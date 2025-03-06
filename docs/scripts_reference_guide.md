# DeepCropDx-PyTorch Scripts Reference Guide

This document provides a comprehensive reference for utility scripts available in the project.

## Table of Contents

- [compile_code_complete.py](#compile_code_completepy)
- [compile_code_compact.py](#compile_code_compactpy)
- [extract_model_snapshot.py](#extract_model_snapshotpy)
- [reset_project.py](#reset_projectpy)

## compile_code_complete.py

### Overview

A script that compiles all Python files in the project into a single markdown document, preserving all docstrings and comments. This is useful for sharing code with LLMs for detailed analysis.

### Location

`/scripts/compile_code_complete.py`

### Arguments

| Argument | Flag | Default | Description |
|----------|------|---------|-------------|
| Directory | `--directory`, `-d` | Project root directory | Directory to scan for Python files |
| Output | `--output`, `-o` | `project_docs/project_code_full.md` | Output markdown file |
| Ignore directories | `--ignore`, `-i` | `.git`, `.github`, `__pycache__`, etc. | Directories to ignore |
| Ignore files | `--ignore-files` | `compile_code_complete.py`, etc. | Specific files to ignore |
| Ignore paths | `--ignore-paths` | `src/app`, `project_docs` | Directory paths to ignore |
| Skip minimal | `--skip-minimal` | False | Skip minimal files like empty `__init__.py` |
| No project structure | `--no-project-structure` | False | Exclude project structure from output |

### Example Usage

```bash
# Basic usage with default settings
python scripts/compile_code_complete.py

# Specify output file and skip minimal files
python scripts/compile_code_complete.py --output custom_output.md --skip-minimal

# Ignore additional directories
python scripts/compile_code_complete.py --ignore data logs temp
```

### Output

The script generates a markdown file with:

- Project structure tree
- Table of contents with links to each file
- Complete code content for each Python file, preserving all comments and docstrings
- Summary statistics about the code

## compile_code_compact.py

### Overview

A script that compiles Python files into a compact markdown document by optionally removing docstrings and comments. This produces a more concise code overview suitable for LLM analysis when token count is a concern.

### Location

`/scripts/compile_code_compact.py`

### Arguments

| Argument | Flag | Default | Description |
|----------|------|---------|-------------|
| Directory | `--directory`, `-d` | Project root directory | Directory to scan for Python files |
| Output | `--output`, `-o` | `project_docs/project_code.md` | Output markdown file |
| Ignore directories | `--ignore`, `-i` | `.git`, `.github`, `__pycache__`, etc. | Directories to ignore |
| Ignore files | `--ignore-files` | `compile_code_compact.py`, etc. | Specific files to ignore |
| Ignore paths | `--ignore-paths` | `src/app`, `project_docs` | Directory paths to ignore |
| Keep docstrings | `--keep-docstrings` | False | Keep docstrings in the output |
| Keep comments | `--keep-comments` | False | Keep comments in the output |
| Include minimal | `--include-minimal` | False | Include minimal files like empty `__init__.py` |
| No project structure | `--no-project-structure` | False | Exclude project structure from output |

### Example Usage

```bash
# Basic usage with default settings (removes docstrings and comments)
python scripts/compile_code_compact.py
s

# Keep docstrings but remove comments
python scripts/compile_code_compact.py --keep-docstrings

# Keep both docstrings and comments
python scripts/compile_code_compact.py --keep-docstrings --keep-comments

# Customize output location
python scripts/compile_code_compact.py --output ./my_code_overview.md
```

### Output

The script generates a markdown file with:

- Project structure tree
- Table of contents with links to each file
- Optimized code content for each Python file (docstrings and comments removed by default)
- Only includes substantive files by default (skips minimal files unless specified)

## extract_model_snapshot.py

### Overview

A utility script designed to extract and document key information from model trials into a structured markdown document. It captures directory structures, log files, configuration files, visualizations, and other critical data points for easy review or sharing.

### Location

`/scripts/extract_model_snapshot.py`

### Key Features

- Creates a visual representation of your project's directory structure
- Extracts content from JSON files, log files, and other text-based configuration files
- Embeds visualization images directly in the markdown output
- Identifies and highlights empty directories
- Supports focused extraction on specific trial directories
- Customizable output location and filename

### Arguments

| Argument | Flag | Default | Description |
|----------|------|---------|-------------|
| Trial number | `--trial` | None | Trial number to focus on (e.g., 0 for trial_0) |
| Output file | `--output` | "model_snapshot.md" | Output markdown file name |
| Output directory | `--output-dir` | Current directory | Directory to save output file |
| Include visualizations | `--include-visualizations` | False | Include visualization images in the output |

### Function Reference

#### `get_models_dir()`

- **Purpose**: Identifies the location of the models directory from the script's location
- **Returns**: Absolute path to the models directory
- **Behavior**: Assumes the script is located in a `scripts` folder that's one level below the project root

#### `build_directory_tree(models_dir)`

- **Purpose**: Creates a visual tree representation of the entire models directory
- **Parameters**: `models_dir` - Path to the models directory
- **Returns**: Markdown-formatted string representing the directory structure
- **Behavior**: Shows all top-level folders and their immediate contents

#### `build_trial_directory_tree(models_dir, trial_num)`

- **Purpose**: Creates a detailed tree representation of a specific trial directory
- **Parameters**:
  - `models_dir` - Path to the models directory
  - `trial_num` - Trial number to focus on
- **Returns**: Tuple of (tree_content, empty_dirs)
- **Behavior**: Shows the complete structure of the specified trial, marking empty directories

#### `embed_image(image_path)`

- **Purpose**: Embeds an image file as base64 in markdown
- **Parameters**: `image_path` - Path to the image file
- **Returns**: Markdown string with embedded base64 image
- **Behavior**: Converts the image to base64 and formats it for inline display in markdown

#### `create_markdown_note(models_dir, trial_num=None, include_visualizations=False)`

- **Purpose**: Creates the complete markdown document with tree and file contents
- **Parameters**:
  - `models_dir` - Path to the models directory
  - `trial_num` - Optional trial number to focus on
  - `include_visualizations` - Whether to include visualization images
- **Returns**: Complete markdown content as a string
- **Behavior**:
  - Always includes the directory tree and key files
  - When trial_num is specified, includes detailed information about that trial
  - Extracts content from files based on their extensions
  - Embeds images when include_visualizations is True

### Example Usage

```bash
# Basic usage - extract all project data to a file named model_snapshot.md
python scripts/extract_model_snapshot.py

# Focus on a specific trial
python scripts/extract_model_snapshot.py --trial 0

# Include visualizations in the output
python scripts/extract_model_snapshot.py --trial 0 --include-visualizations

# Customize output filename and directory
python scripts/extract_model_snapshot.py --output summary.md --output-dir ./reports

# Complete example with all options
python scripts/extract_model_snapshot.py --trial 0 --include-visualizations --output trial0_report.md --output-dir "/path/to/your/reports"
```

### Output Structure

The generated markdown file contains these main sections:

1. **Complete Models Directory Structure**: A tree view of the entire models directory
2. **Trial X Directory Structure** (if a trial is specified): Detailed tree view of the trial directory
3. **File Contents**: Content of all relevant files, organized by file path
4. **Empty Directories Summary**: List of all empty directories found

### Files Included in the Extract

The script automatically extracts the following files:

- `model_registry.json`
- `optuna_study/best_params_summary.txt`
- `optuna_study/study_results.json`
- All `class_mapping.txt` files from model directories
- All log files (*.log) from the models directory and its subdirectories
- When include_visualizations is enabled, all PNG, JPG, JPEG, and SVG files in the trial directory

### Troubleshooting

1. **"Models directory not found"**: Ensure the script is being run from within the `scripts` directory or from the project root
2. **Missing file contents**: Check if the file exists and is readable. The script skips extraction for certain file types (CSV, PTH, DB)
3. **Trial directory not found**: Verify the trial number exists in your project structure
4. **Permission errors**: Ensure you have read access to all project files and write access to the output directory
5. **Large output file**: If including visualizations, be aware that the output file size may be large due to base64-encoded images

## reset_project.py

### Overview

A utility script to clean and reset the project environment by removing generated files, logs, and caches to start with a clean slate for a new training run.

### Location

`/scripts/reset_project.py`

### Arguments

| Argument | Flag | Default | Description |
|----------|------|---------|-------------|
| All | `--all` | True if no other flags specified | Remove all generated files |
| Models | `--models` | False | Remove only model files and directories |
| Reports | `--reports` | False | Remove only report files and directories |
| Logs | `--logs` | False | Remove only log files |
| Python cache | `--pycache` | False | Remove only **pycache** directories |
| Trials | `--trials` | False | Remove only trials directory contents |
| Keep registry | `--keep-registry` | False | Keep the model registry file when cleaning models |
| Dry run | `--dry-run` | False | Show what would be removed without actually removing |
| Confirm | `--confirm` | False | Skip confirmation prompt (use with caution) |

### Example Usage

```bash
# Reset everything (will prompt for confirmation)
python scripts/reset_project.py

# Reset only model files but keep the registry
python scripts/reset_project.py --models --keep-registry

# Clean python cache files only
python scripts/reset_project.py --pycache

# Perform a dry run to see what would be deleted
python scripts/reset_project.py --all --dry-run

# Reset everything without confirmation (be careful!)
python scripts/reset_project.py --all --confirm
```

### Safety Features

The script includes several safety mechanisms:

1. Confirmation prompt before deletion (unless `--confirm` is used)
2. Dry-run option to preview changes
3. Internal checks to prevent accidental deletion of source code or raw data

### Areas Affected

When run with `--all` or their specific flags, the script cleans:

- **Models directory**: All trained models, checkpoints, and associated files
- **Reports directory**: All evaluation reports, metrics, and visualizations
- **Logs**: All log files throughout the project
- **Python cache**: All `__pycache__` directories
- **Trials directory**: All trial-related files and directories
