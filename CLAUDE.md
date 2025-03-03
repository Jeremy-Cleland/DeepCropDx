# CLAUDE.md - Crop Disease Detection Project Guidelines

## Build & Run Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training
```bash
python -m src.training.train --data_dir data/raw --output_dir models/experiment1 --model efficientnet --pretrained
```

### Evaluation
```bash
python -m src.training.evaluate --model_path models/experiment1/best_model.pth --data_dir data/raw --visualize
```

### Web App
```bash
python run_app.py --model models/experiment1/best_model.pth --host 127.0.0.1 --port 5000
```

### Cleanup
```bash
./cleanup.py --dry-run  # Show what would be deleted
./cleanup.py --models   # Delete model files
```

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports second
- Local application imports third
- Sort alphabetically within each group

### Formatting
- Max line length: 79 characters (PEP8)
- Use 4 spaces for indentation

### Naming
- snake_case for functions and variables
- PascalCase for classes
- UPPERCASE for constants

### Error Handling
- Use specific exception classes
- Handle exceptions with appropriate context
- Use try/except blocks for specific error cases

### Documentation
- Docstrings for all functions
- Include Args/Returns sections in docstrings
- Use """triple double quotes""" for docstrings