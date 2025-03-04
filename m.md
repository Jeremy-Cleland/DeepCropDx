plant_disease_detection/
├── config/
│   ├── config.yaml          # Global configuration (e.g., dataset paths, logging settings)
│   └── model_params.yaml    # Hyperparameters for each model
├── data/
│   ├── raw/                 # Original unmodified datasets
│   └── processed/           # Preprocessed and augmented data ready for training
├── experiments/
│   ├── logs/                # Training logs, error reports, and tensorboard logs
│   └── results/             # Model outputs, evaluation metrics, and benchmark summaries
├── models/
│   ├── base_model.py        # Base classes or common functionality (e.g., shared layers)
│   ├── resnet18.py          # ResNet-18 architecture (can extend from torchvision.models)
│   ├── resnet50.py          # ResNet-50 architecture
│   ├── resnet_tension.py    # Customized ResNet with a tension mechanism
│   ├── mobilenet_v2.py      # MobileNet-V2 architecture
│   ├── mobilenet_v3.py      # MobileNet-V3 architecture
│   └── mobilenet_tension.py # MobileNet with a tension mechanism
├── scripts/
│   ├── train.py             # Script to train a given model
│   ├── evaluate.py          # Script to evaluate a trained model on validation/test sets
│   ├── benchmark.py         # Script to run benchmark experiments across models
│   └── predict.py           # Script for inference on new data
├── utils/
│   ├── data_loader.py       # Data loading and augmentation routines
│   ├── metrics.py           # Custom evaluation metrics (e.g., accuracy, F1 score)
│   └── helpers.py           # Utility functions (logging, configuration parsing, etc.)
├── requirements.txt         # All Python dependencies
├── README.md                # Project overview, instructions, and setup details
└── setup.py                 # Installation script (if you plan to distribute or package your code)
