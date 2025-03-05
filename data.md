# Handling Plant Disease Image Dataset in Crop Disease Detection

## Dataset Overview

This project uses a plant disease image dataset containing various crop species (Tomato, Potato, Pepper) affected by different diseases. The dataset includes both healthy and diseased plant images, making it suitable for training models to detect and classify plant diseases.

## Data Organization

The dataset is organized as follows:

- **Raw Images**: Located in `data/raw/` directory, organized by disease categories
- **Processed Data**: Split into train, validation, and test sets with CSV files in `models/efficientnet_b0_v1/processed_data/`
  - `train_set.csv`: Images for model training
  - `val_set.csv`: Images for validation during training
  - `test_set.csv`: Images for final model evaluation

Each CSV file contains two columns:

- `image_path`: Relative path to the image file
- `label`: Numeric class label

## Class Distribution

The dataset contains 15 classes covering different crop diseases:

| Class ID | Class Name |
|----------|------------|
| 0 | Tomato_healthy |
| 1 | Potato___Early_blight |
| 2 | Tomato__Tomato_YellowLeaf__Curl_Virus |
| 3 | Tomato_Early_blight |
| 4 | Tomato__Target_Spot |
| 5 | Potato___Late_blight |
| 6 | Tomato_Leaf_Mold |
| 7 | Tomato_Spider_mites_Two_spotted_spider_mite |
| 8 | Tomato_Septoria_leaf_spot |
| 9 | Tomato__Tomato_mosaic_virus |
| 10 | Pepper__bell___Bacterial_spot |
| 11 | Tomato_Bacterial_spot |
| 12 | Tomato_Late_blight |
| 13 | Pepper__bell___healthy |
| 14 | Potato___healthy |

## Data Preprocessing

Before feeding images into the model, the following preprocessing steps are applied:

1. **Resizing**: All images are resized to a standard size (typically 224x224 pixels for EfficientNet-B0)
2. **Normalization**: Pixel values are normalized to the range expected by the model
3. **Augmentation**: For training data, augmentation techniques like rotation, flipping, and brightness adjustment may be applied to increase dataset variety

## Loading the Data

The dataset can be loaded using the CSV files. Example code for loading and preprocessing:

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_dataset(csv_path, batch_size=32, is_training=False):
    df = pd.read_csv(csv_path)
    
    def preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0  # Normalize to [0,1]
        
        return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, df['label'].values))
    dataset = dataset.map(preprocess_image)
    
    if is_training:
        # Add augmentation for training data
        dataset = dataset.map(apply_augmentation)
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def apply_augmentation(image, label):
    # Apply random augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label
```

## Dataset Statistics

- **Total Images**: Varies by split (train/validation/test)
- **Image Format**: JPG files
- **Image Size**: Variable (resized during preprocessing)
- **Color Channels**: RGB (3 channels)

## Using the Dataset for Model Training

To use this dataset for training a crop disease detection model:

1. Load the training and validation datasets using the provided CSV files
2. Apply appropriate preprocessing and augmentation
3. Train the model on the training dataset while monitoring performance on the validation set
4. Evaluate final model performance on the test set

## Data Privacy and Ethics

The plant disease dataset used in this project is intended for research and educational purposes only. When using or distributing this dataset, please:

- Respect any original dataset licenses
- Cite the original dataset sources
- Use the models trained on this data responsibly
