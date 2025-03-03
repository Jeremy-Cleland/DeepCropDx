# Crop Disease Detection API Documentation

This document describes the API endpoints available for programmatic access to the Crop Disease Detection system.

## Base URL

All API endpoints are relative to the base URL of your deployed application:

```
http://[host]:[port]/api
```

For example, if you're running the application locally on the default port, the base URL would be:

```
http://127.0.0.1:5000/api
```

## Authentication

Currently, the API does not require authentication. In a production environment, you should implement an authentication mechanism such as API keys or OAuth.

## Endpoints

### Get Available Models

Retrieve a list of all available models.

**Endpoint:** `/models`  
**Method:** GET  
**Response Format:** JSON

**Example Response:**

```json
[
  {
    "id": "efficientnet_b0_20250301.pth",
    "name": "efficientnet_b0_20250301.pth",
    "num_classes": 10,
    "is_current": true
  },
  {
    "id": "resnet50_20250228.pth",
    "name": "resnet50_20250228.pth",
    "num_classes": 10,
    "is_current": false
  }
]
```

### Set Current Model

Change the currently active model for disease detection.

**Endpoint:** `/set-model/{model_id}`  
**Method:** POST  
**Parameters:**

- `model_id` (path parameter): ID of the model to set as current

**Example Response:**

```json
{
  "success": true,
  "current_model": "resnet50_20250228.pth",
  "model_name": "resnet50_20250228.pth"
}
```

### Get Diagnosis History

Retrieve history of past diagnoses.

**Endpoint:** `/history`  
**Method:** GET  
**Response Format:** JSON

**Example Response:**

```json
[
  {
    "id": "20250301_123045_bacterial_leaf_blight",
    "timestamp": "20250301_123045",
    "date": "2025-03-01 12:30:45",
    "prediction": "bacterial_leaf_blight",
    "confidence": 0.95,
    "model_id": "efficientnet_b0_20250301.pth",
    "image_path": "/static/uploads/plant_20250301_123045.jpg"
  },
  {
    "id": "20250228_153022_healthy",
    "timestamp": "20250228_153022",
    "date": "2025-02-28 15:30:22",
    "prediction": "healthy",
    "confidence": 0.88,
    "model_id": "resnet50_20250228.pth",
    "image_path": "/static/uploads/rice_20250228_153022.jpg"
  }
]
```

### Diagnose Crop Disease

Submit an image for crop disease diagnosis.

**Endpoint:** `/diagnose`  
**Method:** POST  
**Content-Type:** `multipart/form-data`  
**Parameters:**

- `file` (required): Image file (JPG, JPEG, or PNG)
- `model_id` (optional): Specific model to use for diagnosis. If not provided, the current model will be used.
- `crop_type` (optional): Type of crop in the image (e.g., "rice", "tomato")

**Example Request:**

```bash
curl -X POST \
  -F "file=@/path/to/crop_image.jpg" \
  -F "model_id=efficientnet_b0_20250301.pth" \
  -F "crop_type=rice" \
  http://127.0.0.1:5000/api/diagnose
```

**Example Response:**

```json
{
  "status": "success",
  "prediction": {
    "class": "bacterial_leaf_blight",
    "confidence": 0.95,
    "probabilities": {
      "bacterial_leaf_blight": 0.95,
      "brown_spot": 0.03,
      "healthy": 0.01,
      "leaf_blast": 0.01
    }
  },
  "metadata": {
    "model_id": "efficientnet_b0_20250301.pth",
    "model_name": "efficientnet_b0_20250301.pth",
    "timestamp": "20250301_124532",
    "crop_type": "rice",
    "image_filename": "crop_image_20250301_124532.jpg"
  },
  "disease_info": {
    "description": "Bacterial leaf blight is characterized by water-soaked lesions that turn yellow to white as they mature. The disease typically starts at leaf margins and can spread throughout the plant.",
    "treatment": "Remove and destroy infected plant parts. Avoid overhead irrigation as wet foliage promotes disease spread. Apply copper-based bactericides as a preventative measure. Practice crop rotation and use resistant varieties when available."
  }
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: The request was successful
- `400 Bad Request`: The request was malformed or missing required parameters
- `404 Not Found`: The requested resource (model, diagnosis, etc.) does not exist
- `500 Internal Server Error`: An error occurred on the server

Error responses include a JSON object with an `error` field describing the issue:

```json
{
  "status": "error",
  "error": "Model not found"
}
```

## Versioning

This documentation describes version 1.0 of the Crop Disease Detection API. Future versions may include additional endpoints or parameters.

## Rate Limiting

Currently, there are no rate limits on API usage. In a production environment, you should implement rate limiting to prevent abuse.

## Contact

For issues, suggestions, or feature requests, please contact the development team or submit an issue on the project's GitHub repository.
