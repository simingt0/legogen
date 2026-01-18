# LEGO Brick Detection and Identification Model

This project implements a complete pipeline for LEGO brick detection and identification using YOLO (You Only Look Once) object detection combined with the Brickognize API for enhanced brick identification.

## Overview

The system consists of several key components:

1. **YOLO Model Training**: Train custom YOLO models on the B100 LEGO Detection Dataset
2. **Brickognize API Integration**: Enhanced brick identification using the Brickognize API
3. **Complete Inference Pipeline**: End-to-end processing from image input to brick inventory
4. **Evaluation and Validation**: Comprehensive model evaluation tools

## Project Structure

```
model/
├── data/                          # Dataset directory
│   ├── raw/                      # Raw downloaded dataset
│   ├── processed/                # Processed dataset files
│   └── yolo/                     # YOLO format dataset
├── models/                       # Trained model storage
├── scripts/                      # Training and inference scripts
│   ├── prepare_dataset.py        # Dataset preparation
│   ├── train_yolo.py            # YOLO model training
│   ├── evaluate_model.py        # Model evaluation
│   └── inference_pipeline.py    # Complete inference pipeline
├── brickognize/                  # Brickognize API integration
│   └── brickognize_client.py   # API client implementation
├── config/                       # Configuration files
│   ├── training_config.yaml     # Training configuration
│   ├── inference_config.yaml    # Inference configuration
│   └── dataset_template.yaml    # Dataset configuration template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone the repository and navigate to the model directory:**
   ```bash
   cd model/
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   export BRICKOGNIZE_API_KEY="your_brickognize_api_key_here"
   # Optional: Set Kaggle credentials for dataset download
   export KAGGLE_USERNAME="your_kaggle_username"
   export KAGGLE_KEY="your_kaggle_api_key"
   ```

## Dataset Preparation

### Downloading the Dataset

The project uses the [B100 LEGO Detection Dataset](https://www.kaggle.com/datasets/ronanpickell/b100-lego-detection-dataset) from Kaggle.

**Option 1: Automatic Download (requires Kaggle API credentials)**
```bash
python scripts/prepare_dataset.py
```

**Option 2: Manual Download**
1. Download the dataset from Kaggle
2. Extract to `data/raw/`
3. Run the preparation script:
```bash
python scripts/prepare_dataset.py --skip-download
```

### Dataset Structure

The preparation script will convert the dataset to YOLO format:
```
data/yolo/
├── images/
│   ├── train/      # Training images
│   └── val/        # Validation images
├── labels/
│   ├── train/      # Training labels (YOLO format)
│   └── val/        # Validation labels (YOLO format)
└── dataset.yaml    # Dataset configuration
```

## Model Training

### Recommended YOLO Models

For LEGO brick detection, we recommend these YOLO models:

1. **YOLOv8m** (Recommended): Good balance of speed and accuracy
2. **YOLOv8s**: Faster inference, slightly lower accuracy
3. **YOLOv8l**: Higher accuracy, slower inference

### Training Commands

**Basic training:**
```bash
python scripts/train_yolo.py --data-yaml data/yolo/dataset.yaml --model-size m --epochs 100
```

**Advanced training with custom configuration:**
```bash
python scripts/train_yolo.py --data-yaml data/yolo/dataset.yaml \
    --model-size m --yolo-version v8 \
    --epochs 150 --batch-size 16 --learning-rate 0.01 \
    --img-size 640 --device cuda
```

**Compare multiple models:**
```bash
python scripts/train_yolo.py --data-yaml data/yolo/dataset.yaml --compare
```

### Training Configuration

Edit `config/training_config.yaml` to customize:
- Model architecture and hyperparameters
- Data augmentation settings
- Training schedule and optimization
- Validation and checkpoint settings

## Model Evaluation

### Evaluate a trained model:
```bash
python scripts/evaluate_model.py --model-path models/best_model.pt --data-yaml data/yolo/dataset.yaml
```

### Evaluation options:
```bash
python scripts/evaluate_model.py \
    --model-path models/best_model.pt \
    --data-yaml data/yolo/dataset.yaml \
    --split val \
    --conf-threshold 0.5 \
    --iou-threshold 0.5 \
    --visualize \
    --report
```

The evaluation script provides:
- Overall precision, recall, F1-score
- mAP@0.5 and mAP@0.5:0.95
- Per-class metrics
- Confusion matrices
- Precision-recall curves
- Comprehensive evaluation report

## Inference Pipeline

### Single Image Processing

```bash
python scripts/inference_pipeline.py \
    --yolo-model models/best_model.pt \
    --input path/to/image.jpg \
    --output results/image_1 \
    --brickognize-api-key your_api_key \
    --visualize \
    --save-results
```

### Batch Processing

```bash
python scripts/inference_pipeline.py \
    --yolo-model models/best_model.pt \
    --input path/to/image_directory \
    --output results/batch_1 \
    --brickognize-api-key your_api_key \
    --visualize \
    --save-results
```

### Inference Configuration

Edit `config/inference_config.yaml` to customize:
- Confidence thresholds
- Output formats and visualizations
- Performance settings
- Post-processing parameters

## Brickognize API Integration

The system can enhance YOLO detections with detailed brick identification using the Brickognize API:

### Features:
- Part number identification
- Brick type classification (brick, plate, tile, slope, etc.)
- Color detection
- Dimension estimation
- Detailed brick metadata

### Usage:
```python
from brickognize.brickognize_client import BrickognizeClient

client = BrickognizeClient(api_key="your_api_key")
result = client.identify_brick_from_file("path/to/brick_image.jpg")
predictions = client.parse_predictions(result)
```

## Optimal Model Selection

Based on the LEGO brick detection task, we recommend:

### For Accuracy (Best Detection Performance):
- **YOLOv8m**: Best balance of speed and accuracy
- **YOLOv8l**: Highest accuracy for critical applications
- **Input size: 640x640**: Good resolution for small brick details

### For Speed (Real-time Applications):
- **YOLOv8n**: Fastest inference
- **YOLOv8s**: Good speed with reasonable accuracy
- **Input size: 416x416**: Faster processing

### Training Recommendations:
- **Epochs: 100-150**: Sufficient for convergence
- **Batch size: 16**: Good for most GPUs
- **Learning rate: 0.01**: Standard starting point
- **Data augmentation: Enabled**: Improves generalization

## Performance Optimization

### Speed Optimization:
- Use smaller model sizes (YOLOv8n/s)
- Reduce input image size
- Enable half-precision inference
- Use GPU acceleration

### Accuracy Optimization:
- Use larger model sizes (YOLOv8m/l)
- Increase input image size
- Use comprehensive data augmentation
- Fine-tune hyperparameters

## Results and Output

The pipeline generates:

1. **Detection Results**:
   - Bounding boxes with confidence scores
   - Class predictions
   - Annotated images

2. **Brick Identification** (with Brickognize):
   - Part numbers and names
   - Brick types and colors
   - Dimensions
   - Complete brick inventory

3. **Evaluation Metrics**:
   - Precision, recall, F1-score
   - mAP scores
   - Per-class performance
   - Confusion matrices

4. **Visualizations**:
   - Detection confidence distributions
   - Class distribution charts
   - Precision-recall curves
   - Training curves (during training)

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size or use smaller model
2. **Dataset not found**: Check dataset paths and YAML configuration
3. **API errors**: Verify Brickognize API key and network connectivity
4. **Poor detection results**: Check training data quality and augmentation settings

### Getting Help:
- Check the evaluation reports for detailed metrics
- Review training curves for convergence issues
- Examine inference visualizations for detection quality
- Verify dataset preparation and annotation quality

## Next Steps

1. **Train your model**: Start with YOLOv8m and baseline configuration
2. **Evaluate performance**: Use comprehensive evaluation tools
3. **Optimize for your use case**: Adjust model size and settings
4. **Integrate with main pipeline**: Connect to the main LegoGen system
5. **Deploy for inference**: Use the inference pipeline for production

## API Integration

The trained model can be integrated with the main LegoGen pipeline through the classifier interface:

```python
from model.scripts.inference_pipeline import LegoInferencePipeline

def classify_bricks(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    pipeline = LegoInferencePipeline("path/to/model.pt")
    results = pipeline.detect_bricks(image)
    
    # Convert to expected format
    brick_counts = {}
    for detection in results['detections']:
        class_name = detection['class_name']
        brick_counts[class_name] = brick_counts.get(class_name, 0) + 1
    
    return brick_counts
```

This provides the interface required by the main LegoGen pipeline as specified in the `AGENTS.md` file.