**YOLO Model Training and Validation**

A comprehensive pipeline for training, validating, and testing YOLO models with custom datasets. This repository includes scripts for model training, dataset verification, and prediction using the Ultralytics YOLO framework.

Table of Contents

	1.	Project Structure
	2.	Setup Instructions
	3.	Usage
	•	Training
	•	Prediction
	•	Dataset Validation
	•	System Compatibility Check
	4.	Dataset Format
	5.	Features
	6.	Troubleshooting
	7.	References

Setup Instructions

	1.	Clone the Repository
Clone this repository to your local machine:

git clone https://github.com/your-repo/yolo-pipeline.git
cd yolo-pipeline


	2.	Install Dependencies
Install the required Python libraries:

pip install ultralytics torch pyyaml opencv-python


	3.	Prepare the Dataset
	•	Organize your dataset in YOLO format with images/ and labels/ folders.
	•	Update the config.yaml file to reflect your dataset structure.

Usage

1. Train the Model

Run the main.py script to train the YOLO model:

python main.py

	•	Key Parameters:
	•	config_path: Path to the dataset configuration file (config.yaml).
	•	model_yaml: Model architecture (default: yolov8n.yaml).
	•	epochs: Number of training epochs (default: 100).
	•	device: Compute device (cpu, cuda, or mps for macOS).

2. Run Predictions

Use the predict.py script to perform inference:

python predict.py

	•	Modify Input:
Update the source parameter in the script with the path to the test image.

3. Validate Dataset

Run the verify_dataset.py script to ensure the dataset is correctly formatted:

python verify_dataset.py

This script:
	•	Ensures all images have corresponding labels.
	•	Checks label file integrity and class distribution.

4. Check System Compatibility

Run the verify.py script to confirm MPS (GPU) compatibility on macOS:

python verify.py

Dataset Format

The dataset must follow the YOLO format:

	•	images/: Contains .jpg, .png, etc.
	•	labels/: Contains .txt files with bounding box annotations.
	•	config.yaml: Describes the dataset structure and class names.

Example config.yaml file:

train: data/train/images
val: data/val/images
nc: 10
names: ['class1', 'class2', 'class3', ..., 'class10']

	•	train: Path to training images.
	•	val: Path to validation images.
	•	nc: Number of classes.
	•	names: List of class names.

Features

Training

	•	Advanced Augmentations:
	•	Mosaic
	•	Mixup
	•	Rotation
	•	Scaling
	•	Optimized Parameters:
	•	Anchor box generation
	•	Learning rate scheduling
	•	Confidence and IoU thresholds

Validation

	•	Checks for missing or mismatched labels.
	•	Reports class distribution and identifies invalid annotations.

Prediction

	•	Generates bounding boxes and saves annotated images for review.

Troubleshooting


Common Issues


Problem	Solution

Dataset format errors	Run verify_dataset.py to validate dataset.
Slow training	Reduce epochs or use a smaller image size.
MPS not available on macOS	Run verify.py to check GPU compatibility.

References

	•	Ultralytics YOLO Documentation
	•	PyTorch
	•	Roboflow Tools
