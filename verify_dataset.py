import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np

def validate_yolo_dataset(base_path, config_path):
    """
    Comprehensive validation of YOLO dataset
    """
    # Load config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Paths
    train_img_path = os.path.join(base_path, config['train'])
    val_img_path = os.path.join(base_path, config['val'])
    
    # Create corresponding label paths
    train_label_path = train_img_path.replace('images', 'labels')
    val_label_path = val_img_path.replace('images', 'labels')
    
    print(f"Validating training dataset in: {train_img_path}")
    validate_subset(train_img_path, train_label_path, config['names'])
    
    print(f"\nValidating validation dataset in: {val_img_path}")
    validate_subset(val_img_path, val_label_path, config['names'])

def validate_subset(image_path, label_path, class_names):
    """
    Validate a subset of the dataset
    """
    # Check if paths exist
    if not os.path.exists(image_path):
        print(f"❌ Image path does not exist: {image_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"❌ Label path does not exist: {label_path}")
        return
    
    # Get image and label files
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(label_path) if f.lower().endswith('.txt')]
    
    print(f"Total images: {len(image_files)}")
    print(f"Total label files: {len(label_files)}")
    
    # Check for matching files
    image_names = {os.path.splitext(f)[0] for f in image_files}
    label_names = {os.path.splitext(f)[0] for f in label_files}
    
    missing_labels = image_names - label_names
    missing_images = label_names - image_names
    
    if missing_labels:
        print(f"⚠️ {len(missing_labels)} images missing corresponding label files:")
        print(list(missing_labels)[:5])  # Show first 5
    
    if missing_images:
        print(f"⚠️ {len(missing_images)} label files missing corresponding images:")
        print(list(missing_images)[:5])  # Show first 5
    
    # Validate label format and content
    invalid_labels = []
    class_distribution = {i: 0 for i in range(len(class_names))}
    
    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            
            # Validate label line
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Check class ID
                if class_id < 0 or class_id >= len(class_names):
                    invalid_labels.append((label_file, line_num, f"Invalid class ID: {class_id}"))
                
                # Check normalized coordinates
                if not all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                    invalid_labels.append((label_file, line_num, "Coordinates must be normalized"))
                
                # Count class distribution
                class_distribution[class_id] += 1
                
            except (ValueError, IndexError):
                invalid_labels.append((label_file, line_num, "Invalid label format"))
    
    # Report label validation results
    if invalid_labels:
        print("\n⚠️ Invalid Labels Found:")
        for file, line, error in invalid_labels[:10]:  # Show first 10
            print(f"  - {file}, Line {line}: {error}")
    else:
        print("\n✅ All label files pass basic validation")
    
    # Print class distribution
    print("\nClass Distribution:")
    for class_id, count in class_distribution.items():
        print(f"  {class_names[class_id]}: {count} objects")

# Usage
if __name__ == "__main__":
    base_path = "/Users/adityaraj/Coding/PersonalProjects/yolo/data"
    config_path = "config.yaml"  # Adjust if needed
    validate_yolo_dataset(base_path, config_path)