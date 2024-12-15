import os
import torch
from ultralytics import YOLO
import logging
from datetime import datetime

def configure_model_for_performance(model, num_classes):
    """
    Customize model configuration for better performance
    """
    # Custom configurations based on dataset characteristics
    custom_cfg = {
        # Adjust anchor generation
        'anchors': 3,  # Number of anchor boxes
        
        # Tune detection thresholds
        'conf': 0.001,  # Lower confidence threshold for detection
        'iou': 0.45,   # Intersection over Union threshold
        
        # Class-specific settings
        'nc': num_classes,  # Number of classes
        
        # Performance optimization
        'max_det': 300,  # Maximum detections per image
        'nms_time_limit': 5.0,  # Increased NMS time limit
    }
    
    return custom_cfg

def train_yolo_model(
    config_path='config.yaml', 
    model_yaml="yolov8n.yaml", 
    epochs=2,  # Increased epochs 
    imgsz=640, 
    device='mps',
    patience=20,  # Increased patience
    batch=-1
):
    """
    Enhanced YOLO training with performance optimizations
    """
    # Load configuration to get number of classes
    import yaml
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    num_classes = cfg.get('nc', 10)

    try:
        # Load base model
        model = YOLO(model_yaml)
        
        # Custom model configuration
        custom_cfg = configure_model_for_performance(model, num_classes)
        
        # Advanced training parameters
        results = model.train(
            data=config_path, 
            epochs=epochs, 
            imgsz=imgsz, 
            device=device,
            patience=patience,
            batch=batch,
            
            # Performance and debugging parameters
            verbose=True,
            plots=True,
            save=True,
            exist_ok=True,
            
            # Custom detection parameters
            conf=custom_cfg['conf'],
            iou=custom_cfg['iou'],
            max_det=custom_cfg['max_det'],
            
            # Augmentation for improved generalization
            mosaic=0.8,       # Mosaic augmentation
            mixup=0.2,        # Mixing images
            hsv_h=0.015,      # Hue shift
            hsv_s=0.7,        # Saturation shift
            hsv_v=0.4,        # Value shift
            degrees=15.0,     # Rotation
            translate=0.2,    # Translation
            scale=0.5,        # Scaling
            shear=10.0,       # Shear
            
            # Learning rate and optimization
            lr0=0.01,         # Initial learning rate
            lrf=0.1,          # Final learning rate factor
            momentum=0.937,   # Momentum
            weight_decay=0.0005  # Weight decay
        )
        
        print("Training Results Summary:")
        print(f"Best Fitness: {results.fitness}")
        
        return results
    
    except Exception as e:
        print(f"Training Error: {e}")
        raise

# Usage
if __name__ == "__main__":
    train_yolo_model(
        config_path="config.yaml", 
        model_yaml="yolo11n.pt", 
        epochs=100,  # More epochs for learning
        device='mps'
    )