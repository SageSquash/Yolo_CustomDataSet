from ultralytics import YOLO

# Load the trained model
model = YOLO('/Users/adityaraj/Coding/PersonalProjects/yolo/runs/detect/train/weights/best.pt')  # Ensure you're loading the best checkpoint

# Predict on an image
results = model.predict(
    source='/Users/adityaraj/Coding/PersonalProjects/yolo/data/valid/images/173916-13239000-2001891400_png.rf.4738a39f0fe0342ebe51d46ccd899fe6.jpg', 
    conf=0.25,  # Lower confidence threshold
    save=True,  # Save annotated images
    verbose=True
)

# Manually inspect results
for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} objects")