from ultralytics import YOLO
import os

# Get the current directory (dataset folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, 'data.yaml')

print(f"Training plastic bag detection model...")
print(f"Dataset config: {data_yaml_path}")
print(f"Dataset contains plastic bag class for YOLOv11 training")

# Load pretrained YOLOv11n (nano - fastest for real-time detection)
model = YOLO('yolo11n.pt')

# Train on plastic bag dataset
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,  # Adjust based on your Mac RAM (reduce if OOM)
    device='mps',  # Uses Apple Silicon GPU
    name='plastic_bag_detector',
    save=True,
    plots=True,
    verbose=True
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}/best.pt")
print(f"Results saved at: {results.save_dir}")
