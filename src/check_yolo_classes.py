from ultralytics import YOLO

model_path = "best.pt"   # change this to your model path

model = YOLO(model_path)

print("YOLO model classes:")
print(model.names)

print("\nClass list:")
for class_id, class_name in model.names.items():
    print(f"{class_id} - {class_name}")