from ultralytics import YOLO
import torch

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Khởi tạo model YOLOv11x classification từ đầu (không pretrained)
# Thay vì dùng 'yolo11x-cls.pt', dùng 'yolo11x-cls.yaml' để tạo model từ đầu
model = YOLO('yolo11-soict.yaml')
model.to(device)

print(f"Đã khởi tạo YOLOv11x Classification model từ đầu trên {device}")
print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# Train với các config giống hệt code gốc của bạn
results = model.train(
    data='/content',
    epochs=150,
    batch=64,
    imgsz=224,
    project='yolo_cls_results',
    seed=42,
    lr0=1e-4,
    mosaic=0,
    augment=True,
    dropout=0.3,
    degrees=15,
    translate=0.1,
    shear=0.2,
    fliplr=0.5,
    flipud=0.2,
    scale=0.1,
    pretrained=False,  # Thêm dòng này để đảm bảo không dùng pretrained
    verbose=True,
    save_period=15,
)

print("Training hoàn thành!")
print(f"Best weights: runs/classify/train/weights/best.pt")
print(f"Last weights: runs/classify/train/weights/last.pt")