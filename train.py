from ultralytics import YOLO
import cv2
import torch
model = YOLO("yolo11-soict.yaml")   # load kiến trúc từ yaml
# print(model)
# model = YOLO("yolo11n.pt")

img = cv2.imread("D:/lele/b.jpg")       # đọc ảnh
img = cv2.resize(img, (640, 640))       # classification thường dùng 224x224
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # đổi sang RGB

# CHW + float32 + chuẩn hóa
img = img.transpose(2, 0, 1)            # HWC -> CHW
img = torch.tensor(img, dtype=torch.float32) / 255.0  # scale 0–1
img = img.unsqueeze(0)                  # thêm batch dim [1,C,H,W]
pred = model.model(img)                      # dự đoán
print(pred)