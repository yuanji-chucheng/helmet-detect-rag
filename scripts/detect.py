from ultralytics import YOLO

model = YOLO('runs/detect/helmet_detect6/weights/best.pt')

# 对单张图片预测，保存结果
results = model.predict(source='asset/未佩戴.png', save=True, conf=0.5)