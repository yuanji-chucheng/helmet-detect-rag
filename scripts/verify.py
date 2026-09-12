from ultralytics import YOLO

# 加载训练好的最佳模型
model = YOLO('runs/detect/helmet_detect6/weights/best.pt')

# 在验证集上计算全套评估指标
results = model.val(data='yolo-bvn.yaml', split='val')

print("===== YOLO模型验证评估结果 =====")
print(f"mAP@0.5:        {results.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {results.box.map:.4f}")
print(f"Precision(AP):  {results.box.mp:.4f}")
print(f"Recall:         {results.box.mr:.4f}")