from ultralytics import YOLO
# 检查数据集是否可读
model = YOLO('yolov8n.pt')
model.val(data='yolo-bvn.yaml', split='val', imgsz=640, batch=4)