from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='yolo-bvn.yaml',
    epochs=30,  #训练轮次（受CPU算力限制，未设置更高
    batch=16,   #批次大小（根据16GB内存设定）
    imgsz=640,
    device='cpu',  #使用CPU训练
    workers=0,    #数据加载线程数（Windows下避免多进程死锁）
    lr0=0.01,    #初始学习率（SGD优化器默认值）
    optimizer='SGD'
)
print("训练完成！最佳模型保存在:", model.trainer.best)