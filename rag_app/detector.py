"""YOLOv8 头盔检测推理封装。

仅加载本地已训练好的 best.pt 做推理，不涉及任何训练 / 数据集代码。
模型类别：head（裸露头部，即未佩戴头盔）、helmet（已佩戴头盔）。
"""
from functools import lru_cache

from ultralytics import YOLO

from .config import BEST_PT_PATH

# 模型类别名 -> 业务含义
CLASS_MEANING = {
    "head": "未佩戴头盔（裸露头部）",
    "helmet": "已佩戴头盔",
}

# 低置信度阈值（低于该值视为置信度不足，需人工复核）
LOW_CONF_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """懒加载 YOLO 模型权重（进程内只加载一次）。"""
    if not BEST_PT_PATH.exists():
        raise FileNotFoundError(f"找不到模型权重文件：{BEST_PT_PATH}")
    return YOLO(str(BEST_PT_PATH))


def detect(image, conf: float = 0.4):
    """对单张图片执行头盔检测。

    Args:
        image: 图片路径或 numpy 数组（Gradio 上传的图片）。
        conf: 置信度阈值。

    Returns:
        (annotated_rgb, info)：带检测框的标注图（RGB，供 Gradio 显示）与检测结果字典。
    """
    model = get_model()
    results = model.predict(source=image, conf=conf, verbose=False)
    result = results[0]
    names = result.names

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = names[cls_id]
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
        detections.append(
            {
                "label": label,
                "meaning": CLASS_MEANING.get(label, label),
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
            }
        )

    no_helmet = [d for d in detections if d["label"] == "head"]
    helmet = [d for d in detections if d["label"] == "helmet"]
    low_conf = [d for d in detections if d["confidence"] < LOW_CONF_THRESHOLD]

    # ultralytics 标注图为 BGR，转为 RGB 供 Gradio 显示
    annotated_rgb = result.plot()[:, :, ::-1]

    info = {
        "total_targets": len(detections),
        "no_helmet_count": len(no_helmet),
        "helmet_count": len(helmet),
        "low_confidence_count": len(low_conf),
        "detections": detections,
    }
    return annotated_rgb, info


def format_detection_summary(info: dict) -> str:
    """将检测结果整理为供 LLM 使用的文字摘要。"""
    lines = [
        f"检测目标总数：{info['total_targets']}",
        f"已佩戴头盔人数（helmet）：{info['helmet_count']}",
        f"未佩戴头盔人数（head，裸露头部）：{info['no_helmet_count']}",
        f"置信度不足（< {LOW_CONF_THRESHOLD}）目标数：{info['low_confidence_count']}",
        "明细：",
    ]
    if info["detections"]:
        for idx, d in enumerate(info["detections"], 1):
            lines.append(
                f"  {idx}. {d['meaning']}，置信度 {d['confidence']:.3f}，"
                f"检测框坐标 {d['bbox']}"
            )
    else:
        lines.append("  未检测到任何目标（可能画面中无人员或置信度过低）。")
    return "\n".join(lines)


if __name__ == "__main__":
    # 快速自检：python -m rag_app.detector
    from .config import PROJECT_ROOT

    test_image = PROJECT_ROOT / "asset" / "未佩戴.png"
    annotated, info = detect(str(test_image))
    print(format_detection_summary(info))
    print("标注图尺寸：", annotated.shape)
