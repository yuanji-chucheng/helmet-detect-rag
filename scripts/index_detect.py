from ultralytics import YOLO
import json
from pathlib import Path

model = YOLO('runs/detect/helmet_detect6/weights/best.pt')

img_dir = "./test_images"       # 待批量推理的图片文件夹
output_json = "./detect_index.json"
img_ext = ["*.jpg", "*.png"]

def main():
    img_paths = []
    for ext in img_ext:
        img_paths.extend(Path(img_dir).glob(ext))
    all_result = []
    for img in img_paths:
        res = model.predict(str(img), conf=0.4, verbose=False)[0]
        boxes = []
        for box in res.boxes:
            boxes.append({
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "xyxy": [round(float(x),2) for x in box.xyxy[0]]
            })
        all_result.append({
            "image": str(img),
            "boxes": boxes
        })
    # 保存索引文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_result, f, ensure_ascii=False, indent=2)
    print(f"批量推理完成，结果索引保存至 {output_json}")

if __name__ == "__main__":
    main()