"""
YOLO头盔数据集清洗脚本
特点：
1. 检查图片损坏、分辨率过小、严重模糊、极端过曝/严重欠曝
2. 自动匹配YOLO标签txt文件，检查图片-label成对完整性
3. 默认仅输出报告，**不会自动删除图片**，人工复核清单后再操作
4. 适配道路监控头盔数据集，降低误杀率，保留轻微模糊难样本

依赖安装：
pip install opencv-python pillow send2trash
"""

import os
from pathlib import Path
import cv2
import numpy as np

# ========== 【可自行调整配置参数】 ==========
DATASET_IMAGE_ROOT = "./bvn/helmet"       # 图片目录
LABEL_ROOT = "./bvn/labels"               # YOLO标签txt目录
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# 图像质量阈值（针对监控头盔数据集，宽松版本，减少误删！）
BLUR_THRESHOLD = 15                      # 拉普拉斯方差，低于这个才判定严重模糊
OVEREXP_THRESHOLD = 230                  # 灰度均值上限，高于判定极端过曝
UNDEREXP_THRESHOLD = 30                  # 灰度均值下限，低于判定严重欠曝
MIN_WIDTH = 320                          # 最小图片宽度，更小直接判定无效
MIN_HEIGHT = 320                         # 最小图片高度

DRY_RUN = True                           # True=仅输出报告，不删除文件；人工确认后再改成False
USE_RECYCLE_BIN = True                   # 删除时丢入回收站，而不是永久删除
# ==========================================

if USE_RECYCLE_BIN:
    try:
        from send2trash import send2trash
    except ImportError:
        print("⚠️ 未安装send2trash，后续删除将直接永久删除！安装命令：pip install send2trash")
        USE_RECYCLE_BIN = False


def find_image_files(root_path):
    """递归查找所有图片文件"""
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(Path(root_path).rglob(f"*{ext}"))
        image_files.extend(Path(root_path).rglob(f"*{ext.upper()}"))
    return sorted(image_files)


def get_corresponding_label(img_path: Path, img_root: Path, label_root: Path) -> Path:
    """根据图片路径，找到对应的YOLO标签txt"""
    rel_path = img_path.relative_to(img_root)
    label_name = rel_path.with_suffix(".txt")
    return label_root / label_name


def check_format_and_size(img_path):
    """
    检查图片是否可读，分辨率是否达标
    返回 (is_valid, width, height, channels, reason)
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            # 兜底用PIL读取
            with Image.open(img_path) as pil_img:
                width, height = pil_img.size
                channels = len(pil_img.getbands())
        else:
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return False, width, height, channels, f"分辨率过小({width}×{height})"
        return True, width, height, channels, "正常"
    except Exception as e:
        return False, 0, 0, 0, f"图片损坏无法读取，{str(e)}"


def judge_image_quality(img_path, blur_thresh, over_thresh, under_thresh):
    """
    图像质量检测：模糊、过曝、欠曝
    返回 (is_bad, reason)
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return True, "图片读取失败"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = np.mean(gray)

        if laplacian_var < blur_thresh:
            return True, f"严重模糊，拉普拉斯方差={laplacian_var:.2f}"
        if mean_brightness > over_thresh:
            return True, f"极端过曝，平均亮度={mean_brightness:.1f}"
        if mean_brightness < under_thresh:
            return True, f"严重欠曝，平均亮度={mean_brightness:.1f}"
        return False, "图像质量正常"
    except Exception as e:
        return True, f"质量检测异常: {e}"


def delete_pair(img_file:Path, label_file:Path):
    """删除图片+配套标签文件，支持回收站"""
    def safe_del(p:Path):
        if not p.exists():
            return
        try:
            if USE_RECYCLE_BIN:
                send2trash(p)
            else:
                os.remove(p)
            print(f"    ✅ 删除: {p}")
        except Exception as e:
            print(f"    ❌ 删除失败 {p} | {e}")
    safe_del(img_file)
    safe_del(label_file)


def main():
    print("="*70)
    print("YOLO数据集清洗工具【优化完整版】")
    print(f"图片目录：{DATASET_IMAGE_ROOT}")
    print(f"标签目录：{LABEL_ROOT}")
    print(f"DRY_RUN模式：{DRY_RUN} | 自动删除开关：{not DRY_RUN}")
    print("="*70)

    img_root = Path(DATASET_IMAGE_ROOT)
    label_root = Path(LABEL_ROOT)
    all_images = find_image_files(img_root)
    print(f"\n共检索图片数量：{len(all_images)}")

    invalid_pairs = []       # 图片损坏/分辨率不合格
    bad_quality_pairs = []   # 严重模糊/过曝/欠曝
    missing_label_list = []  # 有图片但缺失txt标签

    for img_path in all_images:
        label_path = get_corresponding_label(img_path, img_root, label_root)
        valid, w, h, c, reason = check_format_and_size(img_path)
        if not valid:
            invalid_pairs.append( (img_path, label_path, reason) )
            continue

        # 检查标签是否存在
        if not label_path.exists():
            missing_label_list.append( (img_path, label_path) )

        # 图像质量检测
        is_bad_quality, quality_reason = judge_image_quality(img_path, BLUR_THRESHOLD, OVEREXP_THRESHOLD, UNDEREXP_THRESHOLD)
        if is_bad_quality:
            bad_quality_pairs.append( (img_path, label_path, quality_reason) )

    # 输出报告
    print("\n---------- 1. 损坏/分辨率不合格图片 ----------")
    for img,lab,msg in invalid_pairs:
        print(f"  {img.name} -> {msg}")

    print("\n---------- 2. 严重质量缺陷图片（模糊/过曝/欠曝） ----------")
    for img,lab,msg in bad_quality_pairs:
        print(f"  {img.name} -> {msg}")

    print("\n---------- 3. 存在图片，但缺失YOLO标签txt ----------")
    for img,lab in missing_label_list:
        print(f"  {img.name} 缺失标签：{lab}")

    total_bad = len(invalid_pairs)+len(bad_quality_pairs)
    print(f"\n【汇总统计】")
    print(f"损坏/分辨率过小：{len(invalid_pairs)}")
    print(f"严重模糊/极端曝光：{len(bad_quality_pairs)}")
    print(f"有图无标签：{len(missing_label_list)}")
    print(f"待人工复核总量：{total_bad}")

    # 导出复核清单
    with open("dataset_review_list.txt", "w", encoding="utf-8") as f:
        f.write("===== YOLO数据集清洗复核清单（人工查看！）=====\n\n")
        f.write("【1 损坏/分辨率过小图片】\n")
        for img,lab,msg in invalid_pairs:
            f.write(f"{str(img)} | 标签:{str(lab)} | 原因:{msg}\n")
        f.write("\n【2 严重质量缺陷图片】\n")
        for img,lab,msg in bad_quality_pairs:
            f.write(f"{str(img)} | 标签:{str(lab)} | 原因:{msg}\n")
        f.write("\n【3 有图片但缺少标签文件】\n")
        for img,lab in missing_label_list:
            f.write(f"{str(img)} | 预期标签:{str(lab)}\n")
    print("\n✅ 清单文件已生成：dataset_review_list.txt，请打开人工预览图片确认！")

    if not DRY_RUN:
        print("\n⚠️ DRY_RUN=False，准备批量删除【损坏+严重质量缺陷】图片+标签！")
        user_input = input("确认批量删除？输入 yes 确认，其他任意字符取消：")
        if user_input.strip() == "yes":
            for img,lab,msg in invalid_pairs:
                delete_pair(img, lab)
            for img,lab,msg in bad_quality_pairs:
                delete_pair(img, lab)
            print("\n✅ 批量删除完成！")
        else:
            print("❌ 用户取消删除操作")
    else:
        print("\n✅ 当前DRY_RUN=True，没有删除任何文件。")
        print("   确认清单图片全部需要删除后，修改 DRY_RUN=False，重新运行脚本，并且输入yes确认。")


if __name__ == "__main__":
    from PIL import Image
    main()