"""RAG 检索增强生成模块（头盔检测 + 交通法规知识库问答）。

该包为独立新增模块，不修改原有 YOLO 训练、数据集代码，
仅复用本地 runs/detect/helmet_detect6/weights/best.pt 权重做推理。
"""
import os as _os

# 必须在导入 langchain / transformers / huggingface_hub 之前设置，
# 否则镜像端点不会生效。国内默认使用 HuggingFace 镜像下载嵌入模型。
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
