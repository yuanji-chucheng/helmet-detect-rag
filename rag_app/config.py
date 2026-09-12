"""全局配置：文件路径、嵌入模型、切分参数、DeepSeek 在线 LLM 参数。

DeepSeek API Key 读取顺序：
1. 环境变量 DEEPSEEK_API_KEY；
2. 项目根目录 .env 文件中的 DEEPSEEK_API_KEY=sk-xxxx；
3. Gradio 界面顶部输入框（运行时传入）。
本地不运行任何大语言模型，LLM 推理全部走 DeepSeek 在线 API。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（rag_app 的上一级，即 ultralytics-main/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载项目根目录的 .env（不存在时静默跳过），不覆盖已设置的环境变量
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _find_existing(*candidates: Path) -> Path:
    """返回候选路径中第一个存在的；都不存在则返回第一个候选。"""
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# ---------- 知识库文档 ----------
KB_PATH = _find_existing(
    PROJECT_ROOT / "runs" / "knowledge_base" / "traffic_safety.md",
    PROJECT_ROOT / "knowledge_base" / "traffic_safety.md",
)

# ---------- YOLO 权重（复用本地 best.pt） ----------
BEST_PT_PATH = _find_existing(
    PROJECT_ROOT / "runs" / "detect" / "helmet_detect6" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "weights" / "best.pt",
)

# ---------- Chroma 本地向量数据库 ----------
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "traffic_safety"

# ---------- 本地嵌入模型（小型中文语义向量模型，非大语言模型） ----------
# 国内网络默认走 HuggingFace 镜像，首次运行自动下载（约 100MB，之后走缓存）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# ---------- 文档切分参数 ----------
CHUNK_SIZE = 300      # 每个文本块最大字符数
CHUNK_OVERLAP = 50    # 相邻块重叠字符数
RETRIEVER_K = 3       # 每次检索返回的相关文本块数量

# ---------- DeepSeek 在线 LLM ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


def get_deepseek_api_key() -> str:
    """从环境变量读取 DeepSeek API Key。"""
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()
