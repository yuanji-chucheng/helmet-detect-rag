"""RAG 知识库构建与检索。

流程：读取本地 traffic_safety.md -> 按 Markdown 标题 + 字符长度切分
-> 本地嵌入模型向量化 -> 存入 Chroma 本地向量数据库。
"""
import shutil
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    KB_PATH,
    RETRIEVER_K,
)


def load_markdown() -> str:
    """读取交通法规 Markdown 文档原文。"""
    if not KB_PATH.exists():
        raise FileNotFoundError(f"知识库文档不存在：{KB_PATH}")
    return KB_PATH.read_text(encoding="utf-8")


def split_documents() -> list[Document]:
    """文档切分：先按 Markdown 标题（# / ##）切分并保留章节元数据，
    再按字符长度递归切分，保证每个块语义完整、长度适中。"""
    text = load_markdown()

    # 第一步：按标题切分，章节名写入 metadata
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "section")],
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(text)

    # 第二步：按长度切分（中文友好分隔符，尽量在句号/换行处断开）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "：", "、", " ", ""],
    )
    chunks = text_splitter.split_documents(header_docs)

    # 补充元数据：来源文件、章节、块编号
    for idx, chunk in enumerate(chunks):
        chunk.metadata["source"] = KB_PATH.name
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["section"] = (
            chunk.metadata.get("section")
            or chunk.metadata.get("h1")
            or "正文"
        )
    return chunks


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """本地中文嵌入模型（bge-small-zh），首次使用自动下载并缓存。"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(force: bool = False) -> Chroma:
    """构建 / 加载 Chroma 本地向量库。

    Args:
        force: True 时删除已有向量库并重新构建（知识库更新后使用）。
    """
    if force and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    # 若知识库源文件比向量库目录更新，自动重建以同步最新内容
    if (
        not force
        and CHROMA_DIR.exists()
        and any(CHROMA_DIR.iterdir())
        and KB_PATH.exists()
        and KB_PATH.stat().st_mtime > CHROMA_DIR.stat().st_mtime
    ):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        force = True

    # 已存在持久化数据则直接加载复用
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        if vectorstore._collection.count() > 0 and not force:
            return vectorstore

    # 不存在则切分文档并写入
    chunks = split_documents()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """获取向量库单例（首次调用时自动构建）。"""
    return build_vectorstore()


def get_retriever(k: int = RETRIEVER_K):
    """获取 LangChain 检索器。"""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int = RETRIEVER_K) -> list[Document]:
    """根据问题检索最相关的 k 个法规文本块。"""
    return get_retriever(k).invoke(query)


def collection_stats() -> dict:
    """返回向量库状态信息（供界面展示）。"""
    vs = get_vectorstore()
    return {
        "文档": KB_PATH.name,
        "文本块数量": vs._collection.count(),
        "嵌入模型": EMBEDDING_MODEL,
        "持久化目录": str(CHROMA_DIR),
    }


if __name__ == "__main__":
    # 手动构建/重建知识库：python -m rag_app.knowledge_base
    docs = split_documents()
    print(f"文档切分完成，共 {len(docs)} 个文本块：")
    for d in docs:
        preview = d.page_content.replace("\n", " ")[:60]
        print(f"  [#{d.metadata['chunk_id']}｜{d.metadata['section']}] {preview}...")
    vs = build_vectorstore(force=True)
    print(f"Chroma 向量库已构建，文本块数量：{vs._collection.count()}")
    print(f"持久化目录：{CHROMA_DIR}")
