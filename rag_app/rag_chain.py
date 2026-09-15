"""RAG 链：Chroma 检索 + DeepSeek 在线 LLM 生成。

能力：answer_question_stream 基于交通法规知识库进行流式问答（带对话历史）。

LLM 全程调用 DeepSeek 在线 API（OpenAI 兼容接口）。
"""
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .knowledge_base import retrieve


@lru_cache(maxsize=4)
def _get_llm(api_key: str, temperature: float = 0.3) -> ChatOpenAI:
    """构造 DeepSeek 在线 LLM 客户端（进程内单例缓存）。

    使用 lru_cache 按 (api_key, temperature) 缓存实例，避免重复创建；
    开启 streaming=True 以支持逐 token 流式输出。
    """
    if not api_key:
        raise ValueError(
            "未提供 DeepSeek API Key，请在项目根目录 .env 中"
            "填写 DEEPSEEK_API_KEY=sk-xxxx 后重启应用。"
        )
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=1024,
        timeout=60,
        streaming=True,
    )


def _format_docs(docs) -> str:
    """把检索到的文本块整理成带编号、章节的上下文。"""
    parts = []
    for idx, doc in enumerate(docs, 1):
        section = doc.metadata.get("section", "正文")
        parts.append(f"【资料{idx}｜章节：{section}】\n{doc.page_content}")
    return "\n\n".join(parts) if parts else "（未检索到相关法规内容）"


# ---------------- 法规知识库问答 ----------------

QA_SYSTEM_TEMPLATE = """你是一名交通安全法规智能助手，服务于电动自行车安全头盔智能巡检场景。
请严格依据下面"检索到的法规知识库内容"回答用户问题：
1. 只使用知识库中的信息作答，不得编造法规条款、罚款金额或统计数据；
2. 知识库未覆盖的内容，明确说明"知识库中暂无相关规定，建议移交人工复核"；
3. 使用中文回答，条理清晰，必要时分点列出；
4. 涉及执法时提醒：AI 识别结果仅作参考，不能直接单独作为执法凭证。

检索到的法规知识库内容：
{context}"""


def answer_question_stream(question: str, api_key: str, history=None):
    """基于知识库流式回答用户问题（生成器，逐 chunk yield 文本）。

    Args:
        question: 用户问题。
        api_key: DeepSeek API Key。
        history: 对话历史，格式为 [(用户问题, 助手回答), ...]。

    Yields:
        文本片段（str）。LLM 输出完毕后，不再追加文档来源引用。
    """
    # 1. 检索知识库（top-3）
    docs = retrieve(question)
    context = _format_docs(docs)

    # 2. 组装带历史的 prompt
    messages = [("system", QA_SYSTEM_TEMPLATE.format(context=context))]
    for user_msg, ai_msg in (history or [])[-6:]:  # 最多带最近 6 轮历史
        messages.append(("human", user_msg))
        messages.append(("assistant", ai_msg))
    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | _get_llm(api_key) | StrOutputParser()

    # 3. 流式输出
    for chunk in chain.stream({"question": question}):
        if isinstance(chunk, str):
            text = chunk
        else:
            text = getattr(chunk, "content", "") or str(chunk)
        if text:
            yield text

