"""RAG 链：Chroma 检索 + DeepSeek 在线 LLM 生成。

提供两类能力：
1. answer_question：交通法规知识库问答（带对话历史）；
2. analyze_detection：结合 YOLO 头盔检测结果，检索法规并生成巡检解读报告。

LLM 全程调用 DeepSeek 在线 API（OpenAI 兼容接口），本地不运行大模型。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .knowledge_base import retrieve


def _get_llm(api_key: str, temperature: float = 0.3) -> ChatOpenAI:
    """构造 DeepSeek 在线 LLM 客户端。"""
    if not api_key:
        raise ValueError(
            "未提供 DeepSeek API Key，请在界面顶部输入，或设置环境变量 DEEPSEEK_API_KEY。"
        )
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=1024,
        timeout=60,
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


def answer_question(question: str, api_key: str, history=None):
    """基于知识库回答用户问题。

    Args:
        question: 用户问题。
        api_key: DeepSeek API Key。
        history: 对话历史，格式为 [(用户问题, 助手回答), ...]。

    Returns:
        (answer, docs)：生成的回答与检索到的来源文档。
    """
    docs = retrieve(question)
    context = _format_docs(docs)

    messages = [("system", QA_SYSTEM_TEMPLATE.format(context=context))]
    for user_msg, ai_msg in (history or [])[-6:]:  # 最多带最近 6 轮历史
        messages.append(("human", user_msg))
        messages.append(("assistant", ai_msg))
    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | _get_llm(api_key) | StrOutputParser()
    answer = chain.invoke({"question": question})
    return answer, docs


# ---------------- 头盔检测结果解读 ----------------

DETECT_SYSTEM_TEMPLATE = """你是电动自行车头盔佩戴智能巡检系统的安全分析助手。
下面给出 AI 视觉模型对一张图片的检测结果，以及从交通安全法规知识库中检索到的相关内容。
请严格依据这两部分信息，输出一份"智能巡检解读报告"，包含以下四个部分：
1. **检测结论**：说明画面中已佩戴 / 未佩戴头盔的人数与置信度情况；
2. **安全风险提示**：结合知识库中的事故风险数据说明不佩戴头盔的危害；
3. **法规与处罚依据**：引用知识库中的相关规定（不得编造条款或罚款金额）；
4. **处置建议**：针对未佩戴人员给出劝导 / 处罚 / 人工复核建议，并提醒 AI
   识别结果仅作参考、不能直接单独作为执法凭证。

要求：使用中文，分点清晰；若检测结果显示全部规范佩戴，也请给出合规确认与
保持佩戴的安全提示；若置信度不足目标数大于 0，必须提示移交人工复核。

AI 视觉检测结果：
{detection_summary}

检索到的法规知识库内容：
{context}"""


def analyze_detection(detection_summary: str, detection_info: dict, api_key: str):
    """结合检测结果检索法规并生成解读报告。

    Args:
        detection_summary: format_detection_summary 输出的文字摘要。
        detection_info: detect() 输出的结果字典，用于构造检索问题。
        api_key: DeepSeek API Key。

    Returns:
        (report, docs)：巡检解读报告与检索到的来源文档。
    """
    # 根据检测结论选择更贴合的检索问题
    if detection_info.get("no_helmet_count", 0) > 0:
        query = (
            "电动自行车 未佩戴安全头盔 法律责任 处罚 罚款 "
            "颅脑损伤 安全风险 人工复核 执法凭证"
        )
    elif detection_info.get("helmet_count", 0) > 0:
        query = "安全头盔 规范佩戴 3C认证 合格头盔标准 安全防护作用"
    else:
        query = "电动自行车 安全头盔 佩戴规定 安全风险 智能巡检 人工复核"

    docs = retrieve(query)
    context = _format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [("system", DETECT_SYSTEM_TEMPLATE), ("human", "请生成巡检解读报告。")]
    )
    chain = prompt | _get_llm(api_key, temperature=0.4) | StrOutputParser()
    report = chain.invoke(
        {"detection_summary": detection_summary, "context": context}
    )
    return report, docs
