"""Gradio 交互界面：YOLOv8 头盔检测 + RAG 交通法规问答。"""
import os
import gradio as gr

from .config import get_deepseek_api_key
from .detector import detect
from .rag_chain import answer_question_stream

# 读取CSS 文件
CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        CUSTOM_CSS = f.read()
else:
    CUSTOM_CSS = ""

# ---------------- 通用工具 ----------------

def _friendly_error(exc: Exception) -> str:
    """把异常转换为友好的中文提示，避免向用户暴露堆栈。"""
    msg = str(exc)
    if "API" in msg or "api" in msg or "401" in msg:
        return "DeepSeek API 调用失败，请检查 API Key 是否正确或稍后重试。"
    if "timed out" in msg or "timeout" in msg:
        return "请求超时，请检查网络连接后重试。"
    if "Connection" in msg or "connection" in msg:
        return "网络连接异常，请检查网络后重试。"
    if "rate limit" in msg.lower() or "429" in msg:
        return "请求过于频繁，已触发限流，请稍后再试。"
    if "insufficient" in msg.lower() or "quota" in msg.lower() or "余额" in msg:
        return "API 额度不足，请充值后再使用。"
    return f"处理失败：{msg}（如需排查请查看后端日志）"


def _stats_markdown(info: dict) -> str:
    """检测结果统计卡片"""
    return (
        "| 🧑 检测目标 | ✅ 已佩戴 | ❌ 未佩戴 | ⚠️ 待人工复核 |\n"
        "|:---:|:---:|:---:|:---:|\n"
        f"| **{info['total_targets']}** | **{info['helmet_count']}** "
        f"| **{info['no_helmet_count']}** | **{info['low_confidence_count']}** |"
    )


def _risk_assessment(info: dict) -> str:
    """基于检测结果的规则化风险判定。"""
    total = info["total_targets"]
    no_helmet = info["no_helmet_count"]
    low_conf = info["low_confidence_count"]
    helmet = info["helmet_count"]

    if total == 0:
        return (
            "⚪ **风险判定**：未检测到人员目标，可能画面中无人员或置信度过低，"
            "建议调整图片或提高置信度阈值后重新检测。"
        )

    if no_helmet > 0:
        level = "🔴 高风险"
        advice = (
            f"检测到 **{no_helmet}** 人未佩戴头盔，存在明显安全隐患。"
            "建议现场劝导并依据地方性法规给予警告或 20-50 元罚款；"
            "未佩戴头盔是导致颅脑损伤致死的主要原因之一。"
        )
    elif low_conf > 0:
        level = "🟡 待复核"
        advice = (
            f"已检测到 **{helmet}** 人佩戴头盔，但有 **{low_conf}** 个目标置信度不足，"
            "AI 识别结果仅供参考，建议移交人工复核确认后再作处置。"
        )
    else:
        level = "🟢 合规"
        advice = (
            f"画面中 **{helmet}** 人均已规范佩戴头盔，符合安全要求。"
            "请继续保持，注意头盔需系紧下颌带且具备 3C 认证。"
        )

    return f"### {level}\n{advice}\n\n> 提醒：AI 识别结果仅作参考，不能直接单独作为执法凭证。"


# ---------------- 头盔检测 Tab 回调 ----------------

def on_detect(image, conf):
    """执行 YOLO 检测，输出标注图 + 统计 + 规则化风险判定。"""
    if image is None:
        yield None, "", "⚪ **风险判定**：等待上传图片进行检测..."
        return

    # 运行状态提示
    yield None, "🔍 正在执行 YOLOv8 头盔检测，请稍候……", ""

    try:
        annotated, info = detect(image, conf=conf)
    except FileNotFoundError as exc:
        yield None, f"❌ 检测失败：{exc}", ""
        return
    except Exception as exc:
        yield None, f"❌ 检测失败：{_friendly_error(exc)}", ""
        return

    stats = _stats_markdown(info)
    risk = _risk_assessment(info)
    yield annotated, stats, risk


# ---------------- 法规问答 Tab 回调 ----------------

def _history_to_pairs(history):
    """把 messages 格式的聊天记录转为 [(问, 答), ...]。"""
    pairs = []
    msgs = history or []
    for i in range(0, len(msgs) - 1, 2):
        pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return pairs


def on_chat(question, history):
    """流式问答：逐 chunk 更新聊天记录，实现前端打字效果。"""
    question = (question or "").strip()
    history = history or []

    if not question:
        yield "", history
        return

    # 未配置 API Key 的友好提示
    api_key = get_deepseek_api_key()
    if not api_key:
        history = history + [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "⚠️ 未配置 DeepSeek API Key，请在项目根目录 `.env` 中"
                "填写 `DEEPSEEK_API_KEY=sk-xxxx` 后重启应用。",
            },
        ]
        yield "", history
        return

    # 先写入用户消息 + 占位助手消息
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": '<span class="waiting-animation">正在检索知识库并生成回答，请稍候……</span>'},
    ]
    yield "", history

    # 清空占位，开始流式输出
    history[-1]["content"] = ""
    try:
        for chunk in answer_question_stream(
            question, api_key, history=_history_to_pairs(history[:-2])
        ):
            history[-1]["content"] += chunk
            yield "", history
    except Exception as exc:
        history[-1]["content"] = f"❌ {_friendly_error(exc)}"
        yield "", history


# ---------------- 界面构建 ----------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="电动自行车头盔智能巡检系统", css=CUSTOM_CSS) as app:
        gr.Markdown("# 电动自行车头盔智能巡检系统")

        with gr.Tab("头盔检测"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="equal-height-column"):
                    # 给上传图片增加 fixed-image-box 类，锁定高度
                    input_image = gr.Image(label="请在此上传巡检图片", type="numpy", elem_classes="fixed-image-box")
                    conf_slider = gr.Slider(
                        0.1, 0.9, value=0.4, step=0.05,
                        label="检测置信度阈值（conf）",
                    )
                    detect_btn = gr.Button("🔍 开始检测", variant="primary")
                    gr.Examples(
                        examples=["asset/未佩戴.png", "asset/佩戴头盔.png"],
                        inputs=input_image,
                        label="示例图片",
                    )
                with gr.Column(scale=1, elem_classes="equal-height-column right-panel"):
                    # 结果图片同样锁定相同高度
                    output_image = gr.Image(label="检测结果（标注图）", elem_classes="fixed-image-box")
                    with gr.Column(elem_classes="stats-wrapper"):
                        stats_md = gr.Markdown(elem_classes="stats-box")
                        risk_md = gr.Markdown(
                            value="⚪ **风险判定**：等待上传图片进行检测...", 
                            label="风险判定与处置建议", 
                            elem_classes="risk-box"
                        )

        with gr.Tab("交通法规问答"):
            with gr.Column(elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    label="交通法规问答聊天框",
                    elem_classes="chatbot-area",
                    placeholder="例如：未佩戴头盔会怎么处罚？什么样的头盔才算合格？",
                    sanitize_html=False,
                )
                with gr.Row(elem_classes="input-row"):
                    question_box = gr.Textbox(
                        label="请输入问题",
                        placeholder="按回车或点击按钮发送……",
                        scale=4,
                        show_label=False,
                        container=False,
                        elem_id="question-input",
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                gr.Examples(
                    examples=[
                        "未佩戴安全头盔一般会受到什么处罚？",
                        "佩戴头盔能降低多少事故伤亡风险？",
                        "什么样的头盔才是合格头盔？",
                        "AI 识别结果可以直接作为执法凭证吗？",
                    ],
                    inputs=question_box,
                )

        detect_btn.click(
            on_detect,
            inputs=[input_image, conf_slider],
            outputs=[output_image, stats_md, risk_md],
        )
        send_btn.click(
            on_chat,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )
        question_box.submit(
            on_chat,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )

    return app


app = build_ui()
if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())