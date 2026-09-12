"""Gradio 交互界面：YOLOv8 头盔检测 + RAG 交通法规问答。

启动方式（在项目根目录 ultralytics-main/ 下）：
    conda activate langchain
    python -m rag_app.app

DeepSeek API Key 从项目根目录 .env 读取（DEEPSEEK_API_KEY=sk-xxxx）。
"""
import gradio as gr

from .config import get_deepseek_api_key
from .detector import detect, format_detection_summary
from .rag_chain import analyze_detection, answer_question



def _stats_markdown(info: dict) -> str:
    """检测结果统计卡片（Markdown 表格，不含逐目标明细）。"""
    return (
        "| 🧑 检测目标 | ✅ 已佩戴 | ❌ 未佩戴 | ⚠️ 待人工复核 |\n"
        "|:---:|:---:|:---:|:---:|\n"
        f"| **{info['total_targets']}** | **{info['helmet_count']}** "
        f"| **{info['no_helmet_count']}** | **{info['low_confidence_count']}** |"
    )


# ---------------- 头盔检测 Tab 回调 ----------------

def on_detect(image, conf):
    """执行检测并（可选）调用 RAG 生成巡检解读报告。"""
    if image is None:
        return None, "", ""
    annotated, info = detect(image, conf=conf)
    stats = _stats_markdown(info)

    api_key = get_deepseek_api_key()
    if not api_key:
        return annotated, stats, (
            "⚠️ 未配置 DeepSeek API Key，无法生成 AI 解读。\n\n"
            "请在项目根目录 `.env` 文件中填写 `DEEPSEEK_API_KEY=sk-xxxx` 后重启应用。"
        )
    try:
        report, _docs = analyze_detection(format_detection_summary(info), info, api_key)
        return annotated, stats, report
    except Exception as exc:  # 网络 / API 异常时保留检测结果
        return annotated, stats, f"AI 解读生成失败：{exc}"


# ---------------- 法规问答 Tab 回调 ----------------

def _history_to_pairs(history):
    """把 messages 格式的聊天记录转为 [(问, 答), ...]。"""
    pairs = []
    msgs = history or []
    for i in range(0, len(msgs) - 1, 2):
        pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return pairs


def on_chat(question, history):
    question = (question or "").strip()
    if not question:
        return "", history
    history = (history or []) + [{"role": "user", "content": question}]

    api_key = get_deepseek_api_key()
    if not api_key:
        history.append(
            {
                "role": "assistant",
                "content": "⚠️ 未配置 DeepSeek API Key，请在项目根目录 `.env` 中"
                "填写 `DEEPSEEK_API_KEY=sk-xxxx` 后重启应用。",
            }
        )
        return "", history

    try:
        answer, _docs = answer_question(
            question, api_key, history=_history_to_pairs(history)
        )
    except Exception as exc:
        answer = f"调用 DeepSeek API 失败：{exc}"
    history.append({"role": "assistant", "content": answer})
    return "", history


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="电动自行车头盔智能巡检系统") as app:
        gr.Markdown("# 🪖 电动自行车头盔智能巡检系统")

        with gr.Tab("🖼️ 头盔检测"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="请在此上传巡检图片", type="numpy")
                    conf_slider = gr.Slider(
                        0.1, 0.9, value=0.4, step=0.05,
                        label="检测置信度阈值（conf）",
                    )
                    detect_btn = gr.Button("🔍 开始检测并生成解读", variant="primary")
                    gr.Examples(
                        examples=["asset/未佩戴.png", "asset/佩戴头盔.png"],
                        inputs=input_image,
                        label="示例图片",
                    )
                with gr.Column():
                    output_image = gr.Image(label="检测结果（标注图）")
                    stats_md = gr.Markdown()
                    with gr.Accordion("� AI 巡检解读报告（点击展开）", open=False):
                        report_md = gr.Markdown()

        with gr.Tab("💬 交通法规问答"):
            chatbot = gr.Chatbot(
                label="交通法规问答聊天框",
                height=420,
                placeholder="例如：未佩戴头盔会怎么处罚？什么样的头盔才算合格？",
            )
            with gr.Row():
                question_box = gr.Textbox(
                    label="请输入问题",
                    placeholder="按回车或点击按钮发送……",
                    scale=4,
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
            outputs=[output_image, stats_md, report_md],
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
    app.launch(
        theme=gr.themes.Soft(),
    )
