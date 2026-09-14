# 基于YOLOv8与RAG的头盔佩戴检测智能分析系统
该项目旨在实现道路监控抓拍图片的头盔佩戴检测，结合RAG检索增强生成技术，对模型检测结果进行智能文本解析，提供可视化Web交互界面。

## 项目概述
本项目针对道路监控抓拍图片，完成头盔佩戴检测任务。
项目设计：直接检测头部，分为`head`（未佩戴头盔）、`helmet`（佩戴头盔）两类。无需IoU框匹配，规避多人场景下目标错配问题，适合远距离监控小目标场景。

项目完整链路：数据集清洗与校验 → YOLO模型训练与指标评估 → 模型推理封装 → Gradio可视化Web界面 → RAG大模型智能分析。

### 核心功能
- 数据集自动化处理：编写脚本清洗低质量/损坏图片，校验图片与标注文件配对合法性，过滤无效样本。
- YOLOv8模型训练与评估：训练自定义检测模型，计算mAP50、mAP50-95、Precision、Recall指标评估模型性能。
- 结构化推理：封装推理模块，输出检测框、类别、置信度，对低置信目标标记人工复核提示。
- Web可视化交互：Gradio搭建网页端，上传图片即可运行检测并展示可视化结果。
- RAG智能解读：将YOLO检测结果送入RAG模块，自动生成检测报告、异常情况分析文本。

##  项目目录结构
```text
helmet-detect-rag/
├── app.py                  # Gradio Web 应用启动入口
├── requirements.txt
├── yolo-bvn.yaml
├── LICENSE
├── .gitignore
├── README.md
│
├── asset/                  # 演示图片等静态资源
├── examples/
├── knowledge_base/         # RAG 知识库原始文档
│
├── rag_app/                # 核心 Web 与 RAG 应用包
│   ├── __init__.py
│   ├── app.py              # Gradio 页面主业务代码
│   ├── config.py           # 项目全局配置
│   ├── detector.py         # YOLO 推理封装模块
│   ├── knowledge_base.py   # RAG 知识库管理
│   └── rag_chain.py        # RAG 检索与生成链路
│
├── scripts/                # 离线训练与数据处理脚本
│   ├── clearn_dataset.py   
│   ├── load_dataset.py
│   ├── train.py          
│   ├── verify.py         
│   ├── detect.py         
│   └── index_detect.py
│
└── tests/                  # 单元测试目录                
##  环境准备

### 1. 克隆仓库并安装依赖

```bash
# 拉取代码
git clone https://github.com/yuanji-chucheng/helmet-detect-rag.git
cd helmet-detect-rag

# 创建虚拟环境
python -m venv .venv

# Windows 激活虚拟环境
.venv\Scripts\activate

# Linux / Mac 激活虚拟环境
source .venv/bin/activate

# 安装全部依赖包
pip install -r requirements.txt
```

### 2. 配置 RAG 环境变量

在项目根目录新建 `.env` 文件，填入你的大模型 API 密钥，示例：

```env
LLM_API_KEY=你的密钥
LLM_BASE_URL=模型接口地址
```

### 3. 启动 Web 界面

```bash
# 方式一：直接启动
python gradio_app.py

# 方式二：开发调试，模块运行
python -m rag_app.app
```

---

## 完整模型训练复现流程

> 仓库不提供数据集与训练好的权重文件，需要自行准备 YOLO 标注数据集。

1. 将 YOLO 格式数据集放入项目 `bvn/helmet` 目录，并修改 `yolo-bvn.yaml` 配置。
2. 数据集清洗校验：

```bash
python scripts/clearn_dataset.py
```

3. 启动模型训练：

```bash
python scripts/train.py
```

4. 在验证集评估模型指标：

```bash
python scripts/verify.py
```

---

## 重要说明

- 本仓库不存放**数据集、训练权重、RAG 向量库**，需用户自行准备数据集、训练模型、配置大模型密钥。
- 项目默认面向**监控抓拍静态图片**；视频检测为扩展方向，可基于 `detector` 推理模块增加 OpenCV 帧读取与 ByteTrack 跟踪。
- AI 识别结果仅作为**辅助参考**，低置信目标建议人工复核。