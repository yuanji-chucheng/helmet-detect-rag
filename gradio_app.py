# 启动入口
from rag_app.app import app

if __name__ == "__main__":
    # 部署服务器关键参数：server_name="0.0.0.0"，监听所有网卡
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )