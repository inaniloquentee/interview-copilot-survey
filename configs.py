# config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# --- 百度千帆 (ERNIE) 配置 ---
# 请去百度智能云控制台获取: https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application
QIANFAN_ACCESS_KEY = os.getenv("QIANFAN_ACCESS_KEY", "你的AccessKey")
QIANFAN_SECRET_KEY = os.getenv("QIANFAN_SECRET_KEY", "你的SecretKey")

# --- Milvus 向量数据库配置 ---
MILVUS_URI = "./milvus_interview.db" # Lite版本直接生成本地文件
COLLECTION_NAME = "interview_questions"
VECTOR_DIM = 384  # 根据选用的 Embedding 模型维度决定 (本例使用轻量级模型)

# --- 推荐算法默认参数 ---
# 这里的权重对应策划书中的 Alpha 和 Beta
MODE_CONFIG = {
    "urgent": {"alpha": 0.8, "beta": 0.2},  # 火烧眉毛：重 ROI
    "steady": {"alpha": 0.4, "beta": 0.6}   # 按部就班：重查缺补漏
}