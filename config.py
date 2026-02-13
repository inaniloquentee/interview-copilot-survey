# config.py
# 💡 你的 ALTAK 密钥
QIANFAN_API_KEY = "bce-v3/ALTAK-duMhSEgOXBn5oCOT6xOtG/a7cc2e4f79f972d8e3c1cb55d80998a824fc377c"

MILVUS_URI = "interview_copilot.db"
COLLECTION_NAME = "interview_qa"
VECTOR_DIM = 1024  # 💡 必须是 1024，因为你代码里用了 bge-large-zh 模型

MODE_CONFIG = {
    "steady": {"alpha": 0.4, "beta": 0.6},
    "urgent": {"alpha": 0.8, "beta": 0.2}
}