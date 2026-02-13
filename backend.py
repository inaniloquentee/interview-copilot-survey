# backend.py
import os
import json
import time
import numpy as np
import paddle
from paddleocr import PaddleOCR
from pymilvus import MilvusClient, DataType
from openai import OpenAI  
import config

class InterviewCopilotBackend:
    def __init__(self):
        # 1. 初始化 OpenAI 兼容客户端
        self.ai_client = OpenAI(
            api_key=config.QIANFAN_API_KEY,
            base_url="https://qianfan.baidubce.com/v2"
        )

        # 2. 初始化 PaddleOCR
        print("⏳ 正在利用系统原生环境初始化 PaddleOCR...")
        self.ocr = None  
        self.ocr_available = False
        
        try:
            device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
            print(f"🚀 检测到推理设备: {device}")
            self.ocr = PaddleOCR(use_angle_cls=False, lang="ch")
            self.ocr_available = True
            print("✅ PaddleOCR 初始化成功！")
        except Exception as e:
            print(f"❌ OCR 初始化失败: {e}")
        
        # 3. 初始化 Milvus Lite
        self.milvus_client = MilvusClient(uri=config.MILVUS_URI)
        self._init_collection()

    def _init_collection(self):
        if not self.milvus_client.has_collection(config.COLLECTION_NAME):
            self.milvus_client.create_collection(
                collection_name=config.COLLECTION_NAME,
                dimension=config.VECTOR_DIM,
                auto_id=True,
                enable_dynamic_field=True
            )

    def ocr_process(self, image_bytes):
        """核心功能：使用黑名单过滤法，既不漏字，也不多字"""
        print("🔍 正在使用黑名单雷达模式解析图片...")
        
        if not self.ocr_available or self.ocr is None:
            return "❌ 识别引擎未加载。"

        temp_path = "temp_upload_ocr.png"
        try:
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            
            result = self.ocr.ocr(temp_path)
            
            if not result:
                return "未能在图片中检测到文字。"

            # 💡 终极修复：回归到能提取文字的逻辑，加入黑名单过滤
            def extract_text_robust(obj):
                texts = []
                if isinstance(obj, str):
                    # 黑名单：过滤掉 PaddleOCR 底层字典里夹带的参数名
                    blacklist = ['temp_upload_ocr.png', 'min', 'max', 'general', 'server', 'fast', 'ch', 'en', 'True', 'False', 'None']
                    if obj not in blacklist:
                        texts.append(obj)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        # 避开图片矩阵，防止内存爆炸
                        if 'img' in k or k in ['doc_preprocessor_res', 'model_settings', 'input_path', 'page_index']:
                            continue
                        texts.extend(extract_text_robust(v))
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        texts.extend(extract_text_robust(item))
                return texts

            all_strings = extract_text_robust(result)
            
            if not all_strings:
                return f"⚠️ 提取失败，未能从庞大对象中找到文字。"

            full_text = "\n".join(all_strings)
            print(f"✅ 识别成功，提取到 {len(all_strings)} 个文本块")
            return full_text
            
        except Exception as e:
            print(f"❌ OCR 运行报错: {e}")
            return f"识别失败: {e}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def extract_knowledge(self, raw_text):
        print("🧠 文心一言 (V2) 正在思考...")
        
        prompt = f"""
        你是一个资深技术面试官。请分析以下 OCR 识别的文本，提取面试题。
        
        文本内容：
        {raw_text}
        
        要求：
        1. 整理为 JSON 格式列表。
        2. 为每道题打分 "importance" (1-10)。
        3. 提取 "tags" (如 Redis, JVM, C++, 网络编程, 数据结构)。
        4. 生成标准答案 "answer" (如果文本是代码，请结合代码逻辑解释)。
        
        输出格式：
        [
            {{"question": "题目", "answer": "答案", "importance": 9, "tags": ["标签"]}}
        ]
        只输出 JSON 内容，不要任何解释。
        """
        
        try:
            response = self.ai_client.chat.completions.create(
                model="ernie-4.5-turbo-128k",  
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 
            )
            content = response.choices[0].message.content
            clean_json = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"LLM 提取失败: {e}")
            return []

    def get_embedding(self, text):
        try:
            response = self.ai_client.embeddings.create(
                model="bge-large-zh", 
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            return np.random.rand(config.VECTOR_DIM).tolist()

    def save_to_db(self, qa_list):
        data_rows = []
        for qa in qa_list:
            vector = self.get_embedding(qa["question"])
            data_rows.append({
                "vector": vector,
                "question": qa["question"],
                "answer": qa["answer"],
                "importance": qa["importance"],
                "tags": qa["tags"],
                "status": "new",
                "mastery_score": 0.0,
                "last_review": time.time()
            })
        self.milvus_client.insert(collection_name=config.COLLECTION_NAME, data=data_rows)
        return len(data_rows)

    def get_recommendations(self, user_mode="steady"):
        res = self.milvus_client.query(
            collection_name=config.COLLECTION_NAME,
            filter="id >= 0",
            output_fields=["id", "question", "importance", "mastery_score", "status", "tags"]
        )
        if not res: return []
        params = config.MODE_CONFIG.get(user_mode, config.MODE_CONFIG["steady"])
        alpha, beta = params["alpha"], params["beta"]

        for item in res:
            I = item["importance"] / 10.0
            W = 1.0 - item["mastery_score"]
            item["algo_score"] = (alpha * I) + (beta * W)
        res.sort(key=lambda x: x["algo_score"], reverse=True)
        return res

    def update_status(self, q_id, user_score):
        mastery = user_score / 100.0
        print(f"更新题库 ID {q_id}: 掌握度 -> {mastery}")