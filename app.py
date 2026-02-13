# app.py
import streamlit as st
import pandas as pd
import numpy as np
from backend import InterviewCopilotBackend
import config

# 设置页面配置
st.set_page_config(page_title="Smart Interview Copilot", layout="wide", page_icon="🚀")

# 初始化 Backend (使用缓存避免重复加载模型)
@st.cache_resource
def get_backend():
    return InterviewCopilotBackend()

backend = get_backend()

# --- 侧边栏：全局设置 ---
with st.sidebar:
    st.title("⚙️ 备考设置")
    user_mode = st.radio(
        "当前备考模式",
        ("steady", "urgent"),
        format_func=lambda x: "📅 按部就班 (稳扎稳打)" if x == "steady" else "🔥 火烧眉毛 (只看高频)"
    )
    st.info(f"当前算法权重:\nAlpha(重要性): {config.MODE_CONFIG[user_mode]['alpha']}\nBeta(薄弱项): {config.MODE_CONFIG[user_mode]['beta']}")
    
    st.divider()
    st.write("📊 **总进度概览**")
    # 模拟数据
    st.progress(0.3, text="整体掌握率 30%")

# --- 主界面 Tabs ---
tab1, tab2, tab3 = st.tabs(["📤 智能导入 (Ingestion)", "📈 复习看板 (Dashboard)", "🤖 模拟面试 (Mock Interview)"])

# === Tab 1: 数据导入 ===
with tab1:
    st.header("知识库扩充")
    st.markdown("上传面试题截图，系统将自动识别并清洗入库。")
    
    uploaded_file = st.file_uploader("上传文件", type=['png', 'jpg', 'jpeg', 'pdf'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # 显示图片
        st.image(uploaded_file, caption="原始图片", width=300)
        
        if st.button("🚀 开始 AI 识别与提取"):
            with st.spinner("PaddleOCR 正在识别文字..."):
                raw_text = backend.ocr_process(uploaded_file.getvalue())
                st.session_state['raw_text'] = raw_text
                
            st.success("OCR 识别完成！")
            with st.expander("查看识别原文"):
                st.text(raw_text)
            
            with st.spinner("ERNIE 正在生成结构化题库..."):
                qa_list = backend.extract_knowledge(raw_text)
                st.session_state['qa_list'] = qa_list 

        if 'qa_list' in st.session_state and st.session_state['qa_list']:
            st.write("### 🧠 提取结果预览")
            df = pd.DataFrame(st.session_state['qa_list'])
            st.dataframe(df, use_container_width=True)
            
            if st.button("💾 确认入库"):
                count = backend.save_to_db(st.session_state['qa_list'])
                st.toast(f"成功存入 {count} 道面试题！", icon="✅")
                del st.session_state['qa_list']

# === Tab 2: 复习看板 ===
with tab2:
    st.header("🎯 今日智能推荐")
    st.caption(f"基于「{user_mode}」模式生成的动态优先级列表")
    
    # 获取推荐
    recommendations = backend.get_recommendations(user_mode)
    
    if not recommendations:
        st.info("题库为空，请先去 Tab 1 上传资料！")
    else:
        for idx, item in enumerate(recommendations):
            col1, col2, col3 = st.columns([1, 4, 2])
            with col1:
                st.metric("推荐分", f"{item['algo_score']:.2f}")
            with col2:
                st.subheader(f"{idx+1}. {item['question']}")
                # 兼容不同类型的重要性得分
                importance_score = int(item['importance']) if str(item['importance']).isdigit() else 5
                st.caption(f"标签: {item['tags']} | 考频: {'⭐' * importance_score}")
            with col3:
                status_color = "red" if item['mastery_score'] < 0.5 else "green"
                st.markdown(f"掌握度: :{status_color}[{item['mastery_score']*100:.0f}%]")
                
                # 💡 核心修复：移除 switch_page，改用状态通知
                if st.button("开始复习", key=f"btn_{item['id']}"):
                    # 如果换了一道新题，先清空上一次的聊天记录
                    if st.session_state.get('current_q', {}).get('id') != item['id']:
                        st.session_state.messages = []
                        
                    st.session_state['current_q'] = item
                    # 弹出右下角提示，引导用户点击 Tab 3
                    st.toast("✅ 题目已锁定！请点击上方「🤖 模拟面试」标签页开始作答", icon="👉")

# === Tab 3: 模拟面试 ===
with tab3:
    st.header("🤖 AI 面试官")
    
    # 检查是否有选中的题目
    if 'current_q' not in st.session_state:
        st.info("💡 请先从「复习看板」选择一道题开始。")
    else:
        q = st.session_state['current_q']
        st.info(f"**正在考察**：{q['question']}")
        
        # 聊天界面初始化
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 渲染历史对话
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 接收用户输入
        if user_input := st.chat_input("请输入你的回答..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 💡 核心升级：真正调用文心一言 API 进行智能打分和点评
            with st.chat_message("assistant"):
                with st.spinner("面试官正在仔细评估你的回答..."):
                    eval_prompt = f"""
                    你现在是技术面试官。
                    当前面试题目：{q['question']}
                    该题目的标准答案参考：{q['answer']}
                    
                    候选人的回答：{user_input}
                    
                    请你扮演严格但客观的面试官，给出以下内容：
                    1. 综合打分（满分 100 分，请加粗显示，如 **评分：85分**）
                    2. 点评（指出候选人回答正确的地方，以及欠缺或不准确的地方）
                    3. 改进建议（给出更完善的表述方式）
                    """
                    
                    try:
                        # 复用 backend 里的 ai_client 发起对话
                        response_obj = backend.ai_client.chat.completions.create(
                            model="ernie-4.5-turbo-128k",  
                            messages=[{"role": "user", "content": eval_prompt}],
                            temperature=0.3
                        )
                        response = response_obj.choices[0].message.content
                    except Exception as e:
                        response = f"**评分失败**\n\nAI 接口调用出错: {e}"
                        
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})