"""
本地知识库 RAG 问答系统
================================
基于 LangChain + ChromaDB + Streamlit 构建
支持多文档上传、语义检索、LLM 增强生成

使用:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import os
import tempfile
import hashlib
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.schema import Document

# ---- 页面配置 ----
st.set_page_config(
    page_title="AI 知识库问答",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS ----
st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:800; background:linear-gradient(135deg,#667eea,#764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .stat-card { background:#f8f9fa; border-radius:12px; padding:20px; text-align:center; border:1px solid #e9ecef; }
    .stat-card .num { font-size:2rem; font-weight:700; color:#667eea; }
    .stat-card .label { font-size:0.8rem; color:#6c757d; margin-top:4px; }
    .chat-msg { padding:14px 18px; border-radius:12px; margin:8px 0; }
    .user-msg { background:#e7f1ff; border:1px solid #b6d4fe; }
    .ai-msg { background:#f8f9fa; border:1px solid #dee2e6; }
    .source-tag { display:inline-block; font-size:0.75rem; padding:2px 8px; background:#e9ecef; border-radius:4px; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ---- 初始化 ----
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

# ---- 侧边栏：文档管理 ----
with st.sidebar:
    st.markdown("### 📁 知识库管理")

    uploaded_files = st.file_uploader(
        "上传文档（PDF / TXT / MD）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="支持批量上传，自动创建向量索引"
    )

    # LLM 配置
    st.markdown("---")
    st.markdown("### 🤖 LLM 配置")
    llm_provider = st.selectbox("模型来源", ["本地 Embedding（无需 API）", "OpenAI API"])
    if "OpenAI" in llm_provider:
        openai_key = st.text_input("OpenAI API Key", type="password")
        model_name = st.selectbox("模型", ["gpt-3.5-turbo", "gpt-4o-mini"])
    else:
        openai_key = None
        model_name = "local"

    st.markdown("---")
    st.caption("💡 本地 Embedding 模型首次运行需下载，约 400MB")

# ---- 主界面 ----
st.markdown('<p class="main-header">🧠 AI 知识库问答系统</p>', unsafe_allow_html=True)
st.caption("上传文档 → 自动向量化 → 语义检索 → LLM 生成回答")

# 状态栏
col1, col2, col3, col4 = st.columns(4)

# 处理文档
embeddings = init_embeddings()
splitter = get_text_splitter()
persist_dir = "./chroma_db"

if uploaded_files:
    all_docs = []
    file_names = []

    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = f.name
        all_docs.extend(docs)
        file_names.append(f.name)
        os.unlink(tmp_path)

    # 切片
    chunks = splitter.split_documents(all_docs)

    # 向量化
    with st.spinner("正在建立索引..."):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectorstore.persist()

    st.session_state["vectorstore"] = vectorstore
    st.session_state["doc_count"] = len(file_names)
    st.session_state["chunk_count"] = len(chunks)

    col1.metric("已索引文档", len(file_names))
    col2.metric("文本片段", len(chunks))
    col3.metric("Embedding 模型", "text2vec-base")
    col4.metric("向量数据库", "ChromaDB")
else:
    # 检查已有索引
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            st.session_state["vectorstore"] = vectorstore
            col1.metric("状态", "已加载历史索引")
        except:
            col1.metric("状态", "等待上传文档")

# ---- 问答区 ----
st.markdown("---")
st.markdown("### 💬 知识库问答")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "ai-msg"
    st.markdown(f'<div class="chat-msg {css_class}"><b>{"👤 你" if msg["role"]=="user" else "🤖 AI"}</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

# 输入区
query = st.chat_input("输入问题，基于已上传的文档回答...")

if query:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="chat-msg user-msg"><b>👤 你</b><br>{query}</div>', unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.warning("请先上传文档")
    else:
        with st.spinner("检索中..."):
            vs = st.session_state["vectorstore"]
            retriever = vs.as_retriever(search_kwargs={"k": 4})

            # 语义检索
            docs = retriever.get_relevant_documents(query)

            # 构建上下文
            context_parts = []
            sources = set()
            for d in docs:
                src = d.metadata.get("source", "未知")
                sources.add(src)
                context_parts.append(f"[来源: {src}] {d.page_content}")

            context = "\n\n".join(context_parts)

            # 生成回答
            if openai_key and "OpenAI" in llm_provider:
                llm = OpenAI(api_key=openai_key, model_name=model_name, temperature=0.3)
                prompt = f"""你是一个知识库助手。请严格基于以下文档内容回答问题。如果文档中没有相关信息，请明确说"文档中未找到相关信息"。

文档内容：
{context}

问题：{query}

回答："""
                answer = llm(prompt)
            else:
                # 无 LLM 模式：直接返回检索结果拼接
                answer = f"📚 从 {len(docs)} 个相关片段中检索到的信息：\n\n"
                for i, d in enumerate(docs):
                    preview = d.page_content[:250].replace("\n", " ")
                    answer += f"{i+1}. {preview}...\n\n"

            # 显示来源
            st.session_state.messages.append({"role": "ai", "content": answer})
            st.markdown(f'<div class="chat-msg ai-msg"><b>🤖 AI</b><br>{answer}</div>', unsafe_allow_html=True)

            # 显示引用来源
            st.caption(f"📎 引用来源：")
            cols = st.columns(len(sources))
            for i, src in enumerate(sources):
                cols[i].markdown(f'<span class="source-tag">📄 {src}</span>', unsafe_allow_html=True)

# ---- Footer ----
st.markdown("---")
st.caption("技术栈：LangChain · ChromaDB · Streamlit · HuggingFace Embeddings · sentence-transformers")
