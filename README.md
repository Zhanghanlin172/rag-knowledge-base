# 🧠 AI 本地知识库 RAG 问答系统

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)

基于 **RAG（检索增强生成）** 架构的本地知识库问答系统。上传文档后自动建立向量索引，提问时通过语义检索找到相关段落，再由 LLM 生成精准回答。

> 🎯 **求职亮点：** Spring Boot（后端）+ LangChain（AI）+ ChromaDB（向量库）+ Docker（部署）全链路工程能力

![界面预览](https://img.shields.io/badge/💻-Streamlit_Web界面-FF4B4B)

## 核心功能

- 📄 **多格式文档上传**：PDF / TXT / Markdown
- 🔍 **语义检索**：基于 text2vec 中文 Embedding 模型，余弦相似度匹配
- 🤖 **LLM 增强生成**：支持 OpenAI API 或本地 Embedding 模式
- 💬 **对话式交互**：Streamlit 聊天界面，流式问答体验
- 🐳 **Docker 部署**：一键容器化运行
- 📊 **实时统计**：文档数、片段数可视化

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
streamlit run app.py

# 3. 打开浏览器 http://localhost:8501
```

### Docker 部署

```bash
docker build -t rag-qa .
docker run -p 8501:8501 rag-qa
```

## 技术架构

```
用户上传文档 (PDF/TXT)
        │
        ▼
  文档解析 + 文本切片 (RecursiveCharacterTextSplitter)
        │
        ▼
  向量化 Embedding (text2vec-base-chinese)
        │
        ▼
  存入 ChromaDB 向量数据库
        │
        ▼
  用户提问 → 语义检索 (similarity_search)
        │
        ▼
  检索结果 + 问题 → LLM 生成回答
        │
        ▼
  Streamlit 界面显示（含引用来源）
```

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | Streamlit（Python Web UI 框架） |
| AI 框架 | LangChain（RAG 编排） |
| 向量数据库 | ChromaDB（本地持久化） |
| Embedding | text2vec-base-chinese（HuggingFace） |
| LLM | OpenAI API / 本地模式 |
| 部署 | Docker + Streamlit |
