from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import tempfile
import json

app = FastAPI(title="RAG 知识库问答系统", version="1.0")
client = chromadb.Client(Settings(persist_directory="./chroma_data", is_persistent=True))
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vectorstore = Chroma(client=client, embedding_function=embeddings, collection_name="docs")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    id: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并建立向量索引"""
    content = ""
    if file.filename.endswith('.pdf'):
        try:
            from pypdf import PdfReader
            with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
                tmp.write(await file.read())
                reader = PdfReader(tmp.name)
                for page in reader.pages:
                    t = page.extract_text()
                    if t: content += t
        except ImportError:
            content = "PDF支持需要安装 pypdf 库"
    elif file.filename.endswith('.txt') or file.filename.endswith('.md'):
        content = (await file.read()).decode('utf-8', errors='ignore')
    else:
        raise HTTPException(400, "仅支持 PDF、TXT、MD 格式")

    if len(content.strip()) < 50:
        raise HTTPException(400, "文档内容过少")

    # 切片
    chunks = text_splitter.split_text(content)
    docs = [Document(page_content=c, metadata={"filename": file.filename, "chunk_id": i}) for i, c in enumerate(chunks) if c.strip()]

    # 向量化存储
    ids = vectorstore.add_documents(docs)

    return {
        "success": True,
        "filename": file.filename,
        "chunks": len(chunks),
        "message": f"已索引 {len(chunks)} 个文本片段"
    }

@app.post("/query")
async def query(req: QueryRequest):
    """语义检索 + RAG 生成回答"""
    # 相似度检索
    results = vectorstore.similarity_search_with_score(req.question, k=req.top_k)

    if not results:
        return {"answer": "未找到相关文档内容", "sources": []}

    # 构建上下文
    context_parts = []
    for doc, score in results:
        filename = doc.metadata.get('filename', '未知')
        context_parts.append(f"[来源: {filename}] {doc.page_content}")

    context = "\n\n".join(context_parts)

    # 生成回答（简化版：直接拼接上下文）
    # 实际应用中可接入 LLM API
    answer = f"根据知识库中的 {len(results)} 条相关内容：\n\n"
    for i, (doc, score) in enumerate(results):
        answer += f"{i+1}. [{doc.metadata.get('filename','')}] {doc.page_content[:200]}...\n"

    return {
        "answer": answer,
        "sources": [{"filename": doc.metadata.get('filename',''), "preview": doc.page_content[:150]} for doc, _ in results],
        "total_found": len(results)
    }

@app.get("/status")
async def status():
    """查看知识库状态"""
    try:
        count = vectorstore._collection.count()
    except:
        count = 0
    return {"status": "ok", "documents_indexed": count}

@app.delete("/clear")
async def clear():
    """清空知识库"""
    try:
        client.delete_collection("docs")
        global vectorstore
        vectorstore = Chroma(client=client, embedding_function=embeddings, collection_name="docs")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
