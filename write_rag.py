"""
RAG 核心模块：文档加载、向量存储与语义检索

完整 RAG 流程：文档加载 -> 文本分块 -> 向量化入库 -> 语义检索

依赖: langchain-community, chromadb, sentence-transformers, pypdf
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def _get_env(key, default=""):
    """安全读取环境变量，返回 strip 后的值或默认值。"""
    return os.getenv(key, default).strip()


# ---------- 文档加载 ----------

def load_documents(docs_dir=None):
    """
    从指定目录加载所有 PDF 和 TXT 文件。
    
    参数说明：
        docs_dir: 文档目录路径，默认从环境变量 DOCS_DIR 读取
    
    返回：LangChain Document 列表
    """
    if docs_dir is None:
        docs_dir = _get_env("DOCS_DIR", "./data/docs")
    docs_dir = os.path.abspath(docs_dir)
    if not os.path.isdir(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        return []
    documents = []
    supported_ext = (".pdf", ".txt")
    for filename in sorted(os.listdir(docs_dir)):
        filepath = os.path.join(docs_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_ext:
            continue
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
                docs = loader.load()
            else:
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
                doc.metadata["source_path"] = filepath
            documents.extend(docs)
            print(f"  OK 已加载 {filename} ({len(docs)} 页)")
        except Exception as e:
            print(f"  FAIL 加载失败: {filename} - {e}")
    return documents


# ---------- 文本分块 ----------

def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    将文档按中文标点优先策略递归切分为固定大小的文本块。
    
    参数说明：
        documents: 原始文档列表
        chunk_size: 每块字符数
        chunk_overlap: 块间重叠字符数
    
    返回：切分后的 Document 列表
    """
    if not documents:
        return []
    if chunk_size is None:
        chunk_size = int(_get_env("CHUNK_SIZE", "500"))
    if chunk_overlap is None:
        chunk_overlap = int(_get_env("CHUNK_OVERLAP", "100"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", ". ", "! ", "? ", "\uff0c", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"  -> 切分为 {len(chunks)} 个文本块")
    return chunks


# ---------- 向量化与存储 ----------

def get_embeddings(model_name=None):
    """获取 HuggingFace 嵌入模型实例。"""
    if model_name is None:
        model_name = _get_env("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(persist_dir=None, embedding_model=None):
    """加载或创建 Chroma 向量数据库实例。"""
    if persist_dir is None:
        persist_dir = _get_env("CHROMA_DB_DIR", "./chroma_db")
    if embedding_model is None:
        embedding_model = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name="local_rag_knowledge",
    )


def ingest_documents(docs_dir=None, chunk_size=None, chunk_overlap=None, persist_dir=None):
    """
    完整的「加载 -> 切分 -> 入库」流水线。
    
    返回：入库的文本块数量
    """
    print("=" * 50)
    print("开始文档入库流程...")
    print("=" * 50)
    docs = load_documents(docs_dir)
    if not docs:
        print("未找到任何文档")
        return 0
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    embedding_model = get_embeddings()
    if persist_dir is None:
        persist_dir = _get_env("CHROMA_DB_DIR", "./chroma_db")
    print(f"  -> 向量化并写入 Chroma ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="local_rag_knowledge",
    )
    print(f"  OK 入库完成，共 {len(chunks)} 个文本块")
    return len(chunks)


# ---------- 语义检索 ----------

def retrieve_context(query, k=None, store=None, persist_dir=None):
    """
    对用户问题进行语义检索，返回最相关的 k 个文档片段。
    
    返回：按相关性降序排列的 Document 列表
    """
    if k is None:
        k = int(_get_env("RETRIEVAL_TOP_K", "4"))
    if store is None:
        if persist_dir is None:
            persist_dir = _get_env("CHROMA_DB_DIR", "./chroma_db")
        store = get_vector_store(persist_dir)
    results = store.similarity_search_with_relevance_scores(query, k=k)
    filtered = [doc for doc, score in results if score >= 0.3]
    if not filtered:
        filtered = [doc for doc, _ in results[:2]]
    return filtered


# ---------- 幻觉抑制 Prompt 模板 ----------

HALLUCINATION_SUPPRESSION_PROMPT = """你是专业、严谨的知识库问答助手。

约束规则：
1. 仅使用参考文档中的信息进行回答
2. 文档信息不足时明确告知无法回答
3. 禁止编造文档中未包含的内容

【参考文档】
{context}
"""


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ingest"
    if cmd == "ingest":
        ingest_documents()
    elif cmd == "retrieve":
        q = sys.argv[2] if len(sys.argv) > 2 else "测试问题"
        docs = retrieve_context(q)
        print(f"检索到 {len(docs)} 个相关片段")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source_file", "unknown")
            print(f"--- 片段 {i} (来源: {src}) ---")
            print(d.page_content[:200])
            print()
