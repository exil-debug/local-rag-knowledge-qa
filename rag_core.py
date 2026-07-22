"""RAG 核心模块：文档加载、向量存储与语义检索

完整 RAG 流程：文档加载 -> 文本分块 -> 向量化入库 -> 语义检索

依赖: langchain-community, chromadb, sentence-transformers, pypdf
"""

import os
import logging
from typing import List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import config

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------- 文档加载 ----------

# 常见文本编码列表，用于自动检测中文文档编码
_ENCODING_CANDIDATES = ["utf-8", "gbk", "gb2312", "gb18030", "utf-16", "latin-1"]


def _detect_and_read_text(filepath):
    """尝试多种编码读取文本文件，解决中文编码兼容问题。

    按优先级依次尝试常见编码，UTF-8 失败后自动回退到 GBK 等中文编码。
    所有编码均失败后抛出最后的异常，由上层调用方处理。
    """
    for enc in _ENCODING_CANDIDATES:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"无法解码文件 {filepath}，已尝试编码: {_ENCODING_CANDIDATES}")


def load_documents(docs_dir=None):
    """从指定目录加载所有 PDF 和 TXT 文件。

    参数说明：
        docs_dir: 文档目录路径，默认从 config 读取

    返回：LangChain Document 列表
    """
    if docs_dir is None:
        docs_dir = config.docs_dir
    docs_dir = os.path.abspath(docs_dir)
    if not os.path.isdir(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        logger.info("文档目录不存在，已自动创建: %s", docs_dir)
        return []
    documents = []
    supported_ext = (".pdf", ".txt")
    fail_count = 0
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
                text = _detect_and_read_text(filepath)
                docs = [Document(page_content=text, metadata={"source": filepath})]
            for doc in docs:
                doc.metadata["source_file"] = filename
                doc.metadata["source_path"] = filepath
            documents.extend(docs)
            logger.info("已加载 %s (%d 页)", filename, len(docs))
        except Exception as e:
            fail_count += 1
            logger.error("加载失败已跳过: %s - %s", filename, e)
    if fail_count > 0:
        logger.warning("本次入库共 %d 个文件加载失败，已自动跳过", fail_count)
    return documents


# ---------- 文本分块 ----------

def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """将文档按中文标点优先策略递归切分为固定大小的文本块。"""
    if not documents:
        return []
    if chunk_size is None:
        chunk_size = config.chunk_size
    if chunk_overlap is None:
        chunk_overlap = config.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "，", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("切分为 %d 个文本块 (chunk_size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks


# ---------- 向量化与存储 ----------

def get_embeddings(model_name=None):
    """获取 HuggingFace 嵌入模型实例。"""
    if model_name is None:
        model_name = config.embedding_model_name
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(persist_dir=None, embedding_model=None):
    """加载或创建 Chroma 向量数据库实例。"""
    if persist_dir is None:
        persist_dir = config.chroma_db_dir
    if embedding_model is None:
        embedding_model = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name="local_rag_knowledge",
    )


def ingest_documents(docs_dir=None, chunk_size=None, chunk_overlap=None, persist_dir=None):
    """完整的「加载 -> 切分 -> 入库」流水线。"""
    logger.info("=" * 50)
    logger.info("开始文档入库流程...")
    logger.info("=" * 50)
    docs = load_documents(docs_dir)
    if not docs:
        logger.warning("未找到任何文档")
        return 0
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    embedding_model = get_embeddings()
    if persist_dir is None:
        persist_dir = config.chroma_db_dir
    logger.info("向量化并写入 Chroma (持久化路径: %s) ...", persist_dir)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="local_rag_knowledge",
    )
    logger.info("入库完成，共 %d 个文本块", len(chunks))
    return len(chunks)


# ---------- 语义检索 ----------

def retrieve_context(query, k=None, store=None, persist_dir=None):
    """对用户问题进行语义检索，返回最相关的文档片段。"""
    if k is None:
        k = config.retrieval_top_k
    if store is None:
        if persist_dir is None:
            persist_dir = config.chroma_db_dir
        store = get_vector_store(persist_dir)
    results = store.similarity_search_with_relevance_scores(query, k=k)
    threshold = config.similarity_threshold
    filtered = [doc for doc, score in results if score >= threshold]
    if not filtered:
        # 空结果降级：全部被过滤时返回 Top2 作为兜底
        logger.info("检索结果全部低于相关性阈值 (%.2f)，降级返回 Top2", threshold)
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