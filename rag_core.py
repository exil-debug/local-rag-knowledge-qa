"""
RAG 核心模块：文档加载与文本分块预处理

负责从本地目录加载 PDF/TXT 文档，完成格式兼容处理后，
按中文标点优先策略进行递归文本分块，为后续向量化做准备。

依赖: langchain-community, pypdf
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()


def load_documents(docs_dir=None):
    """
    从指定目录加载所有 PDF 和 TXT 文件。
    
    Args:
        docs_dir: 文档目录路径
    Returns:
        Document 列表
    """
    if docs_dir is None:
        docs_dir = os.getenv("DOCS_DIR", "./data/docs")
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


def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    将文档按中文标点优先策略递归切分为固定大小的文本块。
    
    Args:
        documents: 原始文档列表
        chunk_size: 每块字符数
        chunk_overlap: 块间重叠字符数
    Returns:
        切分后的 Document 列表
    """
    if not documents:
        return []
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", ". ", "! ", "? ", "\uff0c", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"  -> 切分为 {len(chunks)} 个文本块")
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        print(f"总共加载 {len(docs)} 页，切分为 {len(chunks)} 块")
    else:
        print("未找到文档")
