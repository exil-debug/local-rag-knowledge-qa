"""统一配置管理模块

将分块大小、检索参数、模型名称等核心参数集中管理，避免硬编码散落在各个模块中。
每个参数均包含调优说明，方便根据实际场景调整。

使用方式：
    from config import config
    chunk_size = config.chunk_size
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(key, default=""):
    """安全读取环境变量，返回 strip 后的值或默认值。"""
    return os.getenv(key, default).strip()


@dataclass
class AppConfig:
    """应用全局配置项。

    所有参数均提供默认值，支持通过 .env 文件覆盖。
    参数注释说明了调整影响和默认值的选型理由。
    """

    # ---- DeepSeek API 配置 ----
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ---- 文本分块参数 ----
    # chunk_size: 每个文本块的目标字符数
    #   值越小 → 检索粒度越细，但上下文碎片化严重
    #   值越大 → 上下文完整，但块内可能混入噪声
    #   中文场景推荐 500-800；英文可适当放大到 1000
    chunk_size: int = 500

    # chunk_overlap: 相邻块之间的重叠字符数
    #   适当重叠可避免关键信息被切分到两块边界导致遗漏
    #   推荐 chunk_size 的 10%-20%（即 50-160 字符）
    chunk_overlap: int = 100

    # ---- 检索参数 ----
    # retrieval_top_k: 每次检索返回的最相关文本块数量
    #   Top-K 越大召回越全但噪声越多，越小答案越聚焦
    #   小规模知识库推荐 3-5，大规模可适当增加
    retrieval_top_k: int = 4

    # similarity_threshold: 相关性得分过滤阈值
    #   低于此阈值的文档被视为无关并丢弃
    #   阈值过高可能导致无结果，过低则引入噪声
    similarity_threshold: float = 0.3

    # ---- 嵌入模型 ----
    # 中文推荐 BAAI/bge-small-zh-v1.5（轻量）或 bge-base-zh-v1.5（精准）
    # 英文推荐 sentence-transformers/all-MiniLM-L6-v2
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    # ---- 持久化路径 ----
    chroma_db_dir: str = "./chroma_db"
    docs_dir: str = "./data/docs"

    @classmethod
    def load(cls):
        """从环境变量加载配置，未设置的项使用默认值。"""
        cfg = cls()
        cfg.deepseek_api_key = _env("DEEPSEEK_API_KEY", "")
        cfg.deepseek_api_base = _env("DEEPSEEK_API_BASE", cfg.deepseek_api_base)
        cfg.deepseek_model = _env("DEEPSEEK_MODEL", cfg.deepseek_model)
        if v := _env("CHUNK_SIZE"):
            cfg.chunk_size = int(v)
        if v := _env("CHUNK_OVERLAP"):
            cfg.chunk_overlap = int(v)
        if v := _env("RETRIEVAL_TOP_K"):
            cfg.retrieval_top_k = int(v)
        if v := _env("SIMILARITY_THRESHOLD"):
            cfg.similarity_threshold = float(v)
        if v := _env("EMBEDDING_MODEL_NAME"):
            cfg.embedding_model_name = v
        if v := _env("CHROMA_DB_DIR"):
            cfg.chroma_db_dir = v
        if v := _env("DOCS_DIR"):
            cfg.docs_dir = v
        return cfg

    def validate(self):
        """校验配置项是否在合理范围内，返回错误列表。"""
        errors = []
        if not self.deepseek_api_key:
            errors.append("DEEPSEEK_API_KEY 未设置")
        if self.chunk_size < 100 or self.chunk_size > 4000:
            errors.append("CHUNK_SIZE 超出建议范围 (100-4000)")
        if self.chunk_overlap < 0 or self.chunk_overlap > self.chunk_size // 2:
            errors.append("CHUNK_OVERLAP 超出建议范围 (0-chunk_size/2)")
        if self.retrieval_top_k < 1 or self.retrieval_top_k > 20:
            errors.append("RETRIEVAL_TOP_K 超出建议范围 (1-20)")
        if not (0 < self.similarity_threshold <= 1.0):
            errors.append("SIMILARITY_THRESHOLD 超出范围 (0-1)")
        return errors


# 全局单例配置，首次导入时自动加载
config = AppConfig.load()