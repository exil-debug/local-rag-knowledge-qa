"""RAG 系统基础功能测试

包含两个测试用例：
1. 文档分块测试：验证中文标点优先分块逻辑是否正确
2. 单轮问答测试：验证检索+生成流程是否正常（需配置 API Key）

使用方法：
    python test_basic.py
"""

import os
import sys

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_chunking():
    """测试文档分块逻辑：验证中文标点优先切分是否符合预期。"""
    print("=" * 50)
    print("测试 1：文档分块测试")
    print("=" * 50)

    from langchain.schema import Document
    from rag_core import split_documents

    # 模拟一段中文文本，包含句号、问号、感叹号、逗号
    sample_text = """深度学习是机器学习的一个分支。它通过多层神经网络来学习数据的层次化特征表示。
这一技术在图像识别、自然语言处理等领域取得了显著成果。你知道它为什么叫「深度」学习吗？
因为它使用了多个隐藏层！这就是名称的由来。此外，强化学习是另一个重要的机器学习范式。"""

    docs = [Document(page_content=sample_text, metadata={"source_file": "test.txt"})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

    print(f"  原始文本长度: {len(sample_text)} 字符")
    print(f"  切分后块数: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  块 {i} ({len(chunk.page_content)} 字符):")
        print(f"    {chunk.page_content[:80]}...")

    # 验证：分块后至少包含 2 个块
    assert len(chunks) >= 2, f"分块数量不足: {len(chunks)}"
    print(f"\n  ✓ 分块测试通过 ({len(chunks)} 个块)")


def test_qa_pipeline():
    """测试单轮问答流程：验证检索+生成是否连通（跳过无 API Key 场景）。"""
    print("\n" + "=" * 50)
    print("测试 2：单轮问答测试")
    print("=" * 50)

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("  ⚠ 跳过：未检测到有效的 DEEPSEEK_API_KEY")
        print("  如需运行此测试，请配置 .env 文件后重试")
        return

    from deepseek_client import DeepSeekClient
    from rag_core import retrieve_context, HALLUCINATION_SUPPRESSION_PROMPT

    # 先确保知识库中有数据
    from rag_core import ingest_documents as do_ingest
    chunk_count = do_ingest()
    if chunk_count == 0:
        print("  ⚠ 跳过：知识库为空，无文档可供检索")
        print("  请先将 PDF/TXT 文件放入 data/docs/ 目录后重试")
        return

    # 执行检索
    question = "深度学习是什么？"
    print(f"  问题: {question}")

    docs = retrieve_context(question)
    print(f"  检索到 {len(docs)} 个相关片段")

    if not docs:
        print("  ⚠ 跳过生成：未检索到相关内容")
        return

    context_text = "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source_file', 'unknown')}] {doc.page_content}"
        for doc in docs
    )

    # 调用 DeepSeek
    llm = DeepSeekClient()
    try:
        answer = llm.chat_with_context(
            system_prompt=HALLUCINATION_SUPPRESSION_PROMPT,
            user_question=question,
            context=context_text,
        )
        print(f"  回答: {answer[:200]}")
        print(f"\n  ✓ 问答测试通过")
    except (ConnectionError, RuntimeError) as e:
        print(f"  ✗ 问答失败: {e}")


if __name__ == "__main__":
    print("RAG 系统基础功能测试\n")

    test_chunking()
    test_qa_pipeline()

    print("\n" + "=" * 50)
    print("测试结束")
    print("=" * 50)