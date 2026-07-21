"""程序入口 - 交互式 RAG 问答终端

支持命令:
  /ingest    - 重新入库文档（加载 -> 切分 -> 向量化）
  /help      - 显示帮助
  /quit      - 退出程序
  其他输入   - 作为问题发起检索问答
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from deepseek_client import DeepSeekClient
from rag_core import ingest_documents, retrieve_context, HALLUCINATION_SUPPRESSION_PROMPT


def print_banner():
    """打印启动欢迎信息。"""
    banner = """
============================================================
      Local RAG 知识库问答系统 v1.0
      DeepSeek-V4 + Chroma + LangChain
============================================================
"""
    print(banner)
    print("支持命令:  /ingest - 重新入库文档")
    print("           /help   - 显示帮助")
    print("           /quit   - 退出程序")
    print("其他输入将作为问题，基于知识库检索后回答。\n")


def print_help():
    """打印帮助信息。"""
    print("\n" + "=" * 50)
    print("使用说明")
    print("=" * 50)
    print("1. 将你的 PDF/TXT 文档放入 data/docs/ 目录")
    print("2. 输入 /ingest 将文档向量化入库（首次使用必须先执行）")
    print("3. 直接输入问题，系统将检索知识库并给出回答")
    print("4. 输入 /quit 退出程序")
    print("=" * 50 + "\n")


def main():
    """交互式问答主循环。"""
    print_banner()

    try:
        llm = DeepSeekClient()
        print("[系统] DeepSeek 客户端初始化成功\n")
    except ValueError as e:
        print(f"[错误] {e}")
        print("[提示] 请将 .env.example 复制为 .env 并填入你的 DeepSeek API Key")
        sys.exit(1)

    print("[系统] 准备就绪，请输入问题（首次使用请先执行 /ingest 入库文档）\n")

    while True:
        try:
            user_input = input("? 你 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 再见！")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("[系统] 再见！")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/ingest":
            count = ingest_documents()
            if count > 0:
                print(f"[系统] 入库完成，共 {count} 个文本块，现在可以开始提问了！\n")
            else:
                print("[系统] 未找到文档，请先将 PDF/TXT 文件放入 data/docs/ 目录。\n")
            continue

        # ===== 正常问答流程 =====
        print("\n[系统] 正在检索知识库...")

        retrieved_docs = retrieve_context(user_input)
        if not retrieved_docs:
            print("[系统] 知识库中未找到相关文档。请先执行 /ingest 入库文档。\n")
            continue

        context_text = "\n\n---\n\n".join(
            f"[来源: {doc.metadata.get('source_file', 'unknown')}] {doc.page_content}"
            for doc in retrieved_docs
        )

        print(f"[系统] 检索到 {len(retrieved_docs)} 个相关片段")
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source_file", "unknown")
            print(f"       [{i}] {src}: {doc.page_content[:60]}...")

        print("\n[系统] 正在调用 DeepSeek-V4 生成回答...\n")
        try:
            reply = llm.chat_with_context(
                system_prompt=HALLUCINATION_SUPPRESSION_PROMPT,
                user_question=user_input,
                context=context_text,
            )
            print(f"? DeepSeek: {reply}\n")
        except (ConnectionError, RuntimeError) as e:
            print(f"[错误] API 调用失败: {e}\n")


if __name__ == "__main__":
    main()