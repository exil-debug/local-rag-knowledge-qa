"""自动化问答评估工具

批量测试 RAG 系统的召回与生成质量。
从评估数据文件（JSON格式）读取问题列表，逐条执行检索+生成回答，
并输出评估报告。

评估数据文件格式 (eval_questions.json):
[
    {"question": "什么是机器学习？", "expected_answer": "机器学习是..."},
    {"question": "...", "expected_answer": "..."}
]
"""

import json
import os
import sys
import time
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from deepseek_client import DeepSeekClient
from rag_core import retrieve_context, HALLUCINATION_SUPPRESSION_PROMPT


def load_eval_questions(filepath):
    """加载评估问题 JSON 文件。"""
    if not os.path.isfile(filepath):
        print(f"[错误] 评估数据文件不存在: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        print("[错误] 评估数据文件应为非空列表")
        sys.exit(1)
    return data


def run_eval(eval_file="eval_questions.json", output_file="eval_report.md"):
    """执行完整评估流程。"""
    print("=" * 60)
    print("   RAG 系统自动化评估工具")
    print("=" * 60)

    try:
        llm = DeepSeekClient()
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    questions = load_eval_questions(eval_file)
    print(f"\n加载了 {len(questions)} 个评估问题\n")

    results = []
    success_count = 0
    fail_count = 0

    for idx, item in enumerate(questions, 1):
        question = item.get("question", "").strip()
        expected = item.get("expected_answer", "")
        if not question:
            print(f"  [{idx}] 跳过空问题")
            continue

        preview = question[:60] + ("..." if len(question) > 60 else "")
        print(f"\n[{idx}/{len(questions)}] 问题: {preview}")

        t0 = time.time()
        retrieved_docs = retrieve_context(question)
        retrieval_time = time.time() - t0

        context_text = "\n\n---\n\n".join(
            f"[来源: {doc.metadata.get('source_file', 'unknown')}] {doc.page_content}"
            for doc in retrieved_docs
        )

        retrieval_count = len(retrieved_docs)
        print(f"    检索到 {retrieval_count} 个片段 ({retrieval_time:.2f}s)")

        t1 = time.time()
        try:
            answer = llm.chat_with_context(
                system_prompt=HALLUCINATION_SUPPRESSION_PROMPT,
                user_question=question,
                context=context_text,
            )
            gen_time = time.time() - t1
            success_count += 1
            error = None
            print(f"    回答完成 ({gen_time:.2f}s): {answer[:80]}...")
        except (ConnectionError, RuntimeError) as e:
            answer = ""
            gen_time = time.time() - t1
            fail_count += 1
            error = str(e)
            print(f"    回答失败: {error}")

        results.append({
            "question": question,
            "expected_answer": expected,
            "answer": answer,
            "retrieval_count": retrieval_count,
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(gen_time, 2),
            "error": error,
        })

    total = len(results)
    avg_retrieval = sum(r["retrieval_time"] for r in results) / total if total > 0 else 0
    gen_times = [r["generation_time"] for r in results if r["generation_time"] > 0]
    avg_generation = sum(gen_times) / len(gen_times) if gen_times else 0

    report = "# RAG 系统评估报告\n\n"
    report += "## 概要\n\n"
    report += "| 指标 | 数值 |\n|------|------|\n"
    report += f"| 评估问题数 | {total} |\n"
    report += f"| 成功回答数 | {success_count} |\n"
    report += f"| 失败回答数 | {fail_count} |\n"
    report += f"| 平均检索耗时 | {avg_retrieval:.2f}s |\n"
    report += f"| 平均生成耗时 | {avg_generation:.2f}s |\n\n"

    for i, r in enumerate(results):
        report += f"## 问题 {i+1}: {r['question']}\n\n"
        if r["expected_answer"]:
            report += f"- **期望回答**: {r['expected_answer'][:200]}\n"
        report += f"- **检索片段数**: {r['retrieval_count']}\n"
        report += f"- **检索耗时**: {r['retrieval_time']}s\n"
        report += f"- **生成耗时**: {r['generation_time']}s\n"
        if r["error"]:
            report += f"- **错误**: {r['error']}\n"
        else:
            report += f"- **模型回答**: {r['answer'][:300]}\n"
        report += "\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n评估完成！报告已保存到 {output_file}")
    print(f"成功: {success_count} / {total}，失败: {fail_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 系统自动化评估工具")
    parser.add_argument("-f", "--file", default="eval_questions.json", help="评估问题 JSON 文件路径")
    parser.add_argument("-o", "--output", default="eval_report.md", help="评估报告输出路径")
    args = parser.parse_args()
    run_eval(eval_file=args.file, output_file=args.output)