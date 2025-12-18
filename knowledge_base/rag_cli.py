import asyncio
import sys

from knowledge_base.rag_agent import answer_question


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python -m knowledge_base.rag_cli "your question"')
        raise SystemExit(1)

    question = sys.argv[1]
    answer = asyncio.run(answer_question(question, k=5))

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
