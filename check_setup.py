"""Pre-flight check. Run this before the demo / viva:  python check_setup.py"""
import sys

import config


def line(ok, text):
    print(("[PASS] " if ok else "[FAIL] ") + text)
    return ok


def main():
    all_ok = True
    all_ok &= line(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (need 3.10+)")
    try:
        import ollama  # noqa: F401
        all_ok &= line(True, "Python package 'ollama' installed")
    except ImportError:
        line(False, "Package 'ollama' missing -> run: pip install -r requirements.txt")
        sys.exit(1)

    from agents.finance_agent import FinanceAgent, check_ollama
    ok, msg = check_ollama()
    all_ok &= line(ok, msg)
    if not ok:
        sys.exit(1)

    from tools import execute_tool
    all_ok &= line(execute_tool("summarize_spending", {"month": "this month"}).get("ok", False),
                   f"Database works ({config.DB_PATH})")
    all_ok &= line(execute_tool("convert_currency", {"amount": 1, "from_currency": "USD"}).get("ok", False),
                   "Currency tool works (live or offline rate)")

    print("\nLive model test (can take 10-60 s on the first call while the model loads)...")
    agent = FinanceAgent()
    result = agent.chat("Convert 10 USD to LKR")
    used_tool = any(e["tool"] == "convert_currency" for e in result.events)
    all_ok &= line(used_tool, f"Model called a tool correctly. Answer: {result.answer[:120]}")
    print("\nALL CHECKS PASSED - ready to run." if all_ok else "\nSome checks failed - see messages above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
