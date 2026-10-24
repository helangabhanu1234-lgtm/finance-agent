"""Command-line interface for the Personal Finance Helper agent.

Run:  python main.py            (interactive chat)
      python main.py --demo     (runs the demo prompts from demo_prompts.txt, pausing between them)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import config
from agents.finance_agent import FinanceAgent, check_ollama
from tools import TOOL_NAMES

RED, GREEN, YELLOW, CYAN, GREY, BOLD, RESET = "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[90m", "\033[1m", "\033[0m"

HELP = f"""Commands:
  /help           show this help
  /tools          list the agent's tools
  /reset          forget the conversation (saved expenses stay)
  /trace on|off   show / hide the tool-call trace
  /exit           quit
Examples:
  I spent 2500 on lunch yesterday
  How much did I spend this month and where did it go?
  Convert 50 USD to LKR and add it as a Shopping expense
"""


def short(value, limit=300):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def make_printer(show_trace: bool):
    def printer(event):
        if not show_trace:
            return
        if event["type"] == "tool_call":
            print(f"{YELLOW}  [step {event['step']}] TOOL CALL  {event['name']}({short(event['arguments'], 200)}){RESET}")
        elif event["type"] == "guardrail":
            print(f"{RED}  [step {event['step']}] GUARDRAIL  request not finished, missing: {', '.join(event['missing'])} -> reminding the model{RESET}")
        elif event["type"] == "number_guard":
            print(f"{RED}  [step {event['step']}] NUMBER GUARD  figures not found in tool results: {', '.join(event['numbers'])} -> asking the model to correct them{RESET}")
        elif event["type"] == "arg_fix":
            print(f"{YELLOW}  [step {event['step']}] ARG FIX     {event['name']}: {event['note']}{RESET}")
        elif event["type"] == "tool_result":
            colour = GREEN if event["ok"] else RED
            print(f"{colour}  [step {event['step']}] RESULT     ok={event['ok']} ({event['ms']} ms) {short(event['result'])}{RESET}")
    return printer


def ask(agent, text, show_trace=True):
    print(f"{GREY}  thinking...{RESET}", end="\r")
    result = agent.chat(text, on_event=make_printer(show_trace))
    print(" " * 20, end="\r")
    colour = RED if result.error else CYAN
    print(f"{colour}{BOLD}FinBot:{RESET} {result.answer}")
    if show_trace:
        print(f"{GREY}  (model calls: {result.steps}/{agent.max_steps}, tool calls: {len(result.events)}){RESET}\n")


def run_demo(agent):
    prompts_file = config.BASE_DIR / "demo_prompts.txt"
    prompts = [p.strip() for p in prompts_file.read_text(encoding="utf-8").splitlines() if p.strip() and not p.startswith("#")]
    for number, prompt in enumerate(prompts, 1):
        input(f"{GREY}--- Demo prompt {number}/{len(prompts)}: press Enter to send ---{RESET}")
        print(f"{GREEN}{BOLD}You:{RESET} {prompt}")
        ask(agent, prompt)


def main():
    parser = argparse.ArgumentParser(description="Personal Finance Helper (Ollama agent)")
    parser.add_argument("--demo", action="store_true", help="run the scripted demo prompts")
    parser.add_argument("--fresh", action="store_true", help="reset the database to the sample data first (clean demo)")
    parser.add_argument("--no-trace", action="store_true", help="hide the tool-call trace")
    parser.add_argument("--model", default=None, help=f"Ollama model (default: {config.MODEL})")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.system("")                    # enables ANSI colours in the Windows console
    if args.model:
        config.MODEL = args.model

    if args.fresh:
        import seed_data
        seed_data.load_sample_data(reset=True)
        print("Sample data reset (clean demo database).")
    print(f"{RED}{BOLD}=== Personal Finance Helper (local AI agent) ==={RESET}")
    ok, message = check_ollama(config.MODEL)
    print((GREEN if ok else RED) + message + RESET)
    if not ok:
        sys.exit(1)
    print(f"Model: {config.MODEL} | Tools: {len(TOOL_NAMES)} | Max steps/request: {config.MAX_STEPS} | Type /help\n")

    agent = FinanceAgent(model=config.MODEL)
    show_trace = not args.no_trace
    if args.demo:
        run_demo(agent)
        return

    while True:
        try:
            text = input(f"{GREEN}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not text:
            continue
        low = text.lower()
        if low in {"/exit", "/quit", "exit", "quit"}:
            print("Bye!")
            break
        if low == "/help":
            print(HELP)
        elif low == "/tools":
            print("Tools: " + ", ".join(TOOL_NAMES) + "\n")
        elif low == "/reset":
            agent.reset()
            print("Conversation cleared.\n")
        elif low.startswith("/trace"):
            show_trace = not low.endswith("off")
            print(f"Trace {'ON' if show_trace else 'OFF'}\n")
        elif low.startswith("/"):
            print("Unknown command. Type /help\n")
        else:
            ask(agent, text, show_trace)


if __name__ == "__main__":
    main()
