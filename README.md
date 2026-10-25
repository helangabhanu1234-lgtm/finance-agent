# Personal Finance Helper - Local AI Agent (Ollama)

A tool-using AI agent that tracks expenses, checks budgets, converts currencies and builds spending reports.
The LLM (`llama3.2`, 3B) runs **locally with Ollama** - no paid cloud API. Written in plain Python + the `ollama` library.

LLM Engineering Assignment - HD in AI & SE 26-01 (Group project)

## What the agent can do

| Tool | Purpose | Persistent data |
|---|---|---|
| `add_expense` | Save one expense (auto-converts foreign currency to LKR) | SQLite write |
| `get_expenses` | List expenses by month / dates / category with totals | SQLite read |
| `summarize_spending` | Totals + percentages per category | SQLite read |
| `set_budget` | Create/update a monthly budget per category | SQLite write |
| `check_budget` | Compare spending with budgets (OK / WARNING / OVER) | SQLite read |
| `convert_currency` | Live exchange rate (open.er-api.com, no key) with offline fallback | external API |
| `import_expenses_csv` | Bulk import a CSV from the `data/` folder (skips duplicates and bad rows) | CSV read + SQLite write |
| `generate_report` | Writes an HTML report with chart, budgets and transactions | file write |

Every tool validates its inputs and returns `{"ok": false, "error": "..."}` instead of crashing.

## Architecture

```
User (CLI / Web UI)
      |
      v
FinanceAgent.chat()  --- system prompt + conversation history ---> Ollama (llama3.2)
      ^                                                                  |
      |   tool result (JSON)                          tool_calls <-------+
      +------ execute_tool() <----- tool registry (8 tools) ----> SQLite / CSV / HTML / Exchange-rate API
      |
      +--> logs/agent.log and logs/trace.jsonl (every tool call, arguments, result, time)
```

Agent loop (`agents/finance_agent.py`): send messages -> model asks for tool(s) -> run them -> append results -> repeat until the
model answers in plain text. Safety nets: max 8 steps, repeated-call guard, completion guardrail (reminds the model if it skips a required tool or claims an action it never performed), number guard (every amount in the answer must appear in a tool result), argument repair (exact converted amount), tool errors fed back to the model, friendly Ollama errors,
history trimming at user-message boundaries, **duplicate-write guard** (an expense is never saved twice in one request), **invalid-input short-circuit** (negative amounts are answered from the tool's own error), **category-pair guard** and a **verified-answer fallback** (if the model keeps writing wrong figures the answer is rebuilt from the tool results).

## Setup (Windows, e.g. `D:\finance_agent`)

1. Install **Python 3.10+** and **Ollama** (https://ollama.com). Start the Ollama app.
2. Pull the model once: `ollama pull llama3.2`
3. Open a terminal in the project folder and run:
   ```
   cd /d D:\finance_agent
   setup.bat
   ```
   This creates `.venv`, installs `requirements.txt`, loads ~100 sample expenses + budgets and runs `check_setup.py`.

Manual equivalent:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python seed_data.py --load --reset
python check_setup.py
```

## Run

| Command | What it does |
|---|---|
| `run_cli.bat` or `python main.py` | Interactive CLI with tool-call trace |
| `run_demo.bat` or `python main.py --demo --fresh` | Scripted demo prompts (`demo_prompts.txt`); `--fresh` resets the sample data first |
| `reset_demo.bat` | Reset the database to the sample data (same numbers every demo) |
| `run_web.bat` or `python web_ui.py` | Web UI at http://127.0.0.1:8000 (chat + live trace panel) |
| `run_tests.bat` or `python -m unittest discover -s tests -v` | 68 automated tests (no Ollama needed) |

CLI commands: `/help`, `/tools`, `/reset`, `/trace on|off`, `/exit`.

## Three example prompts

1. `I spent 2500 on lunch yesterday` -> single tool call (`add_expense`)
2. `How much did I spend this month, which category is the biggest, and am I over budget anywhere?` -> multi-step (`summarize_spending` + `check_budget`)
3. `Convert 50 USD to LKR and add it as a Shopping expense for today` -> multi-step (`convert_currency` + `add_expense`)

Follow-ups work because history is kept: after #2 try `and what about last month?`

## Troubleshooting

| Message | Meaning / fix |
|---|---|
| `Ollama is running but did not answer within N seconds` | The model is still loading or the PC is slow (llama3.2 on CPU can need 1-3 min per step). Send the message again; raise `OLLAMA_TIMEOUT` in `.env` if needed. |
| `Cannot reach Ollama at http://localhost:11434` | Ollama is not running. Start the Ollama app or run `ollama serve`. |
| Duplicate / wrong rows in the demo data | Run `reset_demo.bat` (or `python seed_data.py --load --reset`) before every demo or video. |

The real cause of any model error is always written to `logs/agent.log` (`OLLAMA CALL FAILED: <type>: <details>`).
See `CHANGES.md` for the bugs found in live testing and how each one was fixed.

## Configuration

Optional `.env` (copy `.env.example`): `OLLAMA_MODEL` (e.g. `qwen3:4b`), `OLLAMA_HOST`, `AGENT_MAX_STEPS`, `OLLAMA_TIMEOUT`, `OLLAMA_KEEP_ALIVE`, `FINANCE_OFFLINE=1`.

## Project structure

```
main.py  web_ui.py  seed_data.py  check_setup.py  config.py
agents/   finance_agent.py (loop) guardrails.py formatters.py prompts.py logging_setup.py
tools/    validators.py db.py expense_tools.py budget_tools.py currency_tools.py report_tools.py __init__.py (registry)
tests/    test_tools.py test_agent_loop.py test_guardrails.py test_reliability.py test_regressions.py helpers.py (fake Ollama client)
web/      index.html      data/ (CSV + SQLite)   reports/   logs/
```

## Safety / scope

No real payments, transfers or emails - all actions are local records. No API keys are used or committed (`.env` is git-ignored).

## Credits and AI usage

- Libraries: `ollama` (Python client), Python standard library (sqlite3, http.server, unittest). Exchange rates: https://open.er-api.com (free tier).
- Structure of the agent loop follows the course lecture notes (Week 1 - Agent Core and Tools).
- AI assistant usage (Claude) is documented in the Appendix of the technical report.

## Team

Group of 3 - HD in AI & SE 26-01 (Lecturer: Praneeth Silva)

| Member | Student ID | Main responsibility |
|---|---|---|
| Helanga Kotalawala | CCLARKE590 | Agent lead: agent loop, system prompt, conversation memory, guardrails, logging, CLI (`agents/`, `main.py`, `config.py`) |
| Dev Jayawardena | CCLARKE600 | Tools lead: 8 tools, input validation, SQLite layer, currency API, CSV import, reports, sample data (`tools/`, `seed_data.py`, `data/`) |
| Pradeep Weerasinghe | CCLARKE610 | Testing and docs lead: web UI, setup/run scripts, regression tests, README, report, demo video (`web_ui.py`, `web/`, `*.bat`, `tests/test_regressions.py`) |
