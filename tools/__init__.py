"""Tool registry: JSON schemas shown to the LLM + a safe executor that never crashes the agent."""
import inspect
import json
import logging

from tools.budget_tools import check_budget, set_budget
from tools.currency_tools import convert_currency
from tools.expense_tools import add_expense, get_expenses, import_expenses_csv, summarize_spending
from tools.report_tools import generate_report
from tools.validators import CATEGORIES, ToolError, UserInputError

log = logging.getLogger("finance_agent")

_PERIOD = {
    "month": {"type": "string", "description": "Month as YYYY-MM, or 'this month' / 'last month'."},
}


def _schema(func, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": func.__name__, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
    }


_CATEGORY = {"type": "string", "enum": CATEGORIES, "description": "Expense category."}

TOOLS = {
    "add_expense": (add_expense, _schema(
        add_expense, "Save ONE new expense to the database. Use whenever the user says they spent money.",
        {
            "amount": {"type": "number", "description": "Amount spent, positive number."},
            "category": _CATEGORY,
            "description": {"type": "string", "description": "Short note, e.g. 'lunch at Cafe'."},
            "date": {"type": "string", "description": "YYYY-MM-DD, 'today' or 'yesterday'. Default today."},
            "currency": {"type": "string", "description": "3-letter code, default LKR."},
        }, ["amount", "category"])),
    "get_expenses": (get_expenses, _schema(
        get_expenses, "List individual expenses (newest first) with their total. Filter by month, dates or category.",
        {**_PERIOD,
         "start_date": {"type": "string", "description": "YYYY-MM-DD"},
         "end_date": {"type": "string", "description": "YYYY-MM-DD"},
         "category": _CATEGORY,
         "limit": {"type": "integer", "description": "Max rows to show (default 20, max 100)."}})),
    "summarize_spending": (summarize_spending, _schema(
        summarize_spending, "Total spending per category with percentages for a period. Use for 'where does my money go' questions.",
        {**_PERIOD,
         "start_date": {"type": "string", "description": "YYYY-MM-DD"},
         "end_date": {"type": "string", "description": "YYYY-MM-DD"}})),
    "set_budget": (set_budget, _schema(
        set_budget, "Create or update the monthly budget limit (in LKR) for a category.",
        {"category": _CATEGORY, "monthly_limit": {"type": "number", "description": "Monthly limit in LKR."}},
        ["category", "monthly_limit"])),
    "check_budget": (check_budget, _schema(
        check_budget, "Compare spending with all budgets for a month; returns OK / WARNING / OVER per category.",
        _PERIOD)),
    "convert_currency": (convert_currency, _schema(
        convert_currency, "Convert an amount between currencies (live rate, offline fallback).",
        {"amount": {"type": "number", "description": "Amount to convert."},
         "from_currency": {"type": "string", "description": "3-letter code, e.g. USD."},
         "to_currency": {"type": "string", "description": "3-letter code, default LKR."}},
        ["amount", "from_currency"])),
    "import_expenses_csv": (import_expenses_csv, _schema(
        import_expenses_csv, "Import expenses from a CSV file stored in the data folder (date, amount, category, description).",
        {"file_name": {"type": "string", "description": "CSV file name, e.g. 'sample_expenses.csv'."}}, ["file_name"])),
    "generate_report": (generate_report, _schema(
        generate_report, "Create an HTML spending report (chart, budget status, transactions) for a month and save it to a file.",
        _PERIOD)),
}

TOOL_SCHEMAS = [schema for _, schema in TOOLS.values()]
TOOL_NAMES = list(TOOLS)


def execute_tool(name, arguments) -> dict:
    """Run a tool by name. ALWAYS returns a dict with 'ok'; never raises."""
    if name not in TOOLS:
        return {"ok": False, "error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_NAMES)}"}
    func = TOOLS[name][0]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"ok": False, "error": "Tool arguments must be a JSON object."}
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "Tool arguments must be a JSON object."}

    params = inspect.signature(func).parameters
    unknown = [k for k in arguments if k not in params]
    args = {k: v for k, v in arguments.items() if k in params}
    missing = [p for p, spec in params.items() if spec.default is inspect.Parameter.empty and p not in args]
    if missing:
        return {"ok": False, "error": f"Missing required argument(s): {', '.join(missing)}. Ask the user or retry with them."}
    try:
        result = func(**args)
    except UserInputError as exc:
        return {"ok": False, "error_type": "invalid_user_input", "error": str(exc)}
    except ToolError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:                      # unexpected bug: log it, keep the agent alive
        log.exception("Tool %s crashed", name)
        return {"ok": False, "error": f"Internal error in {name}: {type(exc).__name__}: {exc}"}
    if unknown:
        result["ignored_arguments"] = unknown
    return result
