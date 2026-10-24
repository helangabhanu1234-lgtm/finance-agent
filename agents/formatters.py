"""Deterministic answers built ONLY from tool results.

Used as a safety net when the small local model keeps writing wrong figures (see guardrails.py), when the
step limit is hit, or when the model stays silent. Nothing here is calculated by the LLM, so every number is exact.
"""
import re


def money(value) -> str:
    return f"Rs. {float(value):,.2f}"


def _file_name(path: str) -> str:
    return re.split(r"[\\/]", str(path))[-1]


def _convert(r):
    source = "approximate offline rate" if r.get("rate_source") == "offline_fallback" else f"{r.get('rate_source', 'live')} rate"
    return f"{r['amount']:,.2f} {r['from']} = {r['converted']:,.2f} {r['to']} ({source} {r['rate']:,.4f})."


def _add(r):
    e = r["expense"]
    line = f"Saved: {money(e['amount_lkr'])} under {e['category']} on {e['date']}"
    if e.get("currency") and e["currency"] != "LKR":
        line += f" (originally {e['amount']:,.2f} {e['currency']})"
    if e.get("description"):
        line += f" - {e['description']}"
    return line + f" (expense id {e['id']})."


def _summary(r):
    if not r.get("by_category"):
        return f"No spending recorded for {r['period']}."
    lines = [f"Spending for {r['period']}: {r['total_display']} in {r['transactions']} transactions. "
             f"Biggest category: {r['top_category']}."]
    lines += [f"- {c['category']}: {money(c['total_lkr'])} ({c['percent']}%)" for c in r["by_category"]]
    return "\n".join(lines)


def _expenses(r):
    scope = f" in {r['category']}" if r.get("category") else ""
    lines = [f"{r['matching_count']} expenses{scope} for {r['period']}, total {money(r['total_lkr'])}."]
    lines += [f"- {e['date']}  {e['category']}  {money(e['amount_lkr'])}  {e.get('description', '')}".rstrip()
              for e in r.get("expenses", [])[:10]]
    if r["matching_count"] > r["showing"]:
        lines.append(f"(showing the latest {r['showing']})")
    return "\n".join(lines)


def _budget(r):
    if not r.get("budgets"):
        return r.get("message", "No budgets set yet.")
    lines = [f"Budget status for {r['month']}:"]
    flagged = [b for b in r["budgets"] if b["status"] != "OK"]
    for b in flagged:
        if b["status"] == "OVER":
            lines.append(f"- {b['category']}: OVER by {money(b['over_by_lkr'])} "
                         f"(spent {money(b['spent_lkr'])} of {money(b['limit_lkr'])}, {b['percent_used']}%)")
        else:
            lines.append(f"- {b['category']}: close to the limit - spent {money(b['spent_lkr'])} "
                         f"of {money(b['limit_lkr'])} ({b['percent_used']}%)")
    if not flagged:
        lines.append("- All budgets are within their limits.")
    return "\n".join(lines)


def _report(r):
    return (f"Report for {r['month']} saved as {_file_name(r['file'])} in the reports folder. "
            f"Total spent {money(r['total_lkr'])}, top category {r['top_category']}.")


def _set_budget(r):
    return r["message"]


def _import(r):
    return (f"Imported {r['imported']} rows from {r['file']} "
            f"({r['skipped_duplicates']} duplicates skipped, {r['rows_with_errors']} rows with errors).")


_RENDERERS = {
    "convert_currency": _convert, "summarize_spending": _summary, "get_expenses": _expenses,
    "check_budget": _budget, "generate_report": _report, "set_budget": _set_budget,
    "import_expenses_csv": _import,
}


def render_answer(events: list):
    """Answer text built from the successful tool results of this request, or None if nothing succeeded."""
    ok = [e for e in events if e.get("ok")]
    if not ok:
        return None
    parts, last = [], {}
    for event in ok:
        if event["tool"] == "add_expense":
            parts.append(("add_expense", _add(event["result"])))
        else:
            last[event["tool"]] = event            # later results replace earlier ones of the same tool
    order = []
    for event in ok:
        if event["tool"] != "add_expense" and event["tool"] not in order:
            order.append(event["tool"])
    for tool in order:
        render = _RENDERERS.get(tool)
        if render:
            parts.append((tool, render(last[tool]["result"])))
    # keep the order in which the tools were first used
    first_seen = {}
    for idx, event in enumerate(ok):
        first_seen.setdefault(event["tool"], idx)
    parts.sort(key=lambda item: first_seen.get(item[0], 0))
    return "\n".join(text for _, text in parts) or None
