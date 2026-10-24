"""Completion guardrail for small models.

Live testing with llama3.2 (3B) showed two weaknesses: it sometimes stops after the FIRST tool of a
multi-step request and then claims the rest was done, and it sometimes answers without using the right tool.
This module looks at the user's request and at the tools that were really called, and lists what is still missing.
The agent loop then reminds the model (max 2 times) instead of accepting a made-up answer.
"""
import re

# (tool name, regex on the lower-cased user message)
_RULES = [
    ("convert_currency", r"\bconvert\b"),
    ("add_expense", r"\b(i|we)\s+(just\s+|have\s+)?(spent|paid|bought)\b|\b(add|save|record|log)\b(?!\s+up)"),
    ("set_budget", r"\b(set|change|update|create)\b.*\bbudget\b"),
    ("check_budget", r"over budget|under budget|within budget|budget status|check (my |the )?budgets?"),
    ("summarize_spending", r"biggest|largest|highest|top category|which category|where (did|does) my money|breakdown|by category|summari[sz]e"),
    ("generate_report", r"\b(generate|create|make|build|produce)\b.*\breport\b"),
    ("import_expenses_csv", r"\bimport\b.*\.csv"),
]
_ORDER = [name for name, _ in _RULES]
_SAVED_WORDS = re.compile(r"\b(saved|added|recorded|logged)\b", re.I)
_FAIL_WORDS = re.compile(r"\b(error|invalid|could not|couldn't|cannot|can't|unable|failed|not valid|must be)\b", re.I)


def required_tools(user_text: str) -> list:
    text = (user_text or "").lower()
    return [name for name, pattern in _RULES if re.search(pattern, text)]


def missing_tools(user_text: str, events: list, answer: str = "") -> list:
    """Tools the request needs that were never called (or whose only attempt failed while the answer claims success)."""
    text = (user_text or "").lower()
    called = {}
    for e in events:
        called.setdefault(e["tool"], []).append(e)
    missing = []
    for tool in required_tools(user_text):
        if tool == "generate_report" and re.search(r"\bif\b", text):
            # conditional request ("if yes, generate a report"): only needed when something is over budget
            over = [c for e in called.get("check_budget", []) if e["ok"] for c in e["result"].get("over_budget_categories", [])]
            if not over:
                continue
        if tool not in called:
            missing.append(tool)
        elif tool == "add_expense" and not any(e["ok"] for e in called[tool]):
            if _SAVED_WORDS.search(answer or "") and not _FAIL_WORDS.search(answer or ""):
                missing.append(tool)           # all attempts failed but the answer claims it was saved
    return missing


def build_reminder(missing: list, user_text: str, events: list) -> str:
    hints = []
    if "add_expense" in missing:
        converted = [e for e in events if e["tool"] == "convert_currency" and e["ok"]]
        if converted:
            hints.append(f"For add_expense use amount={converted[-1]['result'].get('converted')} and currency='LKR' "
                         f"(the converted value) and the category the user asked for.")
        else:
            hints.append("For add_expense pass amount, category, description and date from the user's message.")
    if "generate_report" in missing:
        hints.append("For generate_report pass the month the user asked for (default 'this month').")
    if "summarize_spending" in missing:
        hints.append("For 'biggest category' questions call summarize_spending(month='this month') and read top_category.")
    names = ", ".join(missing)
    return (f"[system check] The user's request is NOT finished. You have not successfully called: {names}. "
            f"Those parts were NOT done, so do not say they were done. Call the missing tool(s) now, one at a time, "
            f"then give the final answer. " + " ".join(hints))


# ---------------------------------------------------------------------------------------------
# Number guard. Live testing showed llama3.2 copying figures wrongly (Rs. 206,960 became Rs. 20,696),
# doing wrong sums ("over budget by 18,420" instead of 17,420) and re-using numbers from another month.
# Every money amount / percentage in the final answer must therefore appear in a tool result
# (or in the user's own message). Anything else is reported back to the model as "ungrounded".
# ---------------------------------------------------------------------------------------------
_MONEY = re.compile(r"(?:Rs\.?|LKR|USD|EUR|GBP|INR|AUD|\$)\s*(\d[\d,]*(?:\.\d+)?)", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_TOLERANCE = 0.51          # a figure written WITHOUT decimals may be rounded (16600.51 -> 16,601)
_TOLERANCE_EXACT = 0.011   # a figure written WITH decimals must match exactly (16,601.00 is NOT 16,600.51)


_COUNT_KEYS = {"transactions", "matching_count", "showing", "id", "imported", "skipped_duplicates", "rows_with_errors"}


def _collect_numbers(obj, out: set, only_percent: bool = False, in_percent: bool = False) -> None:
    """Collect numbers from a tool result. Money and percentages are kept apart so that a percentage
    of 100.0 cannot make a wrong 'Rs. 100.00' look verified; counters and ids are not money."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if in_percent == only_percent:
            out.add(float(obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            is_percent = "percent" in str(key)
            if not only_percent and (is_percent or key in _COUNT_KEYS):
                continue
            _collect_numbers(value, out, only_percent, is_percent)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, out, only_percent, in_percent)
    elif isinstance(obj, str):
        if not only_percent:
            for raw in _ANY_NUMBER.findall(obj):
                try:
                    out.add(float(raw.replace(",", "").rstrip(".")))
                except ValueError:
                    pass


def known_numbers(user_text: str, events: list, percent: bool = False) -> set:
    known = set()
    for raw in _ANY_NUMBER.findall(user_text or ""):
        try:
            known.add(float(raw.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    for event in events:
        if not percent:
            _collect_numbers(event.get("arguments"), known)
        _collect_numbers(event.get("result"), known, only_percent=percent)
    return known


def ungrounded_numbers(answer: str, events: list, user_text: str = "") -> list:
    """Money amounts / percentages in the answer that no tool result (or user message) contains."""
    bad = []
    for pattern, suffix in ((_MONEY, ""), (_PERCENT, "%")):
        known = known_numbers(user_text, events, percent=bool(suffix))
        for match in pattern.finditer(answer or ""):
            raw = match.group(1).rstrip(",")
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            tolerance = _TOLERANCE if "." not in raw else _TOLERANCE_EXACT
            if not any(abs(value - k) <= tolerance for k in known):
                label = (match.group(0).strip() if not suffix else raw + suffix)
                if label not in bad:
                    bad.append(label)
    return bad


def build_number_reminder(bad: list) -> str:
    return (f"[system check] Your answer contains figures that do NOT appear in any tool result: {', '.join(bad)}. "
            f"Do not calculate, round or guess numbers, and do not reuse numbers from another month. "
            f"Write the answer again and copy every figure exactly from the tool results above "
            f"(for example total_display, total_lkr, spent_lkr, over_by_lkr, percent_used). "
            f"Only call a category 'over budget' if it is listed in over_budget_categories. "
            f"Do NOT call any tool again - everything you need is already in the tool results above.")


# ---------------------------------------------------------------------------------------------
# Category-pair guard. Live testing: the model wrote "Food: Rs. 2,500.00" (the lunch it had just added)
# in a category breakdown where the tool said Food = 22,510. Both numbers exist in the conversation, so the
# plain number guard accepts them. Here every line that names ONE category must use that category's own figures.
# ---------------------------------------------------------------------------------------------
def _category_numbers(events: list) -> dict:
    """category -> money figures the READ tools reported for it (plus grand totals under the key None)."""
    cats = {None: set()}

    def add(cat, item):
        bucket = cats.setdefault(cat, set())
        for key, value in item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and "percent" not in key and key not in _COUNT_KEYS:
                bucket.add(abs(float(value)))

    for event in events:
        if not event.get("ok"):
            continue
        result = event.get("result") or {}
        for key in ("by_category", "budgets", "budget_details"):
            for item in result.get(key) or []:
                if isinstance(item, dict) and item.get("category"):
                    add(item["category"], item)
        if event.get("tool") == "get_expenses" and result.get("category"):
            add(result["category"], {"total_lkr": result.get("total_lkr", 0)})
            for row in result.get("expenses") or []:
                add(result["category"], {"amount_lkr": row.get("amount_lkr", 0)})
        if isinstance(result.get("total_lkr"), (int, float)) and event.get("tool") != "get_expenses":
            cats[None].add(float(result["total_lkr"]))
    return cats


def misattributed_numbers(answer: str, events: list) -> list:
    """[(category, 'Rs. 2,500.00', correct_figures)] for lines whose figure belongs to a different category."""
    cats = _category_numbers(events)
    names = [c for c in cats if c]
    if not names:
        return []
    bad = []
    for line in (answer or "").splitlines():
        if _SAVED_WORDS.search(line):                      # "Saved Rs. 2,500 under Food" is a confirmation, not a breakdown
            continue
        found = [c for c in names if re.search(rf"\b{re.escape(c)}\b", line, re.I)]
        if len(found) != 1:
            continue
        allowed = cats[found[0]] | cats[None]
        for match in _MONEY.finditer(line):
            raw = match.group(1).rstrip(",")
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            tolerance = _TOLERANCE if "." not in raw else _TOLERANCE_EXACT
            if not any(abs(value - k) <= tolerance for k in allowed):
                correct = sorted(cats[found[0]])
                bad.append((found[0], match.group(0).strip(), correct))
    return bad


def build_pair_reminder(pairs: list) -> str:
    parts = []
    for category, figure, correct in pairs:
        shown = ", ".join(f"Rs. {v:,.2f}" for v in correct[:4])
        parts.append(f"'{category}' is NOT {figure} (the tool reports for {category}: {shown})")
    return ("[system check] Some figures are attached to the wrong category: " + "; ".join(parts) + ". "
            "Write the answer again. Copy each category's own figure from the tool result "
            "(by_category[].total_lkr or budgets[].spent_lkr). Do NOT call any tool again.")
