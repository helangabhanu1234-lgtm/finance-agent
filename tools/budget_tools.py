"""Tools: set_budget, check_budget."""
from tools import db
from tools.validators import ToolError, normalize_category, parse_amount, parse_month

WARNING_THRESHOLD = 0.8


def set_budget(category, monthly_limit) -> dict:
    cat = normalize_category(category)
    limit = parse_amount(monthly_limit, "monthly_limit")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO budgets(category, monthly_limit) VALUES (?,?) "
            "ON CONFLICT(category) DO UPDATE SET monthly_limit=excluded.monthly_limit",
            (cat, limit),
        )
    return {"ok": True, "message": f"Monthly budget for {cat} set to {limit:,.2f} LKR.",
            "category": cat, "monthly_limit_lkr": limit}


def check_budget(month=None) -> dict:
    start, end = parse_month(month)
    with db.connect() as conn:
        budgets = conn.execute("SELECT category, monthly_limit FROM budgets ORDER BY category").fetchall()
        if not budgets:
            return {"ok": True, "month": f"{start:%B %Y}", "budgets": [],
                    "message": "No budgets set yet. Use set_budget first."}
        spent = {
            r["category"]: r["t"] for r in conn.execute(
                "SELECT category, SUM(amount_lkr) t FROM expenses WHERE date>=? AND date<=? GROUP BY category",
                (start.isoformat(), end.isoformat()),
            )
        }
    items, over, warn = [], [], []
    for b in budgets:
        used = round(spent.get(b["category"], 0.0), 2)
        limit = b["monthly_limit"]
        ratio = used / limit if limit else 0
        status = "OVER" if ratio > 1 else "WARNING" if ratio >= WARNING_THRESHOLD else "OK"
        if status == "OVER":
            over.append(b["category"])
        elif status == "WARNING":
            warn.append(b["category"])
        item = {
            "category": b["category"], "limit_lkr": limit, "spent_lkr": used,
            "remaining_lkr": round(limit - used, 2), "percent_used": round(100 * ratio, 1), "status": status,
        }
        if status == "OVER":
            item["over_by_lkr"] = round(used - limit, 2)      # computed here so the model never does the maths
        items.append(item)
    return {"ok": True, "month": f"{start:%B %Y}", "budgets": items,
            "over_budget_categories": over, "warning_categories": warn}
