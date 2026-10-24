"""Tools that write/read expenses: add_expense, get_expenses, summarize_spending, import_expenses_csv."""
import csv
from pathlib import Path

import config
from tools import db
from tools.currency_tools import to_base_currency
from tools.validators import (
    ToolError, clean_text, normalize_category, normalize_currency, parse_amount, parse_date, resolve_period,
)


def _insert(conn, amount, category, description, date_value, currency):
    """Validate already-parsed values and insert one row. Returns the stored row as a dict."""
    amount_lkr, rate, source = to_base_currency(amount, currency)
    cur = conn.execute(
        "INSERT INTO expenses(date, amount, currency, amount_lkr, category, description) VALUES (?,?,?,?,?,?)",
        (date_value.isoformat(), amount, currency, amount_lkr, category, description),
    )
    row = {
        "id": cur.lastrowid, "date": date_value.isoformat(), "amount": amount, "currency": currency,
        "amount_lkr": amount_lkr, "category": category, "description": description,
    }
    if currency != config.BASE_CURRENCY:
        row["exchange_rate"] = round(rate, 6)
        row["rate_source"] = source
    return row


def add_expense(amount, category, description="", date=None, currency="LKR") -> dict:
    value = parse_amount(amount)
    cat = normalize_category(category)
    desc = clean_text(description, "description")
    when = parse_date(date)
    cur = normalize_currency(currency or config.BASE_CURRENCY)
    with db.connect() as conn:
        row = _insert(conn, value, cat, desc, when, cur)
    return {"ok": True, "message": "Expense saved.", "expense": row}


def get_expenses(month=None, start_date=None, end_date=None, category=None, limit=20) -> dict:
    start, end, label = resolve_period(month, start_date, end_date)
    cat = normalize_category(category) if category not in (None, "") else None
    try:
        max_rows = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        raise ToolError("limit must be a whole number between 1 and 100")
    where, params = db.build_filters(start, end, cat)
    with db.connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) n, COALESCE(SUM(amount_lkr),0) t FROM expenses{where}", params).fetchone()
        rows = conn.execute(
            f"SELECT id, date, amount_lkr, category, description FROM expenses{where} "
            f"ORDER BY date DESC, id DESC LIMIT ?", params + [max_rows],
        ).fetchall()
    return {
        "ok": True, "period": label, "category": cat, "matching_count": total["n"],
        "total_lkr": round(total["t"], 2), "showing": len(rows),
        "expenses": [dict(r) for r in rows],
    }


def summarize_spending(month=None, start_date=None, end_date=None) -> dict:
    start, end, label = resolve_period(month, start_date, end_date)
    where, params = db.build_filters(start, end)
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT category, ROUND(SUM(amount_lkr),2) total, COUNT(*) n FROM expenses{where} "
            f"GROUP BY category ORDER BY total DESC", params,
        ).fetchall()
    grand = round(sum(r["total"] for r in rows), 2)
    breakdown = [
        {"category": r["category"], "total_lkr": r["total"], "transactions": r["n"],
         "percent": round(100 * r["total"] / grand, 1) if grand else 0.0}
        for r in rows
    ]
    return {
        "ok": True, "period": label, "total_lkr": grand, "total_display": f"Rs. {grand:,.2f}",
        "transactions": sum(r["n"] for r in rows),
        "top_category": breakdown[0]["category"] if breakdown else None,
        "by_category": breakdown,
    }


def import_expenses_csv(file_name) -> dict:
    """Import a CSV (columns: date, amount, category, description[, currency]) from the data/ folder."""
    if not isinstance(file_name, str) or not file_name.strip():
        raise ToolError("file_name is required, e.g. 'sample_expenses.csv'")
    name = Path(file_name.strip().replace("\\", "/")).name          # drop any folder part (no path traversal)
    if not name.lower().endswith(".csv"):
        raise ToolError("Only .csv files can be imported")
    path = config.DATA_DIR / name
    if not path.is_file():
        available = sorted(p.name for p in config.DATA_DIR.glob("*.csv"))
        raise ToolError(f"File '{name}' not found in the data folder. Available CSV files: {available or 'none'}")

    imported, duplicates, errors = 0, 0, []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ToolError("CSV file is empty")
        fields = {f.strip().lower() for f in reader.fieldnames if f}
        missing = {"date", "amount", "category"} - fields
        if missing:
            raise ToolError(f"CSV is missing required columns: {sorted(missing)}. Found: {sorted(fields)}")
        with db.connect() as conn:
            for line_no, raw in enumerate(reader, start=2):
                if line_no > 5001:
                    errors.append("Row limit (5000) reached; remaining rows ignored")
                    break
                row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
                try:
                    value = parse_amount(row.get("amount"))
                    cat = normalize_category(row.get("category"))
                    when = parse_date(row.get("date"))
                    desc = clean_text(row.get("description"), "description")
                    cur = normalize_currency(row.get("currency") or config.BASE_CURRENCY)
                    dup = conn.execute(
                        "SELECT 1 FROM expenses WHERE date=? AND amount=? AND category=? AND description=? AND currency=?",
                        (when.isoformat(), value, cat, desc, cur),
                    ).fetchone()
                    if dup:
                        duplicates += 1
                        continue
                    _insert(conn, value, cat, desc, when, cur)
                    imported += 1
                except ToolError as exc:
                    if len(errors) < 5:
                        errors.append(f"line {line_no}: {exc}")
    return {
        "ok": True, "file": name, "imported": imported, "skipped_duplicates": duplicates,
        "rows_with_errors": len(errors), "error_samples": errors,
    }
