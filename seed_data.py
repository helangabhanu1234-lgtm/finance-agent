"""Create realistic sample data (relative to today) and optionally load it into the database.

python seed_data.py                 -> writes data/sample_expenses.csv
python seed_data.py --load          -> also imports it and sets sample budgets
python seed_data.py --load --reset  -> deletes the existing database first
"""
import argparse
import csv
import random
from datetime import date, timedelta

import config

random.seed(2026)

# (category, description, min, max, times per month)
PATTERNS = [
    ("Rent", "Monthly house rent", 45000, 45000, 1),
    ("Bills", "CEB electricity bill", 4200, 7800, 1),
    ("Bills", "Dialog mobile + data", 1500, 2500, 1),
    ("Bills", "SLT fibre internet", 4990, 4990, 1),
    ("Groceries", "Keells weekly groceries", 6500, 14500, 4),
    ("Groceries", "Cargills fruits & veg", 1800, 4200, 3),
    ("Food", "Lunch at office canteen", 450, 900, 8),
    ("Food", "Dinner out", 2500, 6500, 3),
    ("Food", "Uber Eats order", 1800, 4200, 2),
    ("Transport", "PickMe ride", 600, 2200, 6),
    ("Transport", "Fuel", 4000, 9000, 2),
    ("Entertainment", "Netflix subscription", 1750, 1750, 1),
    ("Entertainment", "Cinema tickets", 1800, 3600, 1),
    ("Shopping", "Clothes / accessories", 3500, 18000, 1),
    ("Health", "Pharmacy", 900, 5200, 1),
    ("Education", "Online course / books", 2500, 12000, 1),
    ("Other", "Gift / miscellaneous", 1000, 6000, 1),
]


def generate(today: date):
    rows = []
    first_this = today.replace(day=1)
    months = [(first_this - timedelta(days=1)).replace(day=1)]
    months.insert(0, (months[0] - timedelta(days=1)).replace(day=1))
    months.append(first_this)
    for start in months:
        next_month = (start + timedelta(days=32)).replace(day=1)
        last_day = min(next_month - timedelta(days=1), today)
        span = (last_day - start).days + 1
        share = span / ((next_month - start).days)
        for category, desc, low, high, times in PATTERNS:
            count = max(1, round(times * share)) if times > 1 else (1 if span >= 5 else 0)
            for _ in range(count):
                day = start + timedelta(days=random.randint(0, span - 1))
                amount = round(random.randint(low, high) / 10) * 10 if low != high else low
                rows.append((day.isoformat(), amount, category, desc))
    rows.sort()
    return rows


def load_sample_data(reset: bool = True) -> dict:
    """Write the CSV, (optionally) delete the old database, import the CSV and set the sample budgets."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / "sample_expenses.csv"
    rows = generate(date.today())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "amount", "category", "description"])
        writer.writerows(rows)
    if reset and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    from tools import execute_tool
    result = execute_tool("import_expenses_csv", {"file_name": "sample_expenses.csv"})
    for category, limit in [("Food", 25000), ("Groceries", 30000), ("Transport", 18000), ("Entertainment", 6000), ("Shopping", 15000)]:
        execute_tool("set_budget", {"category": category, "monthly_limit": limit})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="import into the database and set budgets")
    parser.add_argument("--reset", action="store_true", help="delete the existing database first")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / "sample_expenses.csv"
    rows = generate(date.today())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "amount", "category", "description"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sample expenses to {path}")

    if args.load:
        if args.reset and config.DB_PATH.exists():
            config.DB_PATH.unlink()
            print("Old database deleted.")
        from tools import execute_tool
        print(execute_tool("import_expenses_csv", {"file_name": "sample_expenses.csv"}))
        for category, limit in [("Food", 25000), ("Groceries", 30000), ("Transport", 18000), ("Entertainment", 6000), ("Shopping", 15000)]:
            execute_tool("set_budget", {"category": category, "monthly_limit": limit})
        print("Sample budgets set (Food, Groceries, Transport, Entertainment, Shopping).")


if __name__ == "__main__":
    main()
