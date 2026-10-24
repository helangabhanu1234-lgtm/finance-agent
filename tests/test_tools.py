import csv
import unittest
from datetime import date, timedelta

import config
from tests.helpers import TempEnvTestCase
from tools import execute_tool
from tools import validators as v


class ValidatorTests(unittest.TestCase):
    def test_amount_parsing(self):
        self.assertEqual(v.parse_amount("Rs. 1,500"), 1500.0)
        self.assertEqual(v.parse_amount("2500.50"), 2500.5)
        self.assertEqual(v.parse_amount("$50"), 50.0)
        self.assertEqual(v.parse_amount(" 75 "), 75.0)
        for bad in (0, -5, "abc", "5 to 10", "Rs.", None, True, float("nan"), float("inf"), 10 ** 10, ""):
            with self.assertRaises(v.ToolError):
                v.parse_amount(bad)

    def test_category_synonyms_and_errors(self):
        self.assertEqual(v.normalize_category("lunch"), "Food")
        self.assertEqual(v.normalize_category(" GROCERIES "), "Groceries")
        with self.assertRaises(v.ToolError):
            v.normalize_category("spaceships")

    def test_dates(self):
        self.assertEqual(v.parse_date("yesterday"), date.today() - timedelta(days=1))
        self.assertEqual(v.parse_date(None), date.today())
        for bad in ("2026-13-45", "10/12/2026", "tomorrow", (date.today() + timedelta(days=3)).isoformat()):
            with self.assertRaises(v.ToolError):
                v.parse_date(bad)

    def test_months(self):
        start, end = v.parse_month("2026-02")
        self.assertEqual((start.isoformat(), end.isoformat()), ("2026-02-01", "2026-02-28"))
        last = v.parse_month("last month")[0]
        self.assertTrue(last < date.today().replace(day=1))
        with self.assertRaises(v.ToolError):
            v.parse_month("2026-99")


class ToolTests(TempEnvTestCase):
    def test_add_and_get_expense(self):
        r = execute_tool("add_expense", {"amount": 1500, "category": "lunch", "description": "rice", "date": "today"})
        self.assertTrue(r["ok"])
        self.assertEqual(r["expense"]["category"], "Food")
        got = execute_tool("get_expenses", {"month": "this month"})
        self.assertEqual(got["matching_count"], 1)
        self.assertEqual(got["total_lkr"], 1500.0)

    def test_add_expense_errors_do_not_crash(self):
        self.assertFalse(execute_tool("add_expense", {"amount": -5, "category": "Food"})["ok"])
        self.assertFalse(execute_tool("add_expense", {"amount": 5, "category": "nonsense"})["ok"])
        self.assertFalse(execute_tool("add_expense", {"category": "Food"})["ok"])          # missing amount
        self.assertFalse(execute_tool("add_expense", "not json")["ok"])
        self.assertFalse(execute_tool("no_such_tool", {})["ok"])
        self.assertFalse(execute_tool("add_expense", {"amount": 5, "category": "Food", "currency": "XXX"})["ok"])

    def test_unknown_arguments_are_ignored_and_reported(self):
        r = execute_tool("add_expense", {"amount": 10, "category": "Food", "colour": "red"})
        self.assertTrue(r["ok"])
        self.assertEqual(r["ignored_arguments"], ["colour"])

    def test_foreign_currency_expense_is_converted(self):
        r = execute_tool("add_expense", {"amount": 10, "category": "Shopping", "currency": "USD"})
        self.assertTrue(r["ok"])
        self.assertEqual(r["expense"]["amount_lkr"], 3000.0)       # offline fallback 300 LKR/USD
        self.assertEqual(r["expense"]["rate_source"], "offline_fallback")

    def test_summary_and_percentages(self):
        execute_tool("add_expense", {"amount": 3000, "category": "Food"})
        execute_tool("add_expense", {"amount": 1000, "category": "Transport"})
        s = execute_tool("summarize_spending", {"month": "this month"})
        self.assertEqual(s["total_lkr"], 4000.0)
        self.assertEqual(s["top_category"], "Food")
        self.assertEqual(s["by_category"][0]["percent"], 75.0)

    def test_budget_status_levels(self):
        execute_tool("set_budget", {"category": "Food", "monthly_limit": 1000})
        execute_tool("set_budget", {"category": "Transport", "monthly_limit": 1000})
        execute_tool("set_budget", {"category": "Bills", "monthly_limit": 1000})
        execute_tool("add_expense", {"amount": 1500, "category": "Food"})
        execute_tool("add_expense", {"amount": 850, "category": "Transport"})
        execute_tool("add_expense", {"amount": 100, "category": "Bills"})
        r = execute_tool("check_budget", {})
        status = {b["category"]: b["status"] for b in r["budgets"]}
        self.assertEqual(status, {"Bills": "OK", "Food": "OVER", "Transport": "WARNING"})
        self.assertEqual(r["over_budget_categories"], ["Food"])

    def test_check_budget_without_budgets(self):
        r = execute_tool("check_budget", {})
        self.assertTrue(r["ok"])
        self.assertEqual(r["budgets"], [])

    def test_convert_currency(self):
        r = execute_tool("convert_currency", {"amount": 50, "from_currency": "usd", "to_currency": "LKR"})
        self.assertEqual(r["converted"], 15000.0)
        self.assertFalse(execute_tool("convert_currency", {"amount": 5, "from_currency": "US"})["ok"])
        self.assertFalse(execute_tool("convert_currency", {"amount": 5, "from_currency": "ZZZ"})["ok"])

    def test_csv_import_with_bad_rows_and_duplicates(self):
        path = config.DATA_DIR / "t.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "amount", "category", "description"])
            w.writerow([date.today().isoformat(), "500", "Food", "tea"])
            w.writerow([date.today().isoformat(), "oops", "Food", "bad amount"])
            w.writerow(["2026-99-99", "100", "Food", "bad date"])
        first = execute_tool("import_expenses_csv", {"file_name": "t.csv"})
        self.assertEqual((first["imported"], first["rows_with_errors"]), (1, 2))
        again = execute_tool("import_expenses_csv", {"file_name": "t.csv"})
        self.assertEqual((again["imported"], again["skipped_duplicates"]), (0, 1))

    def test_csv_import_security_and_missing(self):
        self.assertFalse(execute_tool("import_expenses_csv", {"file_name": "nope.csv"})["ok"])
        self.assertFalse(execute_tool("import_expenses_csv", {"file_name": "../../etc/passwd"})["ok"])
        self.assertFalse(execute_tool("import_expenses_csv", {"file_name": "x.txt"})["ok"])
        (config.DATA_DIR / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        self.assertFalse(execute_tool("import_expenses_csv", {"file_name": "bad.csv"})["ok"])

    def test_report_file_created(self):
        execute_tool("add_expense", {"amount": 700, "category": "Food", "description": "<b>x</b>"})
        r = execute_tool("generate_report", {"month": "this month"})
        self.assertTrue(r["ok"])
        html = (config.REPORTS_DIR / f"report_{date.today():%Y-%m}.html").read_text(encoding="utf-8")
        self.assertIn("Personal Finance Report", html)
        self.assertNotIn("<b>x</b>", html)            # HTML is escaped

    def test_get_expenses_limit_validation(self):
        self.assertFalse(execute_tool("get_expenses", {"limit": "many"})["ok"])
        self.assertFalse(execute_tool("get_expenses", {"start_date": "2026-05-01", "end_date": "2026-04-01"})["ok"])


if __name__ == "__main__":
    unittest.main()
