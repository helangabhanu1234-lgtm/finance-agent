"""Tests for the fixes made after live testing with llama3.2 (wrong numbers, truncated amounts, 'None' text)."""
import unittest

from agents.finance_agent import FinanceAgent
from agents.guardrails import ungrounded_numbers
from tests.helpers import FakeClient, TempEnvTestCase, text_reply, tool_reply
from tools import execute_tool
from tools.validators import clean_text


class NumberGuardUnitTests(unittest.TestCase):
    EVENTS = [{
        "tool": "summarize_spending", "ok": True, "arguments": {"month": "this month"},
        "result": {"ok": True, "total_lkr": 206960.0, "total_display": "Rs. 206,960.00",
                   "by_category": [{"category": "Shopping", "total_lkr": 49020.0, "percent": 23.7}]},
    }]

    def test_detects_digit_error_from_real_run(self):
        # real llama3.2 output: Rs. 206,960.00 was written as Rs. 20,696.00
        self.assertEqual(ungrounded_numbers("This month's spending: Rs. 20,696.00", self.EVENTS, "how much"), ["Rs. 20,696.00"])

    def test_accepts_correct_numbers_and_percentages(self):
        answer = "Total Rs. 206,960.00. Shopping: Rs. 49,020.00 (23.7%)."
        self.assertEqual(ungrounded_numbers(answer, self.EVENTS, "how much"), [])

    def test_detects_wrong_percentage_and_user_numbers_are_allowed(self):
        self.assertEqual(ungrounded_numbers("Shopping is 73.7% of spending", self.EVENTS, ""), ["73.7%"])
        self.assertEqual(ungrounded_numbers("You spent Rs. 2,500.00", [], "I spent 2500 on lunch"), [])

    def test_rounding_is_tolerated(self):
        events = [{"tool": "convert_currency", "ok": True, "arguments": {}, "result": {"converted": 16600.51}}]
        self.assertEqual(ungrounded_numbers("That is Rs. 16,601", events, ""), [])


class NumberGuardLoopTests(TempEnvTestCase):
    def test_wrong_total_is_corrected_by_reminder(self):
        execute_tool("add_expense", {"amount": 1000, "category": "Food", "description": "x", "date": "today"})
        client = FakeClient([
            tool_reply("summarize_spending", month="this month"),
            text_reply("You spent Rs. 100.00 this month."),                 # wrong figure
            text_reply("You spent Rs. 1,000.00 this month."),               # corrected after the reminder
        ])
        seen = []
        result = FinanceAgent(client=client).chat("How much did I spend this month?", on_event=seen.append)
        self.assertEqual(result.answer, "You spent Rs. 1,000.00 this month.")
        self.assertIn("number_guard", [e["type"] for e in seen])
        self.assertIn("Rs. 100.00", client.requests[2][-1]["content"])

    def test_persistent_wrong_number_is_replaced_by_answer_built_from_tool_results(self):
        execute_tool("add_expense", {"amount": 1000, "category": "Food", "date": "today"})
        client = FakeClient([tool_reply("summarize_spending", month="this month")] + [text_reply("Total Rs. 5.00")] * 3)
        result = FinanceAgent(client=client).chat("How much did I spend this month?")
        self.assertNotIn("Rs. 5.00", result.answer)                    # the wrong figure is never shown
        self.assertIn("Rs. 1,000.00", result.answer)                   # the real one comes straight from the tool
        self.assertIn("built directly from the tool results", result.answer)


class ArgumentRepairTests(TempEnvTestCase):
    def test_truncated_converted_amount_is_repaired(self):
        client = FakeClient([
            tool_reply("convert_currency", amount=50, from_currency="EUR", to_currency="LKR"),
            tool_reply("add_expense", amount=16304, category="Shopping", description="shopping", date="today"),
            text_reply("Saved Rs. 16,304.35 under Shopping."),
        ])
        result = FinanceAgent(client=client).chat("Convert 50 EUR to LKR and add it as a Shopping expense for today")
        converted = result.events[0]["result"]["converted"]
        saved = result.events[1]["result"]["expense"]["amount"]
        self.assertEqual(saved, converted)                               # 16304.35, not 16304
        self.assertIn("argument_fix", result.events[1])

    def test_unrelated_amount_is_not_touched(self):
        client = FakeClient([
            tool_reply("convert_currency", amount=50, from_currency="EUR", to_currency="LKR"),
            tool_reply("add_expense", amount=5000, category="Shopping", description="x", date="today"),
            text_reply("Saved Rs. 5,000.00."),
        ])
        result = FinanceAgent(client=client).chat("convert 50 EUR and add 5000 shopping")
        self.assertEqual(result.events[1]["result"]["expense"]["amount"], 5000.0)
        self.assertNotIn("argument_fix", result.events[1])


class ToolDetailTests(TempEnvTestCase):
    def test_none_text_description_becomes_empty(self):
        self.assertEqual(clean_text("None", "description"), "")
        self.assertEqual(clean_text("null", "description"), "")
        self.assertEqual(clean_text("lunch", "description"), "lunch")
        result = execute_tool("add_expense", {"amount": 500, "category": "Food", "description": "None"})
        self.assertEqual(result["expense"]["description"], "")

    def test_budget_reports_over_by_and_warning_list(self):
        execute_tool("set_budget", {"category": "Shopping", "monthly_limit": 15000})
        execute_tool("set_budget", {"category": "Food", "monthly_limit": 1000})
        execute_tool("add_expense", {"amount": 32420, "category": "Shopping", "date": "today"})
        execute_tool("add_expense", {"amount": 850, "category": "Food", "date": "today"})
        result = execute_tool("check_budget", {"month": "this month"})
        shopping = next(b for b in result["budgets"] if b["category"] == "Shopping")
        self.assertEqual(shopping["over_by_lkr"], 17420.0)               # the model had said 18,420
        self.assertEqual(result["warning_categories"], ["Food"])

    def test_report_contains_real_budget_details(self):
        execute_tool("set_budget", {"category": "Shopping", "monthly_limit": 15000})
        execute_tool("add_expense", {"amount": 20000, "category": "Shopping", "date": "today"})
        result = execute_tool("generate_report", {"month": "this month"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["budget_details"][0]["over_by_lkr"], 5000.0)

    def test_summary_has_display_total(self):
        execute_tool("add_expense", {"amount": 206960, "category": "Rent", "date": "today"})
        self.assertEqual(execute_tool("summarize_spending", {"month": "this month"})["total_display"], "Rs. 206,960.00")


if __name__ == "__main__":
    unittest.main()
