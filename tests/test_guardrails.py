import unittest

from agents.finance_agent import FinanceAgent
from agents.guardrails import missing_tools, required_tools
from tests.helpers import FakeClient, TempEnvTestCase, text_reply, tool_reply


class RequiredToolTests(unittest.TestCase):
    def test_detects_needed_tools(self):
        self.assertEqual(required_tools("I spent 2500 on lunch yesterday"), ["add_expense"])
        self.assertEqual(required_tools("Convert 50 USD to LKR and add it as a Shopping expense for today"),
                         ["convert_currency", "add_expense"])
        self.assertIn("summarize_spending", required_tools("which category is the biggest?"))
        self.assertEqual(required_tools("Am I over budget? Then generate a report for this month."),
                         ["check_budget", "generate_report"])

    def test_no_false_positives(self):
        self.assertEqual(required_tools("Make it for last month instead"), [])
        self.assertEqual(required_tools("can you add up my spending"), [])
        self.assertEqual(required_tools("hello"), [])

    def test_conditional_report_only_when_over_budget(self):
        text = "Am I over budget? If yes, generate a report"
        ok_event = {"tool": "check_budget", "ok": True, "result": {"over_budget_categories": []}}
        self.assertEqual(missing_tools(text, [ok_event]), [])
        over_event = {"tool": "check_budget", "ok": True, "result": {"over_budget_categories": ["Food"]}}
        self.assertEqual(missing_tools(text, [over_event]), ["generate_report"])


class GuardrailLoopTests(TempEnvTestCase):
    def test_model_that_claims_save_without_tool_is_corrected(self):
        """Reproduces the real llama3.2 failure: convert, then 'Expense saved' without calling add_expense."""
        client = FakeClient([
            tool_reply("convert_currency", amount=50, from_currency="USD", to_currency="LKR"),
            text_reply("Expense saved. Rs. 15,000"),                       # wrong: nothing was saved
            tool_reply("add_expense", amount=15000, category="Shopping", description="shopping", date="today"),
            text_reply("Saved Rs. 15,000.00 under Shopping."),
        ])
        seen = []
        result = FinanceAgent(client=client).chat("Convert 50 USD to LKR and add it as a Shopping expense for today", on_event=seen.append)
        self.assertEqual([e["tool"] for e in result.events], ["convert_currency", "add_expense"])
        self.assertEqual(result.answer, "Saved Rs. 15,000.00 under Shopping.")
        self.assertIn("guardrail", [e["type"] for e in seen])
        reminder = client.requests[2][-1]["content"]
        self.assertIn("add_expense", reminder)
        self.assertIn("15000.0", reminder)                                 # converted amount hint

    def test_wrong_tool_for_biggest_category_is_corrected(self):
        client = FakeClient([
            tool_reply("get_expenses", month="this month", category="Other"),
            text_reply("Other is the biggest."),
            tool_reply("summarize_spending", month="this month"),
            text_reply("Rent is the biggest."),
        ])
        result = FinanceAgent(client=client).chat("Which category is the biggest this month?")
        self.assertEqual(result.events[-1]["tool"], "summarize_spending")
        self.assertEqual(result.answer, "Rent is the biggest.")

    def test_gives_up_with_honest_warning_after_two_reminders(self):
        client = FakeClient([text_reply("Done!")] * 3)
        result = FinanceAgent(client=client).chat("I spent 100 on tea")
        self.assertIn("NOT completed", result.answer)
        self.assertIn("add_expense", result.answer)
        self.assertEqual(len(client.requests), 3)

    def test_failed_add_is_not_forced_again(self):
        """A recoverable error (unknown category) is not 'forced' again by the guardrail once the model explains it."""
        client = FakeClient([
            tool_reply("add_expense", amount=100, category="spaceships"),
            text_reply("I cannot save that: unknown category. Valid categories are Food, Groceries, ..."),
        ])
        result = FinanceAgent(client=client).chat("Add an expense of 100 for spaceships")
        self.assertFalse(result.events[0]["ok"])
        self.assertNotIn("NOT completed", result.answer)
        self.assertEqual(len(client.requests), 2)

    def test_plain_question_not_affected(self):
        client = FakeClient([text_reply("Hello! How can I help with your finances?")])
        result = FinanceAgent(client=client).chat("hello")
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
