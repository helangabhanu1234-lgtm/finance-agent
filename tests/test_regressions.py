"""Regression tests for the bugs found in live testing with llama3.2 (see logs/trace of 24 Oct):
duplicate expenses, 'Cannot reach Ollama' that was really a timeout, wrong figures, invented validation examples."""
import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

import config
import web_ui
from agents.finance_agent import FinanceAgent, friendly_error
from agents.formatters import render_answer
from agents.guardrails import misattributed_numbers, ungrounded_numbers
from tests.helpers import FakeClient, TempEnvTestCase, text_reply, tool_reply
from tools import execute_tool
from tools.db import connect


def count_expenses():
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]


class DuplicateExpenseTests(TempEnvTestCase):
    def test_real_trace_convert_and_add_is_saved_only_once(self):
        """Live run: add_expense ran at step 2 AND step 5 (ids 112 + 113) after a number-guard reminder."""
        client = FakeClient([
            tool_reply("convert_currency", amount=50, from_currency="USD", to_currency="LKR"),
            tool_reply("add_expense", amount=15000, category="Shopping", description="shopping", date="today"),
            text_reply("Saved Rs. 14,999.00 under Shopping."),                              # wrong figure -> reminder
            tool_reply("convert_currency", amount=50, from_currency="USD", to_currency="LKR"),     # model starts over...
            tool_reply("add_expense", description="shopping", amount=15000, category="Shopping", date="today"),
            text_reply("Saved Rs. 15,000.00 under Shopping."),
        ])
        result = FinanceAgent(client=client).chat("Convert 50 USD to LKR and add it as a Shopping expense for today")
        self.assertEqual(count_expenses(), 1)
        self.assertEqual(result.answer, "Saved Rs. 15,000.00 under Shopping.")
        blocked = [e for e in result.events if not e["ok"]]
        self.assertEqual({e["result"]["error_type"] for e in blocked}, {"repeat", "duplicate"})

    def test_cosmetic_differences_do_not_bypass_the_duplicate_guard(self):
        client = FakeClient([
            tool_reply("add_expense", amount=2500, category="Food", description="lunch", date="yesterday"),
            tool_reply("add_expense", amount="2500", category="food", description="None", date="yesterday", currency="LKR"),
            text_reply("Saved Rs. 2,500.00 under Food."),
        ])
        result = FinanceAgent(client=client).chat("I spent 2500 on lunch yesterday")
        self.assertEqual(count_expenses(), 1)
        self.assertIn("already saved", result.events[1]["result"]["error"])

    def test_two_different_expenses_are_both_saved(self):
        client = FakeClient([
            tool_reply("add_expense", amount=500, category="Food", description="tea"),
            tool_reply("add_expense", amount=700, category="Transport", description="bus"),
            text_reply("Saved Rs. 500.00 under Food and Rs. 700.00 under Transport."),
        ])
        FinanceAgent(client=client).chat("I spent 500 on tea and 700 on the bus")
        self.assertEqual(count_expenses(), 2)

    def test_duplicate_is_allowed_in_a_new_request(self):
        for _ in range(2):
            client = FakeClient([tool_reply("add_expense", amount=500, category="Food"), text_reply("Saved Rs. 500.00 under Food.")])
            FinanceAgent(client=client).chat("I spent 500 on food")
        self.assertEqual(count_expenses(), 2)       # two separate user requests are two separate expenses


class InvalidUserInputTests(TempEnvTestCase):
    def test_negative_amount_is_answered_from_the_tool_error_with_one_model_call(self):
        client = FakeClient([tool_reply("add_expense", amount="-500", category="Food")])   # only ONE scripted reply
        result = FinanceAgent(client=client).chat("Add an expense of -500 for food")
        self.assertEqual(len(client.requests), 1)
        self.assertIn("amount must be greater than 0", result.answer)
        self.assertIn("Nothing was saved", result.answer)
        self.assertNotIn("valid example of a negative", result.answer)
        self.assertEqual(count_expenses(), 0)
        self.assertEqual(result.events[0]["result"]["error_type"], "invalid_user_input")

    def test_recoverable_error_still_goes_back_to_the_model(self):
        client = FakeClient([
            tool_reply("add_expense", amount=100, category="spaceships"),
            tool_reply("add_expense", amount=100, category="Other"),
            text_reply("Saved Rs. 100.00 under Other."),
        ])
        FinanceAgent(client=client).chat("spent 100 on a spaceship")
        self.assertEqual(count_expenses(), 1)

    def test_answer_after_a_failed_validation_is_kept_in_history(self):
        agent = FinanceAgent(client=FakeClient([tool_reply("add_expense", amount=-1, category="Food")]))
        agent.chat("add -1 for food")
        self.assertEqual(agent.messages[-1]["role"], "assistant")
        self.assertIn("greater than 0", agent.messages[-1]["content"])


class OllamaErrorTests(unittest.TestCase):
    def test_timeout_is_not_reported_as_connection_problem(self):
        class ReadTimeout(Exception):
            pass
        for exc in (ReadTimeout("timed out"), TimeoutError("The read operation timed out")):
            message = friendly_error(exc)
            self.assertIn("did not answer", message)
            self.assertNotIn("Cannot reach", message)

    def test_real_connection_failure_still_says_cannot_reach(self):
        self.assertIn("Cannot reach Ollama", friendly_error(ConnectionError("Failed to connect to Ollama")))
        self.assertIn("Cannot reach Ollama", friendly_error(ConnectionRefusedError("[WinError 10061] refused")))

    def test_speed_options_are_sent_to_ollama(self):
        client = FakeClient([text_reply("hi")])
        FinanceAgent(client=client).chat("hello")
        self.assertEqual(client.kwargs["keep_alive"], config.KEEP_ALIVE)


class FigureGuardTests(TempEnvTestCase):
    EVENTS = [{"tool": "convert_currency", "ok": True, "arguments": {}, "result": {"converted": 16600.51}}]

    def test_figure_with_decimals_must_match_exactly(self):
        # live run: "Rs. 16,601.00" was accepted although the saved amount was 16,600.51
        self.assertEqual(ungrounded_numbers("Saved Rs. 16,601.00", self.EVENTS, ""), ["Rs. 16,601.00"])
        self.assertEqual(ungrounded_numbers("Saved Rs. 16,600.51", self.EVENTS, ""), [])
        self.assertEqual(ungrounded_numbers("About Rs. 16,601", self.EVENTS, ""), [])

    def test_category_figure_from_another_category_is_detected(self):
        # live run: "Food: Rs. 2,500.00" while the tool said Food = 22,510 (2,500 was the lunch just added)
        events = [{"tool": "summarize_spending", "ok": True, "arguments": {}, "result": {
            "total_lkr": 67510.0, "by_category": [
                {"category": "Rent", "total_lkr": 45000.0, "percent": 66.7},
                {"category": "Food", "total_lkr": 22510.0, "percent": 33.3}]}}]
        answer = "* Rent: Rs. 45,000.00\n* Food: Rs. 2,500.00\nTotal Rs. 67,510.00"
        pairs = misattributed_numbers(answer, events)
        self.assertEqual([(c, f) for c, f, _ in pairs], [("Food", "Rs. 2,500.00")])
        self.assertEqual(misattributed_numbers("* Rent: Rs. 45,000.00\n* Food: Rs. 22,510.00", events), [])

    def test_confirmation_lines_are_not_flagged(self):
        events = [{"tool": "summarize_spending", "ok": True, "arguments": {}, "result": {
            "total_lkr": 22510.0, "by_category": [{"category": "Food", "total_lkr": 22510.0, "percent": 100}]}}]
        self.assertEqual(misattributed_numbers("Saved Rs. 2,500.00 under Food.", events), [])

    def test_swapped_category_figures_are_corrected_in_the_loop(self):
        execute_tool("add_expense", {"amount": 1000, "category": "Food", "date": "today"})
        execute_tool("add_expense", {"amount": 500, "category": "Shopping", "date": "today"})
        client = FakeClient([
            tool_reply("summarize_spending", month="this month"),
            text_reply("- Food: Rs. 500.00\n- Shopping: Rs. 1,000.00"),          # both numbers exist, but swapped
            text_reply("- Food: Rs. 1,000.00\n- Shopping: Rs. 500.00"),
        ])
        result = FinanceAgent(client=client).chat("Where did my money go this month?")
        self.assertIn("Food: Rs. 1,000.00", result.answer)
        self.assertIn("wrong category", client.requests[2][-1]["content"])

    def test_renderer_builds_exact_budget_and_report_answer(self):
        events = [
            {"tool": "check_budget", "ok": True, "result": {"month": "October 2026", "budgets": [
                {"category": "Food", "limit_lkr": 25000, "spent_lkr": 25010, "percent_used": 100.0, "status": "OVER", "over_by_lkr": 10},
                {"category": "Transport", "limit_lkr": 18000, "spent_lkr": 16360, "percent_used": 90.9, "status": "WARNING"}]}},
            {"tool": "generate_report", "ok": True, "result": {"month": "October 2026", "file": "D:\\finance_agent\\reports\\report_2026-10.html",
                                                              "total_lkr": 190360.51, "top_category": "Rent"}},
        ]
        text = render_answer(events)
        self.assertIn("Food: OVER by Rs. 10.00", text)
        self.assertIn("report_2026-10.html", text)
        self.assertNotIn("D:\\", text)
        self.assertIsNone(render_answer([{"tool": "add_expense", "ok": False, "result": {"error": "x"}}]))


class HistoryTests(TempEnvTestCase):
    def test_reminders_are_not_kept_in_history_and_old_tool_results_shrink(self):
        for cat in ("Food", "Groceries", "Transport", "Bills", "Rent", "Shopping"):
            execute_tool("add_expense", {"amount": 1000, "category": cat, "date": "today"})
        client = FakeClient([
            tool_reply("summarize_spending", month="this month"),
            text_reply("You spent Rs. 5.00."),                                  # wrong -> reminder
            text_reply("You spent Rs. 6,000.00."),
            text_reply("Sure, anything else?"),
        ])
        agent = FinanceAgent(client=client)
        agent.chat("How much did I spend this month?")
        self.assertFalse(any("[system check]" in (m.get("content") or "") for m in agent.messages))
        self.assertFalse(any(m.get("content") == "You spent Rs. 5.00." for m in agent.messages))
        agent.chat("thanks")
        tool_msgs = [m for m in client.requests[-1] if m["role"] == "tool"]
        self.assertTrue(tool_msgs and tool_msgs[0]["content"].endswith("[older result shortened]"))

    def test_step_limit_returns_what_was_completed(self):
        execute_tool("add_expense", {"amount": 1000, "category": "Food", "date": "today"})
        client = FakeClient([tool_reply("summarize_spending", month=m) for m in ("this month", "last month", "2026-01")])
        result = FinanceAgent(client=client, max_steps=3).chat("keep going")
        self.assertTrue(result.hit_step_limit)
        self.assertIn("Here is what was completed", result.answer)


class WebStreamTests(TempEnvTestCase):
    def test_chat_endpoint_streams_trace_events_live(self):
        web_ui.AGENT = FinanceAgent(client=FakeClient([
            tool_reply("add_expense", amount=2500, category="Food", description="lunch", date="yesterday"),
            text_reply("Saved Rs. 2,500.00 under Food."),
        ]))
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
            conn.request("POST", "/api/chat", body=json.dumps({"message": "I spent 2500 on lunch yesterday"}))
            lines = [json.loads(l) for l in conn.getresponse().read().decode("utf-8").splitlines() if l.strip()]
        finally:
            server.shutdown()
            server.server_close()
            web_ui.AGENT = None
        types = [l["type"] for l in lines]
        self.assertEqual(types, ["model_call", "tool_call", "tool_result", "model_call", "final", "done"])
        self.assertEqual(lines[-1]["answer"], "Saved Rs. 2,500.00 under Food.")
        self.assertEqual(count_expenses(), 1)


if __name__ == "__main__":
    unittest.main()
