import unittest

import config
from agents.finance_agent import FinanceAgent, friendly_error
from tests.helpers import FakeClient, TempEnvTestCase, text_reply, tool_reply


class AgentLoopTests(TempEnvTestCase):
    def test_single_tool_then_answer(self):
        client = FakeClient([
            tool_reply("add_expense", amount=1500, category="Food", description="lunch", date="yesterday"),
            text_reply("Saved Rs. 1,500.00 under Food."),
        ])
        agent = FinanceAgent(client=client)
        result = agent.chat("I spent 1500 on lunch yesterday")
        self.assertEqual(result.answer, "Saved Rs. 1,500.00 under Food.")
        self.assertEqual(len(result.events), 1)
        self.assertTrue(result.events[0]["ok"])
        # the tool result must be fed back to the model on the 2nd call
        roles = [m["role"] for m in client.requests[1]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])

    def test_multi_step_request(self):
        client = FakeClient([
            tool_reply("convert_currency", amount=50, from_currency="USD", to_currency="LKR"),
            tool_reply("add_expense", amount=15000, category="Shopping", description="shoes"),
            tool_reply("check_budget", month="this month"),
            text_reply("Done."),
        ])
        result = FinanceAgent(client=client).chat("convert 50 USD, add as shopping, check budget")
        self.assertEqual([e["tool"] for e in result.events], ["convert_currency", "add_expense", "check_budget"])
        self.assertEqual(result.steps, 4)

    def test_tool_error_is_returned_to_model_and_recovered(self):
        client = FakeClient([
            tool_reply("add_expense", amount=100, category="spaceships"),
            tool_reply("add_expense", amount=100, category="Other"),
            text_reply("Saved under Other."),
        ])
        result = FinanceAgent(client=client).chat("spent 100 on a spaceship")
        self.assertFalse(result.events[0]["ok"])
        self.assertTrue(result.events[1]["ok"])
        self.assertIn("Valid categories", client.requests[1][-1]["content"])

    def test_unknown_tool_does_not_crash(self):
        client = FakeClient([tool_reply("hack_bank"), text_reply("I cannot do that.")])
        result = FinanceAgent(client=client).chat("hack my bank")
        self.assertFalse(result.events[0]["ok"])
        self.assertEqual(result.answer, "I cannot do that.")

    def test_step_limit_prevents_infinite_loop(self):
        client = FakeClient([tool_reply("check_budget") for _ in range(20)])
        result = FinanceAgent(client=client, max_steps=3).chat("loop forever")
        self.assertTrue(result.hit_step_limit)
        self.assertEqual(len(client.requests), 3)

    def test_repeated_identical_call_is_blocked(self):
        client = FakeClient([tool_reply("check_budget") for _ in range(4)] + [text_reply("ok")])
        result = FinanceAgent(client=client, max_steps=8).chat("hi")
        self.assertFalse(result.events[2]["ok"])
        self.assertIn("already made", result.events[2]["result"]["error"])

    def test_json_in_text_is_treated_as_tool_call(self):
        text = '```json\n{"name": "convert_currency", "parameters": {"amount": 10, "from_currency": "USD"}}\n```'
        client = FakeClient([text_reply(text), text_reply("10 USD is Rs. 3,000.00")])
        result = FinanceAgent(client=client).chat("convert 10 usd")
        self.assertEqual(result.events[0]["tool"], "convert_currency")
        self.assertTrue(result.events[0]["ok"])

    def test_history_kept_between_turns(self):
        client = FakeClient([text_reply("Hello!"), text_reply("You said hi.")])
        agent = FinanceAgent(client=client)
        agent.chat("hi")
        agent.chat("what did I say?")
        contents = [m.get("content") for m in client.requests[1]]
        self.assertIn("hi", contents)
        self.assertIn("Hello!", contents)
        agent.reset()
        self.assertEqual(len(agent.messages), 1)

    def test_history_trim_keeps_valid_structure(self):
        old = config.MAX_HISTORY_MESSAGES
        config.MAX_HISTORY_MESSAGES = 6
        try:
            client = FakeClient([tool_reply("check_budget"), text_reply("a")] * 6)
            agent = FinanceAgent(client=client)
            for i in range(6):
                agent.chat(f"question {i}")
            self.assertLessEqual(len(agent.messages), 7)
            self.assertEqual(agent.messages[0]["role"], "system")
            self.assertEqual(agent.messages[1]["role"], "user")      # never starts with an orphan tool message
        finally:
            config.MAX_HISTORY_MESSAGES = old

    def test_invalid_user_input(self):
        agent = FinanceAgent(client=FakeClient([]))
        self.assertTrue(agent.chat("   ").error)
        self.assertTrue(agent.chat("x" * 5000).error)

    def test_connection_error_is_friendly_and_history_rolled_back(self):
        client = FakeClient([ConnectionError("Failed to connect to Ollama")])
        agent = FinanceAgent(client=client)
        result = agent.chat("hello")
        self.assertTrue(result.error)
        self.assertIn("Cannot reach Ollama", result.answer)
        self.assertEqual(len(agent.messages), 1)

    def test_silent_model_after_tool_gets_one_nudge(self):
        client = FakeClient([tool_reply("check_budget"), text_reply(""), text_reply("No budgets yet.")])
        result = FinanceAgent(client=client).chat("budget?")
        self.assertEqual(result.answer, "No budgets yet.")

    def test_trace_is_written_and_events_emitted(self):
        seen = []
        client = FakeClient([tool_reply("check_budget"), text_reply("ok")])
        FinanceAgent(client=client).chat("budget?", on_event=seen.append)
        self.assertEqual([e["type"] for e in seen if e["type"] != "model_call"], ["tool_call", "tool_result", "final"])
        self.assertEqual([e["type"] for e in seen].count("model_call"), 2)
        self.assertTrue((config.LOGS_DIR / "trace.jsonl").exists())

    def test_friendly_error_messages(self):
        self.assertIn("ollama pull", friendly_error(Exception("model 'x' not found")))
        self.assertIn("does not support tool", friendly_error(Exception("registry.ollama.ai/x does not support tools")))


if __name__ == "__main__":
    unittest.main()
