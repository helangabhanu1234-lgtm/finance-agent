"""The agent loop.

user message -> LLM -> (tool calls?) -> run tools -> feed results back -> LLM -> ... -> final answer
Safety nets: max steps, repeated-call guard, tool errors returned as text, friendly Ollama errors.
"""
import json
import re
import time
from dataclasses import dataclass, field

import config
from agents.formatters import render_answer
from agents.guardrails import (
    build_number_reminder, build_pair_reminder, build_reminder, misattributed_numbers, missing_tools, ungrounded_numbers,
)
from agents.logging_setup import setup_logging, write_trace
from agents.prompts import build_system_prompt
from tools import TOOL_NAMES, TOOL_SCHEMAS, execute_tool
from tools.validators import ToolError, normalize_category, normalize_currency, parse_amount, parse_date

log = setup_logging()


class AgentError(Exception):
    """Problem talking to the model (Ollama down, model missing...)."""


@dataclass
class AgentResult:
    answer: str
    events: list = field(default_factory=list)   # tool-call trace for this request
    steps: int = 0
    hit_step_limit: bool = False
    error: bool = False


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return {}


def _extract_tool_calls(message: dict) -> list:
    calls = []
    for raw in message.get("tool_calls") or []:
        raw = _as_dict(raw)
        fn = _as_dict(raw.get("function", raw))
        if fn.get("name"):
            calls.append({"name": fn["name"], "arguments": fn.get("arguments") or {}})
    return calls


def _parse_text_tool_calls(content: str) -> list:
    """Small models sometimes print the call as JSON text instead of using the tool_calls field."""
    text = (content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text or text[0] not in "[{":
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    calls = []
    for item in data if isinstance(data, list) else [data]:
        if isinstance(item, dict) and item.get("name") in TOOL_NAMES:
            args = item.get("parameters", item.get("arguments", {}))
            calls.append({"name": item["name"], "arguments": args if isinstance(args, (dict, str)) else {}})
    return calls


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    kind = type(exc).__name__.lower()
    if "does not support tools" in low:
        return (f"The model '{config.MODEL}' does not support tool calling. "
                f"Use a tool-capable model, e.g. llama3.2 or qwen3:4b (ollama pull llama3.2).")
    if "not found" in low and "model" in low:
        return f"Model '{config.MODEL}' is not installed. Run: ollama pull {config.MODEL}"
    unreachable = "connecttimeout" in kind or any(k in low for k in ("connect", "refused", "unreachable", "failed to establish"))
    if not unreachable and ("timeout" in kind or "timed out" in low or "timeout" in low):
        # Ollama IS running, it just did not answer in time (model still loading / slow CPU). Do not blame the connection.
        return (f"Ollama is running but did not answer within {config.REQUEST_TIMEOUT:.0f} seconds "
                f"(the model is probably still loading or this PC is slow). Wait a moment and send the message again. "
                f"If it keeps happening, raise OLLAMA_TIMEOUT in .env or use a smaller model.")
    if unreachable:
        return (f"Cannot reach Ollama at {config.OLLAMA_HOST}. Start the Ollama app "
                f"(or run 'ollama serve') and try again.")
    return f"Model error: {text}"


WRITE_TOOL = "add_expense"
_INVALID_INPUT_HINTS = {
    "add_expense": 'Nothing was saved. Example: "I spent 500 on food today".',
    "convert_currency": 'Example: "Convert 50 USD to LKR".',
    "set_budget": 'Example: "Set a Food budget of 25000".',
}


def _write_key(name, args):
    """Normalised identity of an add_expense call (amount, category, date, currency - NOT the description).
    The model repeats the same save with cosmetic changes ('None' vs 'shopping', extra currency=LKR), so
    comparing raw arguments is not enough to stop duplicate rows."""
    if name != WRITE_TOOL or not isinstance(args, dict):
        return None
    try:
        return (parse_amount(args.get("amount")), normalize_category(args.get("category")),
                parse_date(args.get("date")).isoformat(), normalize_currency(args.get("currency") or config.BASE_CURRENCY))
    except ToolError:
        return None          # invalid call: let the tool itself report the error


def invalid_input_message(event: dict) -> str:
    hint = _INVALID_INPUT_HINTS.get(event["tool"], "Please check the value and try again.")
    return f"I couldn't do that: {event['result']['error']}. {hint}"


def make_client():
    import ollama   # imported lazily so unit tests run without the package
    return ollama.Client(host=config.OLLAMA_HOST, timeout=config.REQUEST_TIMEOUT)


class FinanceAgent:
    def __init__(self, client=None, model=None, max_steps=None):
        self._client = client
        self.model = model or config.MODEL
        self.max_steps = max_steps or config.MAX_STEPS
        self.messages = [{"role": "system", "content": build_system_prompt()}]

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def reset(self) -> None:
        """Forget the conversation (database data is kept)."""
        self.messages = [{"role": "system", "content": build_system_prompt()}]

    # ------------------------------------------------------------------ main entry
    def chat(self, user_text: str, on_event=None) -> AgentResult:
        emit = on_event or (lambda e: None)
        text = (user_text or "").strip()
        if not text:
            return AgentResult("Please type a question or an expense, e.g. 'I spent 1500 on lunch today'.", error=True)
        if len(text) > config.MAX_INPUT_CHARS:
            return AgentResult(f"Message too long ({len(text)} characters). Please keep it under {config.MAX_INPUT_CHARS}.", error=True)

        self.messages[0]["content"] = build_system_prompt()      # keeps 'today' correct in long sessions
        turn_start = len(self.messages)
        self._compact_old_results(turn_start)                    # shorter prompt = faster calls on a CPU-only PC
        self.messages.append({"role": "user", "content": text})
        log.info("USER: %s", text)

        events, seen, written = [], {}, {}
        scratch = []                  # reminder / nudge messages: used for this request only, removed afterwards
        nudged, reminders = False, 0

        def finish(answer, step, **flags):
            """Store the final answer in the history (without the scratch messages) and build the result."""
            self.messages[:] = [m for m in self.messages if not any(m is x for x in scratch)]
            self.messages.append({"role": "assistant", "content": answer})
            self._trim_history()
            emit({"type": "final", "step": step})
            return AgentResult(answer, events, step, **flags)

        for step in range(1, self.max_steps + 1):
            emit({"type": "model_call", "step": step})
            try:
                message = self._call_model()
            except AgentError as exc:
                del self.messages[turn_start:]                   # roll back so history stays consistent
                log.error("MODEL ERROR: %s", exc)
                return AgentResult(str(exc), events, step, error=True)

            content = (message.get("content") or "").strip()
            calls = _extract_tool_calls(message) or _parse_text_tool_calls(content)

            if not calls:
                if not content and events and not nudged:        # model went silent after tools: ask once more
                    nudged = True
                    nudge = {"role": "user", "content": "Now answer the user briefly using the tool results above."}
                    scratch.append(nudge)
                    self.messages.append(nudge)
                    continue
                answer = content or render_answer(events) or "Sorry, I could not produce an answer. Please rephrase your request."
                # guardrails: did the model do everything the user asked, and are its figures real?
                missing = missing_tools(text, events, answer)
                bad_numbers = [] if missing else ungrounded_numbers(answer, events, text)
                wrong_pairs = [] if (missing or bad_numbers) else misattributed_numbers(answer, events)
                if (missing or bad_numbers or wrong_pairs) and reminders < 2:
                    reminders += 1
                    if missing:
                        log.warning("GUARDRAIL step=%d missing=%s (reminder %d)", step, missing, reminders)
                        emit({"type": "guardrail", "step": step, "missing": missing})
                        reminder = build_reminder(missing, text, events)
                    elif bad_numbers:
                        log.warning("NUMBER GUARD step=%d unverified=%s (reminder %d)", step, bad_numbers, reminders)
                        emit({"type": "number_guard", "step": step, "numbers": bad_numbers})
                        reminder = build_number_reminder(bad_numbers)
                    else:
                        shown = [f"{cat}: {fig}" for cat, fig, _ in wrong_pairs]
                        log.warning("PAIR GUARD step=%d wrong=%s (reminder %d)", step, shown, reminders)
                        emit({"type": "number_guard", "step": step, "numbers": shown})
                        reminder = build_pair_reminder(wrong_pairs)
                    bad_answer = {"role": "assistant", "content": answer}
                    nag = {"role": "user", "content": reminder}
                    scratch.extend([bad_answer, nag])
                    self.messages.extend([bad_answer, nag])
                    continue
                if missing:                                      # still not done after 2 reminders: be honest
                    answer += f"\n\n(Warning: this part was NOT completed: {', '.join(missing)}. Please try again.)"
                elif bad_numbers or wrong_pairs:
                    rebuilt = render_answer(events)
                    if rebuilt:                                  # never show unverified figures: rebuild from the tool results
                        log.warning("ANSWER REBUILT from tool results (unverified: %s)", bad_numbers or wrong_pairs)
                        emit({"type": "number_guard", "step": step, "numbers": ["answer rebuilt from tool results"]})
                        answer = rebuilt + "\n\n(Answer built directly from the tool results because the model's figures did not match the data.)"
                    else:
                        shown = bad_numbers or [fig for _, fig, _ in wrong_pairs]
                        answer += (f"\n\n(Warning: these figures could not be verified against the data: "
                                   f"{', '.join(shown)}. Please check the tool trace.)")
                log.info("ASSISTANT: %s", answer)
                return finish(answer, step)

            self.messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], dict) else {}}} for c in calls],
            })
            step_events = []
            for call in calls:
                name, args = call["name"], call["arguments"]
                fix_note = self._repair_arguments(name, args, events)
                signature = (name, json.dumps(args, sort_keys=True, default=str))
                seen[signature] = seen.get(signature, 0) + 1
                write_key = _write_key(name, args)
                emit({"type": "tool_call", "step": step, "name": name, "arguments": args})
                if fix_note:
                    emit({"type": "arg_fix", "step": step, "name": name, "note": fix_note})
                started = time.perf_counter()
                if write_key is not None and write_key in written:
                    # same expense (amount/category/date/currency) already saved in THIS request: never insert it twice
                    result = {"ok": False, "error_type": "duplicate",
                              "error": f"This expense was already saved in this request (id {written[write_key]}). "
                                       f"Do NOT call add_expense again - just tell the user it is saved."}
                elif seen[signature] > 1:
                    result = {"ok": False, "error_type": "repeat",
                              "error": "You already made this exact call. Stop repeating it and answer the user with what you have."}
                else:
                    result = execute_tool(name, args)
                    if write_key is not None and result.get("ok"):
                        written[write_key] = result["expense"]["id"]
                ms = int((time.perf_counter() - started) * 1000)
                event = {"step": step, "tool": name, "arguments": args, "ok": result.get("ok", False), "result": result, "ms": ms}
                if fix_note:
                    event["argument_fix"] = fix_note
                events.append(event)
                step_events.append(event)
                write_trace(event)
                log.info("STEP %d TOOL_CALL %s args=%s -> ok=%s (%d ms) result=%s",
                         step, name, json.dumps(args, default=str), result.get("ok"), ms, json.dumps(result, default=str)[:500])
                emit({"type": "tool_result", "step": step, "name": name, "ok": result.get("ok", False), "result": result, "ms": ms})
                payload = json.dumps(result, ensure_ascii=False, default=str)
                if len(payload) > config.MAX_TOOL_RESULT_CHARS:
                    payload = payload[:config.MAX_TOOL_RESULT_CHARS] + '... [truncated]'
                self.messages.append({"role": "tool", "tool_name": name, "content": payload})

            # The USER's value is invalid (negative amount...): no argument change can fix it and a 3B model tends to
            # invent examples when asked to explain. Answer from the tool's own error message and save a slow model call.
            bad_input = [e for e in step_events if e["result"].get("error_type") == "invalid_user_input"]
            if bad_input and not any(e["ok"] for e in events):
                answer = invalid_input_message(bad_input[0])
                log.info("ASSISTANT (invalid input): %s", answer)
                return finish(answer, step)

        # loop ended without a final answer
        done = render_answer(events)
        answer = (f"I stopped after {self.max_steps} steps without finishing. "
                  + (f"Here is what was completed:\n{done}" if done else "Please try a simpler or more specific request."))
        log.warning("STEP LIMIT reached (%d)", self.max_steps)
        return finish(answer, self.max_steps, hit_step_limit=True)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _repair_arguments(name, args, events):
        """llama3.2 often drops the decimals of a converted amount (16600.51 -> 16600). If add_expense follows a
        successful convert_currency and the amount is within 1.0 of the converted value, use the exact value."""
        if name != "add_expense" or not isinstance(args, dict):
            return None
        converted = [e["result"].get("converted") for e in events if e["tool"] == "convert_currency" and e["ok"]]
        if not converted or str(args.get("currency") or "LKR").upper() != "LKR":
            return None
        try:
            given = float(str(args.get("amount")).replace(",", ""))
        except ValueError:
            return None
        exact = converted[-1]
        if exact is not None and given != exact and abs(given - exact) <= 1.0:
            args["amount"] = exact
            return f"amount {given:g} replaced by the exact converted value {exact:g}"
        return None

    def _call_model(self) -> dict:
        try:
            response = self.client.chat(
                model=self.model, messages=self.messages, tools=TOOL_SCHEMAS,
                options={"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX, "num_predict": config.NUM_PREDICT},
                keep_alive=config.KEEP_ALIVE,
            )
        except Exception as exc:
            log.error("OLLAMA CALL FAILED: %s: %r", type(exc).__name__, exc)     # the REAL reason, not only the friendly text
            raise AgentError(friendly_error(exc)) from exc
        return _as_dict(_as_dict(response).get("message"))

    def _compact_old_results(self, upto: int, limit: int = 300) -> None:
        """Shrink tool results of EARLIER requests. They stay in the history (so follow-ups like 'make it for last
        month' still know what happened) but no longer make every later model call slower."""
        for message in self.messages[1:upto]:
            content = message.get("content") or ""
            if message.get("role") == "tool" and len(content) > limit and not content.endswith("[older result shortened]"):
                message["content"] = content[:limit] + " ... [older result shortened]"

    def _trim_history(self) -> None:
        """Keep the system prompt + the most recent messages, cutting only at user-message boundaries."""
        while len(self.messages) > config.MAX_HISTORY_MESSAGES + 1:
            del self.messages[1]
            while len(self.messages) > 1 and self.messages[1]["role"] != "user":
                del self.messages[1]


def check_ollama(model: str = None):
    """Return (ok, message). Verifies the server is reachable and the model is installed."""
    model = model or config.MODEL
    try:
        listing = _as_dict(make_client().list())
    except Exception as exc:
        return False, friendly_error(exc)
    names = []
    for m in listing.get("models", []):
        m = _as_dict(m)
        names.append(m.get("model") or m.get("name") or "")
    wanted = {model, model + ":latest"}
    if not any(n in wanted or (":" not in model and n.split(":")[0] == model) for n in names):
        return False, f"Model '{model}' is not installed. Installed: {names or 'none'}. Run: ollama pull {model}"
    return True, f"Ollama OK, model '{model}' available."
