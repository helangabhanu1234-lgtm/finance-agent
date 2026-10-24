"""Shared test helpers: temp database + a scripted fake Ollama client."""
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import config
from tools import currency_tools


class TempEnvTestCase(unittest.TestCase):
    """Points the app at a throw-away data/report/log folder and forces offline exchange rates."""

    def setUp(self):
        # unit tests must not write fake "USER: hello" / error lines into the real logs/agent.log
        self._log = logging.getLogger("finance_agent")
        self._log_was_disabled = self._log.disabled
        self._log.disabled = True
        self.tmp = Path(tempfile.mkdtemp())
        self._saved = (config.DB_PATH, config.DATA_DIR, config.REPORTS_DIR, config.LOGS_DIR, config.OFFLINE_RATES_ONLY)
        config.DATA_DIR = self.tmp / "data"
        config.DATA_DIR.mkdir()
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.REPORTS_DIR = self.tmp / "reports"
        config.LOGS_DIR = self.tmp / "logs"
        config.OFFLINE_RATES_ONLY = True
        currency_tools._CACHE.clear()

    def tearDown(self):
        self._log.disabled = self._log_was_disabled
        (config.DB_PATH, config.DATA_DIR, config.REPORTS_DIR, config.LOGS_DIR, config.OFFLINE_RATES_ONLY) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)


class FakeClient:
    """Returns pre-scripted responses, like ollama.Client().chat(). Records every request."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []

    def chat(self, model, messages, tools=None, options=None, **kwargs):
        self.requests.append([dict(m) for m in messages])
        self.kwargs = kwargs
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_reply(name, **arguments):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}}


def text_reply(text):
    return {"message": {"role": "assistant", "content": text}}
