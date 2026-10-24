"""Central configuration. Values can be overridden with environment variables or a .env file."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (no extra dependency). Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = Path(os.getenv("FINANCE_DB", DATA_DIR / "finance.db"))

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "600"))   # seconds; llama3.2 on a CPU-only PC can need 1-3 min per call
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")           # keep the model in RAM between questions (default 5 min -> slow reload)
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "6144"))           # explicit context window so the system prompt is never cut off
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))    # max tokens per model reply

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))          # max LLM calls per user request
MAX_HISTORY_MESSAGES = int(os.getenv("AGENT_MAX_HISTORY", "40"))
MAX_INPUT_CHARS = 2000
MAX_TOOL_RESULT_CHARS = 4000

BASE_CURRENCY = "LKR"
OFFLINE_RATES_ONLY = os.getenv("FINANCE_OFFLINE", "0") == "1"   # skip live exchange-rate API
