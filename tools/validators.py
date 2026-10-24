"""Input validation helpers shared by all tools. Every failure raises ToolError with a clear message."""
import calendar
import math
import re
from datetime import date, timedelta

CATEGORIES = [
    "Food", "Groceries", "Transport", "Bills", "Rent", "Shopping",
    "Entertainment", "Health", "Education", "Travel", "Savings", "Other",
]

_SYNONYMS = {
    "lunch": "Food", "dinner": "Food", "breakfast": "Food", "restaurant": "Food",
    "eating out": "Food", "snack": "Food", "snacks": "Food", "coffee": "Food", "meal": "Food",
    "grocery": "Groceries", "supermarket": "Groceries",
    "transportation": "Transport", "bus": "Transport", "taxi": "Transport", "fuel": "Transport",
    "petrol": "Transport", "uber": "Transport", "pickme": "Transport", "train": "Transport",
    "bill": "Bills", "electricity": "Bills", "water": "Bills", "internet": "Bills",
    "phone": "Bills", "utilities": "Bills", "utility": "Bills", "mobile": "Bills",
    "clothes": "Shopping", "clothing": "Shopping",
    "movie": "Entertainment", "movies": "Entertainment", "games": "Entertainment",
    "subscription": "Entertainment", "subscriptions": "Entertainment", "fun": "Entertainment",
    "medical": "Health", "medicine": "Health", "pharmacy": "Health", "doctor": "Health",
    "course": "Education", "courses": "Education", "books": "Education", "tuition": "Education",
    "trip": "Travel", "holiday": "Travel", "saving": "Savings",
    "misc": "Other", "miscellaneous": "Other", "others": "Other",
}
_CANON = {c.lower(): c for c in CATEGORIES}
_EMPTY_WORDS = {"none", "null", "nil", "n/a", "na", "undefined", "-"}


class ToolError(Exception):
    """A user-facing, expected error (bad input, missing file...). Returned to the model as text."""


class UserInputError(ToolError):
    """The USER's own value is invalid (e.g. a negative amount). Retrying with other arguments cannot fix it,
    so the agent answers immediately instead of asking the model to 'explain' (small models invent examples)."""


def normalize_category(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"category is required. Valid categories: {', '.join(CATEGORIES)}")
    key = value.strip().lower()
    if key in _CANON:
        return _CANON[key]
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    raise ToolError(f"Unknown category '{value}'. Valid categories: {', '.join(CATEGORIES)}")


def parse_amount(value, field: str = "amount") -> float:
    if value is None or isinstance(value, bool):
        raise ToolError(f"{field} is required and must be a positive number")
    if isinstance(value, str):
        found = re.findall(r"-?\d+(?:\.\d+)?|-?\.\d+", value.replace(",", ""))
        if len(found) != 1:        # none ("abc") or ambiguous ("5 to 10")
            raise ToolError(f"{field} '{value}' is not a valid single number")
        number = float(found[0])
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ToolError(f"{field} must be a number")
    if not math.isfinite(number):
        raise ToolError(f"{field} must be a finite number")
    if number <= 0:
        raise UserInputError(f"{field} must be greater than 0")
    if number > 1_000_000_000:
        raise UserInputError(f"{field} is unrealistically large (max 1,000,000,000)")
    return round(number, 2)


def parse_date(value, allow_future: bool = False, field: str = "date") -> date:
    today = date.today()
    if value is None or (isinstance(value, str) and not value.strip()):
        return today
    if not isinstance(value, str):
        raise ToolError(f"{field} must be text in YYYY-MM-DD format")
    text = value.strip().lower()
    if text == "today":
        return today
    if text == "yesterday":
        return today - timedelta(days=1)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ToolError(f"{field} '{value}' is invalid. Use YYYY-MM-DD, 'today' or 'yesterday'")
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ToolError(f"{field} '{value}' is not a real calendar date")
    if parsed > today and not allow_future:
        raise ToolError(f"{field} {parsed} is in the future (today is {today})")
    return parsed


def _month_range(year: int, month: int):
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def parse_month(value):
    """Return (first_day, last_day). Accepts YYYY-MM, 'this month', 'last month', or a month name."""
    today = date.today()
    if value is None or (isinstance(value, str) and not value.strip()):
        return _month_range(today.year, today.month)
    if not isinstance(value, str):
        raise ToolError("month must be text such as '2026-10', 'this month' or 'last month'")
    text = value.strip().lower()
    if text in {"this month", "current month", "current", "now"}:
        return _month_range(today.year, today.month)
    if text in {"last month", "previous month", "prev month"}:
        first = today.replace(day=1) - timedelta(days=1)
        return _month_range(first.year, first.month)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if not 1 <= month <= 12 or not 1900 <= year <= 2100:
            raise ToolError(f"month '{value}' is invalid. Use YYYY-MM, e.g. 2026-10")
        return _month_range(year, month)
    names = {n.lower(): i for i, n in enumerate(calendar.month_name) if n}
    names.update({n.lower(): i for i, n in enumerate(calendar.month_abbr) if n})
    if text in names:
        return _month_range(today.year, names[text])
    raise ToolError(f"month '{value}' is invalid. Use YYYY-MM, e.g. 2026-10, 'this month' or 'last month'")


def resolve_period(month=None, start_date=None, end_date=None):
    """Return (start, end) as ISO strings (either may be None) and a human label."""
    if month not in (None, ""):
        start, end = parse_month(month)
        return start.isoformat(), end.isoformat(), f"{start:%B %Y}"
    start = parse_date(start_date, allow_future=True, field="start_date") if start_date else None
    end = parse_date(end_date, allow_future=True, field="end_date") if end_date else None
    if start and end and start > end:
        raise ToolError("start_date must not be after end_date")
    if start or end:
        label = f"{start or 'beginning'} to {end or 'today'}"
    else:
        label = "all time"
    return (start.isoformat() if start else None, end.isoformat() if end else None, label)


def normalize_currency(value, field: str = "currency") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{3}", value.strip()):
        raise ToolError(f"{field} must be a 3-letter code such as LKR, USD, EUR")
    return value.strip().upper()


def clean_text(value, field: str, max_len: int = 120, required: bool = False) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ToolError(f"{field} is required")
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = " ".join(value.split())
    if value.lower() in _EMPTY_WORDS:        # small models often send the text "None" / "null"
        if required:
            raise ToolError(f"{field} is required")
        return ""
    if len(value) > max_len:
        raise ToolError(f"{field} is too long (max {max_len} characters)")
    return value
