"""Tool: convert_currency. Uses the free open.er-api.com API (no key) with an offline fallback table."""
import json
import time
import urllib.request

import config
from tools.validators import ToolError, normalize_currency, parse_amount

# Approximate rates per 1 USD, used only when the live API is unreachable (clearly flagged in output).
FALLBACK_PER_USD = {
    "USD": 1.0, "LKR": 300.0, "EUR": 0.92, "GBP": 0.78, "INR": 83.5, "AUD": 1.52,
    "CAD": 1.37, "SGD": 1.35, "JPY": 150.0, "AED": 3.67, "CNY": 7.2, "MYR": 4.7, "SAR": 3.75,
}
_CACHE = {}          # base -> (timestamp, rates)
_TTL_SECONDS = 3600


def _fetch_rates(base: str) -> dict:
    url = f"https://open.er-api.com/v6/latest/{base}"
    with urllib.request.urlopen(url, timeout=5) as resp:       # nosec - fixed https URL
        data = json.load(resp)
    if data.get("result") != "success" or "rates" not in data:
        raise RuntimeError("exchange-rate API returned an error")
    return data["rates"]


def get_rate(from_code: str, to_code: str):
    """Return (rate, source). source is 'live' or 'offline_fallback'."""
    if from_code == to_code:
        return 1.0, "same_currency"
    if not config.OFFLINE_RATES_ONLY:
        try:
            cached = _CACHE.get(from_code)
            if cached and time.time() - cached[0] < _TTL_SECONDS:
                rates = cached[1]
            else:
                rates = _fetch_rates(from_code)
                _CACHE[from_code] = (time.time(), rates)
            if to_code in rates:
                return float(rates[to_code]), "live"
        except Exception:
            pass   # fall through to the offline table
    if from_code in FALLBACK_PER_USD and to_code in FALLBACK_PER_USD:
        return FALLBACK_PER_USD[to_code] / FALLBACK_PER_USD[from_code], "offline_fallback"
    raise ToolError(
        f"Cannot convert {from_code} to {to_code}: unsupported currency and the live rate service is unavailable. "
        f"Offline-supported codes: {', '.join(sorted(FALLBACK_PER_USD))}"
    )


def to_base_currency(amount: float, currency: str):
    """Convert to LKR. Returns (amount_lkr, rate, source)."""
    rate, source = get_rate(currency, config.BASE_CURRENCY)
    return round(amount * rate, 2), rate, source


def convert_currency(amount, from_currency, to_currency="LKR") -> dict:
    value = parse_amount(amount)
    src = normalize_currency(from_currency, "from_currency")
    dst = normalize_currency(to_currency or config.BASE_CURRENCY, "to_currency")
    rate, source = get_rate(src, dst)
    result = {
        "ok": True, "amount": value, "from": src, "to": dst,
        "rate": round(rate, 6), "converted": round(value * rate, 2), "rate_source": source,
    }
    if source == "offline_fallback":
        result["note"] = "Live rates unavailable; this is an approximate offline rate."
    return result
