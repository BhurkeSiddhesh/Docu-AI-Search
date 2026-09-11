"""Convert reviewed tables to exact ledger rows, never guessing missing money."""
import datetime as dt
from decimal import Decimal, InvalidOperation
import re

from .mapping import CANONICAL_FIELDS

ZERO = Decimal("0")
CURRENCIES = set("AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD BTN BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VES VND VUV WST XAF XCD XOF XPF YER ZAR ZMW ZWL".split())
BUY = {"buy", "b", "bought", "purchase", "purchased", "you bought", "buy to open"}
SELL = {"sell", "s", "sold", "sale", "you sold", "sell to close"}
EVENTS = {
    "dividend": "dividend", "div": "dividend", "cdiv": "dividend", "cash dividend": "dividend", "dividends": "dividend",
    "interest": "interest", "int": "interest", "interest income": "interest",
    "fee": "fee", "fees": "fee", "commission": "fee", "brokerage": "fee",
    "tax": "tax", "taxes": "tax", "withholding tax": "tax",
    "deposit": "deposit", "cash deposit": "deposit", "contribution": "deposit",
    "withdrawal": "withdrawal", "withdraw": "withdrawal", "cash withdrawal": "withdrawal",
}


def _empty(value):
    return value is None or (isinstance(value, str) and not value.strip())


def decimal_value(value):
    """Read exact ordinary, US, Indian, or European numbers; reject formulas."""
    if _empty(value):
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not amounts.")
    text = str(value).strip().replace("\u2212", "-")
    if len(text) > 80 or text.startswith(("=", "@")):
        raise ValueError("Formula or oversized numeric value is not supported.")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = re.sub(r"(?i)\b(?:" + "|".join(sorted(CURRENCIES)) + r")\b", "", text)
    text = re.sub(r"[$€£₹¥\s\u00a0]", "", text)
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{2})*,\d{3}\.\d+|[+-]?\d{1,3}(?:,\d{3})+\.\d+", text):
                raise ValueError("Malformed numeric grouping.")
            text = text.replace(",", "")
    elif "," in text:
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+|[+-]?\d{1,3}(?:,\d{2})*,\d{3}", text):
            text = text.replace(",", "")
        elif text.count(",") == 1 and len(text.rsplit(",", 1)[1]) in (1, 2):
            text = text.replace(",", ".")
        else:
            raise ValueError("Ambiguous numeric separators; use an unambiguous decimal format.")
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric value.") from exc
    if not result.is_finite() or abs(result) > Decimal("1e24") or result.as_tuple().exponent < -12:
        raise ValueError("Non-finite or out-of-range numeric value.")
    return -abs(result) if negative else result


def date_value(value, date_order=None):
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if _empty(value):
        raise ValueError("Trade date is required.")
    text = str(value).strip()
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    match = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?", text)
    if match:
        first, second, year = map(int, match.groups())
        order = str(date_order or "").upper()
        if first <= 12 and second <= 12 and first != second and order not in {"MDY", "DMY"}:
            raise ValueError("Ambiguous date; choose MDY or DMY in import review.")
        if order == "DMY" or first > 12:
            day, month = first, second
        else:
            month, day = first, second
        try:
            return dt.date(year, month, day).isoformat()
        except ValueError as exc:
            raise ValueError("Invalid calendar date.") from exc
    for fmt in ("%Y/%m/%d", "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError("Unrecognized date; use YYYY-MM-DD or an explicit MDY/DMY date.")


def _event(values, detected_type):
    raw_side = str(values.get("side") or "").strip().casefold()
    raw_type = str(values.get("transaction_type") or "").strip().casefold()
    for raw in (raw_side, raw_type):
        if any(word in raw for word in ("short", "cover", "split", "merger", "spin", "transfer", "adjustment", "reinvest")):
            raise ValueError(f"Unsupported transaction action '{raw}'; corporate actions, transfers, reinvestments, and short positions require separate accounting.")
    for raw in (raw_side, raw_type):
        if raw in BUY or raw.startswith("you bought "):
            return "trade", "buy"
        if raw in SELL or raw.startswith("you sold "):
            return "trade", "sell"
        if raw in EVENTS:
            return EVENTS[raw], "unknown"
    if detected_type == "dividend_statement" and not raw_side and not raw_type:
        return "dividend", "unknown"
    raise ValueError("Missing or unsupported transaction action; map an explicit buy, sell, dividend, interest, fee, tax, deposit, or withdrawal.")


def _normalize_row(row, mapping, defaults, detected_type):
    raw = row.get("values", {})
    values = {field: raw.get(column) for field, column in mapping.items() if column}
    warnings = list(row.get("warnings", []))
    result = {key: "" for key in CANONICAL_FIELDS}
    for field in ("account_id", "broker", "portfolio_owner", "currency"):
        if _empty(values.get(field)):
            values[field] = defaults.get(field, defaults.get("owner") if field == "portfolio_owner" else None)
    result["trade_date"] = date_value(values.get("trade_date"), defaults.get("date_order"))
    if not _empty(values.get("settlement_date")):
        result["settlement_date"] = date_value(values["settlement_date"], defaults.get("date_order"))
    currency = str(values.get("currency") or "").strip().upper()
    if currency not in CURRENCIES:
        raise ValueError("Missing or unsupported currency; set a three-letter currency code during review.")
    result["currency"] = currency
    result["transaction_type"], result["side"] = _event(values, detected_type)
    for field in ("account_id", "broker", "portfolio_owner", "security_name", "asset_type", "exchange", "isin", "cusip", "notes"):
        result[field] = str(values.get(field) or "").strip()
    result["symbol"] = str(values.get("symbol") or "").strip().upper()
    if not result["account_id"]:
        warnings.append("Account is missing; positions from unrelated source files cannot be safely combined.")
    if any(word in result["asset_type"].casefold() for word in ("option", "future", "derivative", "bond")):
        raise ValueError("This asset type requires contract or fixed-income accounting and is unsupported.")
    numbers = {}
    for field in ("quantity", "price", "gross_amount", "fees", "taxes", "net_amount"):
        try:
            numbers[field] = decimal_value(values.get(field))
        except ValueError as exc:
            raise ValueError(f"{field}: {exc}") from exc
    fees, taxes = abs(numbers["fees"] or ZERO), abs(numbers["taxes"] or ZERO)
    if numbers["fees"] is None and result["transaction_type"] == "trade":
        warnings.append("Fees are missing; calculations assume zero fees for this trade.")
    if numbers["taxes"] is None and result["transaction_type"] == "trade":
        warnings.append("Taxes are missing; calculations assume zero transaction taxes.")
    quantity, price, gross, net = (numbers[key] for key in ("quantity", "price", "gross_amount", "net_amount"))
    if result["transaction_type"] == "trade":
        if not result["symbol"]:
            raise ValueError("Symbol is required for a trade.")
        if quantity is None or quantity == 0:
            raise ValueError("Non-zero quantity is required for a trade.")
        if quantity < 0:
            if result["side"] != "sell":
                raise ValueError("Negative buy quantity is unsupported; verify the transaction action.")
            quantity = abs(quantity)
            warnings.append("Negative sell quantity normalized to a positive share count.")
        if price is not None and price <= 0:
            raise ValueError("Trade price must be positive.")
        if gross is not None:
            gross = abs(gross)
        if gross is None and price is not None:
            gross = quantity * price
            warnings.append("Gross amount calculated from quantity and price.")
        elif gross is None and net is not None:
            gross = abs(net) - fees - taxes if result["side"] == "buy" else abs(net) + fees + taxes
            warnings.append("Gross amount inferred from net cash and charges; verify the amount column represents net cash.")
        if gross is None or gross <= 0:
            raise ValueError("A positive price or gross/net amount is required for a trade.")
        if price is None:
            price = gross / quantity
            warnings.append("Price inferred from gross amount and quantity.")
        elif abs(quantity * price - gross) > Decimal("0.02"):
            raise ValueError("Gross amount does not reconcile with quantity × price; verify column mappings.")
        expected_net = -(gross + fees + taxes) if result["side"] == "buy" else gross - fees - taxes
        if net is not None and abs(abs(net) - abs(expected_net)) > Decimal("0.02"):
            raise ValueError("Net amount does not reconcile with gross amount and charges; verify fee and amount mappings.")
        if net is not None and net != expected_net:
            warnings.append("Cash sign normalized from the explicit trade side.")
        net = expected_net
    else:
        amount = net if net is not None else gross
        if amount is None:
            amount = numbers["fees"] if result["transaction_type"] == "fee" else numbers["taxes"] if result["transaction_type"] == "tax" else None
        if amount is None or amount == 0:
            raise ValueError("A non-zero cash amount is required for this transaction.")
        negative = result["transaction_type"] in {"fee", "tax", "withdrawal"}
        net = -abs(amount) if negative else abs(amount)
        # Fee/tax events carry the charge in net cash; do not duplicate in fees.
        if result["transaction_type"] in {"fee", "tax"}:
            fees, taxes = ZERO, ZERO
        gross = abs(gross) if gross is not None else abs(amount)
        quantity, price = ZERO, ZERO
    result.update({key: format(value, "f") for key, value in {
        "quantity": quantity, "price": price, "gross_amount": gross,
        "fees": fees, "taxes": taxes, "net_amount": net,
    }.items()})
    result.update(source_row_ref=row.get("row_ref", ""), raw_values=dict(raw), warnings=warnings,
                  confidence=1.0 if not warnings else 0.8)
    if row.get("formulas"):
        result["raw_values"]["_source_formulas"] = row["formulas"]
    return result


def normalize_rows(table, mapping, defaults=None):
    """Normalize a user-confirmed mapping; return row errors for atomic import."""
    result = {"transactions": [], "warnings": [], "errors": []}
    defaults = defaults or {}
    if any(field not in CANONICAL_FIELDS for field in mapping):
        result["errors"].append({"row_ref": "", "message": "Mapping contains an unknown canonical field."})
        return result
    headers = set(table.get("headers", []))
    selected = [column for column in mapping.values() if column]
    if any(column not in headers for column in selected) or len(selected) != len(set(selected)):
        result["errors"].append({"row_ref": "", "message": "Mapping contains missing or duplicate source columns."})
        return result
    detected_type = defaults.get("detected_type", table.get("detected_type", "unsupported"))
    if detected_type in {"holdings", "account_summary", "tax_report"}:
        result["errors"].append({"row_ref": "", "message": "Holdings and summary snapshots are not transaction histories; ignore this table or import a trade export."})
        return result
    for row in table.get("rows", []):
        try:
            transaction = _normalize_row(row, mapping, defaults, detected_type)
            result["transactions"].append(transaction)
            result["warnings"].extend({"row_ref": row.get("row_ref", ""), "message": warning} for warning in transaction["warnings"])
        except (ValueError, InvalidOperation, OverflowError) as exc:
            result["errors"].append({"row_ref": row.get("row_ref", ""), "message": str(exc)})
    return result
