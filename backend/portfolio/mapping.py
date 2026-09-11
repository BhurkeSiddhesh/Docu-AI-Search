"""Conservative, explainable column suggestions; confirmation lives in the API."""
import re

CANONICAL_FIELDS = (
    "trade_date", "settlement_date", "symbol", "security_name", "asset_type",
    "transaction_type", "side", "quantity", "price", "gross_amount", "fees",
    "taxes", "net_amount", "currency", "account_id", "broker", "portfolio_owner",
    "exchange", "isin", "cusip", "notes",
)

ALIASES = {
    "trade_date": ("date", "trade date", "transaction date", "activity date", "run date", "executed at", "order executed time", "trade datetime"),
    "settlement_date": ("settlement date", "settle date"),
    "symbol": ("symbol", "ticker", "instrument", "scrip", "security symbol", "isin symbol", "tradingsymbol", "trading symbol"),
    "security_name": ("security name", "description", "instrument name"),
    "asset_type": ("asset type", "asset class", "security type"),
    "transaction_type": ("transaction category", "event", "activity type", "transaction code", "trans code"),
    "side": ("side", "action", "type", "transaction type", "buy sell", "trade type", "order type"),
    "quantity": ("quantity", "qty", "shares", "units", "filled qty", "filled quantity", "executed quantity"),
    "price": ("price", "avg price", "average price", "trade price", "execution price", "rate", "current price", "last price"),
    "gross_amount": ("gross amount", "gross value", "trade value", "principal", "proceeds", "gross proceeds"),
    "fees": ("fees", "fee", "commission", "brokerage", "charges", "transaction fee", "total fees", "commissions fees"),
    "taxes": ("tax", "taxes", "total tax", "withholding tax", "withholding"),
    "net_amount": ("amount", "net amount", "net value", "debit credit", "total", "value", "net", "dividend amount", "dividend", "cash amount"),
    "currency": ("currency", "ccy", "curr", "currency code"),
    "account_id": ("account", "account id", "account number", "account name"),
    "broker": ("broker", "broker name"),
    "portfolio_owner": ("owner", "portfolio owner", "client name", "account holder", "investor name"),
    "exchange": ("exchange", "market"), "isin": ("isin",), "cusip": ("cusip",),
    "notes": ("notes", "note", "remarks", "memo"),
}


def normalized_header(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def header_candidates(headers):
    """Return all aliases, retaining conflicts rather than silently dropping fees."""
    return {field: [h for h in headers if normalized_header(h) in aliases]
            for field, aliases in ALIASES.items()}


def propose_mapping(headers, rows=None):
    from .classifier import classify_table

    candidates = header_candidates(headers)
    mapping, warnings = {}, []
    for field, columns in candidates.items():
        if len(columns) == 1:
            mapping[field] = columns[0]
        elif len(columns) > 1:
            warnings.append(f"Multiple columns could represent {field}: {', '.join(columns)}. Select the correct column; combine charges in the source if necessary.")
    # The only value inference permitted automatically is a categorical action;
    # numeric columns cannot be distinguished safely from values alone.
    if "side" not in mapping and "transaction_type" not in mapping and rows:
        action_columns = []
        known = {"buy", "sell", "bought", "sold", "b", "s", "dividend", "deposit", "withdrawal", "fee", "interest"}
        for header in headers:
            values = [str((r.get("values", r)).get(header, "")).strip().casefold() for r in rows[:20]]
            values = [v for v in values if v]
            if values and all(v in known for v in values):
                action_columns.append(header)
        if len(action_columns) == 1:
            mapping["side"] = action_columns[0]
            warnings.append("Action column inferred from sample values; confirm its meaning.")
    detected = classify_table(headers, rows, mapping)
    confidence = detected["confidence"]
    if warnings:
        confidence = min(confidence, 0.6)
    if detected["detected_type"] == "trade_history":
        missing = [f for f in ("trade_date", "symbol", "quantity") if f not in mapping]
        if not ("side" in mapping or "transaction_type" in mapping):
            missing.append("side")
        if not any(f in mapping for f in ("price", "gross_amount", "net_amount")):
            missing.append("price or amount")
        if missing:
            warnings.append("Missing required trade fields: " + ", ".join(missing))
            confidence = min(confidence, 0.4)
    if "currency" not in mapping:
        warnings.append("Currency is missing; set a currency during review.")
    return {"mapping": mapping, "confidence": confidence, "warnings": warnings,
            "detected_type": detected["detected_type"]}
