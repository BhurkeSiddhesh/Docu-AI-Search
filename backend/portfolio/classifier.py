"""Classify financial tables without mistaking a holdings snapshot for trades."""
from .mapping import header_candidates, normalized_header


def classify_table(headers, rows=None, mapping=None):
    if mapping is None:
        mapping = {key: vals[0] for key, vals in header_candidates(headers).items() if len(vals) == 1}
    fields = set(mapping)
    names = {normalized_header(h) for h in headers}
    action = bool(fields & {"side", "transaction_type"})
    monetary = bool(fields & {"net_amount", "gross_amount", "price"})
    if {"symbol", "quantity"} <= fields and not action:
        return {"detected_type": "holdings", "confidence": 0.9}
    if action and "symbol" in fields and "quantity" in fields:
        return {"detected_type": "trade_history", "confidence": 0.95 if "trade_date" in fields and monetary else 0.5}
    if "trade_date" in fields and any("dividend" in name for name in names):
        return {"detected_type": "dividend_statement", "confidence": 0.9}
    if "trade_date" in fields and monetary and action:
        return {"detected_type": "cash_ledger", "confidence": 0.85}
    if names & {"market value", "cost basis", "unrealized gain loss", "unrealized pnl"}:
        return {"detected_type": "holdings", "confidence": 0.8}
    return {"detected_type": "unsupported", "confidence": 0.2}
