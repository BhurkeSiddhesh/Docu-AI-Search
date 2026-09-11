"""Deterministic, source-linked FIFO accounting over a normalized local ledger.

Amounts remain Decimal until the JSON boundary. Currency conversion, shorts,
corporate actions and tax-specific adjustments are deliberately unsupported.
"""

from collections import defaultdict, deque
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import json


ZERO = Decimal("0")
_SCOPE_FIELDS = ("symbol", "currency", "account_id", "broker", "portfolio_owner")
_UNSUPPORTED = {"short_sell", "buy_to_cover", "split", "merger", "adjustment", "transfer_in", "transfer_out", "unknown"}
_CASH_TYPES = {"dividend", "interest", "fee", "tax", "deposit", "withdrawal"}


def _decimal(value, default=None):
    if value is None or value == "":
        return default
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid decimal amount") from exc
    if not result.is_finite():
        raise ValueError("Non-finite decimal amount")
    return result


def _iso_date(value, name="date"):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc


def _json_safe(value):
    if isinstance(value, Decimal):
        if not value:
            return "0"
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _source(row):
    return {"source_file_id": row.get("source_file_id"), "filename": row.get("filename", ""),
            "sheet_name": row.get("sheet_name"), "source_row_ref": row.get("source_row_ref"),
            "transaction_id": row.get("id")}


def _scope(row):
    values = tuple(str(row.get(field) or "").strip() for field in _SCOPE_FIELDS)
    # Unidentified accounts from separate source files must never share lots.
    source_scope = str(row.get("source_file_id") or row.get("filename") or "") if not values[2] else ""
    return values + (source_scope,)


def _scope_dict(key):
    return dict(zip(_SCOPE_FIELDS + ("scope_source_file_id",), key))


def _new_summary(key):
    return {**_scope_dict(key), "realized_pnl": ZERO, "fees": ZERO, "taxes": ZERO,
            "dividends": ZERO, "interest": ZERO, "deposits": ZERO, "withdrawals": ZERO,
            "unrealized_pnl": ZERO, "is_complete": True, "realized_complete": True,
            "unrealized_complete": True, "sources": []}


def _allocate(remaining_amount, quantity, remaining_quantity):
    # The final allocation consumes the exact remainder, including any division
    # residue, so closing all partial lots neither creates nor loses a fee.
    return remaining_amount if quantity == remaining_quantity else remaining_amount * quantity / remaining_quantity


def _dedupe_sources(sources):
    seen, result = set(), []
    for source in sources:
        identity = tuple(str(source.get(field)) for field in
                         ("source_file_id", "filename", "sheet_name", "source_row_ref", "transaction_id"))
        if identity not in seen:
            seen.add(identity)
            result.append(source)
    return result


def _row_warnings(row):
    warnings = row.get("warnings") or row.get("warnings_json") or []
    if isinstance(warnings, str):
        try:
            warnings = json.loads(warnings)
        except (ValueError, TypeError):
            warnings = [warnings]
    if not isinstance(warnings, list):
        warnings = [warnings]
    return [str(item) for item in warnings if item]


def _matching_price(prices, scope, valuation_date):
    """Only accept dated prices in the same currency and compatible scope."""
    if not prices:
        return None, "No dated price available; unrealized P&L is excluded."
    if isinstance(prices, dict):
        raw = prices.get(scope["symbol"])
        candidates = raw if isinstance(raw, list) else [raw]
    else:
        candidates = list(prices)
    valid, reasons = [], []
    for quote in candidates:
        if not isinstance(quote, dict):
            continue
        if quote.get("symbol", scope["symbol"]) != scope["symbol"]:
            continue
        if any(quote.get(field) not in (None, "", scope[field])
               for field in ("account_id", "broker", "portfolio_owner")):
            continue
        if quote.get("currency") != scope["currency"]:
            reasons.append("Price currency is missing or differs from the position currency.")
            continue
        try:
            quote_date = _iso_date(quote.get("as_of") or quote.get("price_date") or quote.get("timestamp"))
            price = _decimal(quote.get("price", quote.get("current_price")))
        except ValueError:
            reasons.append("Price or timestamp is invalid.")
            continue
        if quote_date is None or price is None or price < ZERO:
            reasons.append("Price must be non-negative and have an explicit timestamp.")
            continue
        if quote_date > valuation_date:
            reasons.append("Future prices cannot value an earlier position.")
            continue
        if (valuation_date - quote_date).days > 7:
            reasons.append("Price is more than 7 days older than the valuation date; unrealized P&L is excluded.")
            continue
        valid.append((quote_date, price))
    if not valid:
        return None, reasons[0] if reasons else "No dated price available; unrealized P&L is excluded."
    quote_date, price = max(valid, key=lambda item: item[0])
    return {"current_price": price, "price_as_of": quote_date.isoformat()}, None


def calculate_portfolio(transactions, filters=None, prices=None):
    """Calculate FIFO P&L and positions, retaining buys before ``date_from``.

    Date filters include their boundary days. Realized P&L belongs to the sell
    date; fees/taxes and income belong to the transaction date. Positions reflect
    all transactions through date_to. Money and quantities in the result are
    decimal strings. Known matched P&L remains available when an oversell makes
    the complete result unknowable; unmatched proceeds never become profit.
    """
    with localcontext() as context:
        context.prec = 50
        return _calculate(transactions, dict(filters or {}), prices)


def _calculate(transactions, filters, prices):
    if str(filters.get("method", "FIFO")).upper() != "FIFO":
        raise ValueError("Only FIFO cost basis is supported")
    date_from = _iso_date(filters.get("date_from"), "date_from")
    date_to = _iso_date(filters.get("date_to"), "date_to")
    if date_from and date_to and date_from > date_to:
        raise ValueError("date_from must not be after date_to")
    valuation_date = date_to or datetime.now(timezone.utc).date()
    aliases = {"owner": "portfolio_owner", "account": "account_id"}
    selected, warnings, excluded, blocked = [], [], [], set()
    sources = []
    for order, original in enumerate(transactions):
        row = dict(original)
        if any(str(row.get(aliases.get(field, field)) or "").casefold() != str(value).casefold()
               for field, value in filters.items()
               if field in {*_SCOPE_FIELDS, "owner", "account"} and value not in (None, "")):
            continue
        key = _scope(row)
        source = _source(row)
        label = f"{row.get('filename') or 'source'} row {row.get('source_row_ref', row.get('id', '?'))}"
        try:
            trade_day = _iso_date(row.get("trade_date"), "trade_date")
            if trade_day is None:
                raise ValueError("Missing trade date")
            if date_to and trade_day > date_to:
                continue
            if not key[1]:
                raise ValueError("Missing currency")
        except ValueError as exc:
            warnings.append(f"{label}: {exc}; this scope is excluded from trusted calculations.")
            excluded.append({"source": source, "reason": str(exc)})
            blocked.add(key)
            sources.append(source)
            continue
        kind = str(row.get("transaction_type") or "trade").lower()
        side = str(row.get("side") or "").lower()
        # Some callers may supply buy/sell directly as the transaction type.
        if kind in {"buy", "sell"}:
            side, kind = kind, "trade"
        if kind in _UNSUPPORTED or side in {"short_sell", "buy_to_cover"} or kind not in {"trade", *_CASH_TYPES}:
            warnings.append(f"{label}: unsupported {side if side in {'short_sell', 'buy_to_cover'} else kind}; "
                            "the affected scope is excluded because its cost basis cannot be trusted.")
            excluded.append({"source": source, "reason": f"Unsupported {kind}/{side}"})
            blocked.add(key)
            sources.append(source)
            continue
        try:
            fees, taxes = _decimal(row.get("fees"), ZERO), _decimal(row.get("taxes"), ZERO)
            if fees < ZERO or taxes < ZERO:
                raise ValueError("Fees and taxes must be non-negative")
            row.update(_fees=fees, _taxes=taxes, _day=trade_day, _kind=kind, _side=side,
                       _key=key, _order=order, _source=source, _label=label)
            if kind == "trade":
                quantity = _decimal(row.get("quantity"))
                price = _decimal(row.get("price"))
                gross = _decimal(row.get("gross_amount"))
                if side not in {"buy", "sell"} or not key[0] or quantity is None or quantity <= ZERO:
                    raise ValueError("A trade needs a symbol, positive quantity and buy/sell side")
                if price is not None and price < ZERO:
                    raise ValueError("Trade price must be non-negative")
                if gross is None and price is not None:
                    gross = quantity * price
                if gross is None or gross < ZERO:
                    raise ValueError("A trade needs a non-negative gross amount or price")
                if price is not None and abs(gross - quantity * price) > max(Decimal("0.01"), abs(gross) * Decimal("0.0001")):
                    raise ValueError("Gross amount disagrees with quantity times price")
                row.update(_quantity=quantity, _gross=gross)
            else:
                gross = _decimal(row.get("gross_amount"))
                net = _decimal(row.get("net_amount"))
                # A standalone fee/tax may be represented by its charge column.
                if gross is None and net is None:
                    gross = fees if kind == "fee" else taxes if kind == "tax" else None
                if gross is None and net is None:
                    raise ValueError("Cash entry needs a gross or net amount")
                row.update(_gross=abs(gross) if gross is not None else None, _net=net)
            selected.append(row)
        except ValueError as exc:
            warnings.append(f"{label}: {exc}; this scope is excluded from trusted calculations.")
            excluded.append({"source": source, "reason": str(exc)})
            blocked.add(key)
            sources.append(source)

    selected.sort(key=lambda row: (row["_day"], str(row.get("trade_date")), row["_order"]))
    lots, summaries, realized, incomplete_scopes = defaultdict(deque), {}, [], set(blocked)
    for row in selected:
        key = row["_key"]
        if key in blocked:
            sources.append(row["_source"])
            continue
        summary = summaries.setdefault(key, _new_summary(key))
        within_range = date_from is None or row["_day"] >= date_from
        kind, source = row["_kind"], row["_source"]
        for warning in _row_warnings(row):
            warnings.append(f"{row['_label']}: {warning}")
        if within_range:
            summary["fees"] += row["_fees"]
            summary["taxes"] += row["_taxes"]
            summary["sources"].append(source)
            sources.append(source)
        if kind != "trade":
            if within_range:
                amount = row["_net"]
                if amount is None:
                    amount = row["_gross"] - row["_fees"] - row["_taxes"]
                if kind in {"dividend", "interest"}:
                    summary["dividends" if kind == "dividend" else "interest"] += amount
                elif kind in {"deposit", "withdrawal"}:
                    summary["deposits" if kind == "deposit" else "withdrawals"] += abs(amount)
                elif kind == "fee":
                    summary["fees"] += (row["_gross"] if row["_gross"] is not None else abs(row["_net"])) - row["_fees"]
                elif kind == "tax":
                    summary["taxes"] += (row["_gross"] if row["_gross"] is not None else abs(row["_net"])) - row["_taxes"]
            continue
        if row["_side"] == "buy":
            lots[key].append({"quantity": row["_quantity"], "gross": row["_gross"],
                              "fees": row["_fees"], "taxes": row["_taxes"], "row": row})
            continue
        sale_remaining = row["_quantity"]
        sale_pools = {"gross": row["_gross"], "fees": row["_fees"], "taxes": row["_taxes"]}
        while sale_remaining > ZERO and lots[key]:
            lot = lots[key][0]
            quantity = min(lot["quantity"], sale_remaining)
            buy_parts = {field: _allocate(lot[field], quantity, lot["quantity"]) for field in sale_pools}
            sell_parts = {field: _allocate(sale_pools[field], quantity, sale_remaining) for field in sale_pools}
            cost = buy_parts["gross"] + buy_parts["fees"] + buy_parts["taxes"]
            proceeds = sell_parts["gross"] - sell_parts["fees"] - sell_parts["taxes"]
            if within_range:
                buy_source = lot["row"]["_source"]
                realization = {**_scope_dict(key), "buy_transaction_id": lot["row"].get("id"),
                               "sell_transaction_id": row.get("id"), "quantity_closed": quantity,
                               "buy_date": lot["row"]["_day"], "sell_date": row["_day"],
                               "gross_proceeds": sell_parts["gross"], "proceeds": proceeds,
                               "cost_basis": cost, "realized_pnl": proceeds - cost,
                               "buy_fees_allocated": buy_parts["fees"], "sell_fees_allocated": sell_parts["fees"],
                               "fees_allocated": buy_parts["fees"] + sell_parts["fees"],
                               "taxes_allocated": buy_parts["taxes"] + sell_parts["taxes"],
                               "holding_period_days": (row["_day"] - lot["row"]["_day"]).days,
                               "buy_source": buy_source, "sell_source": source,
                               "sources": [buy_source, source]}
                realized.append(realization)
                summary["realized_pnl"] += realization["realized_pnl"]
                summary["sources"].extend([buy_source, source])
                sources.extend([buy_source, source])
            for field in sale_pools:
                lot[field] -= buy_parts[field]
                sale_pools[field] -= sell_parts[field]
            lot["quantity"] -= quantity
            sale_remaining -= quantity
            if lot["quantity"] == ZERO:
                lots[key].popleft()
        if sale_remaining > ZERO:
            incomplete_scopes.add(key)
            warnings.append(f"{row['_label']}: missing purchase lots for {_json_safe(sale_remaining)} {key[0]} "
                            f"in account {key[2] or '(unidentified)'}. Unmatched sale proceeds are excluded "
                            "from realized P&L; positions and P&L for this scope are incomplete.")
            sources.append(source)
            excluded.append({"source": source, "reason": "Missing purchase lots", "quantity": sale_remaining,
                             "excluded_gross_proceeds": sale_pools["gross"]})

    positions = []
    for key, queue in lots.items():
        if not queue:
            continue
        scope = _scope_dict(key)
        quantity = sum((lot["quantity"] for lot in queue), ZERO)
        cost_basis = sum((lot["gross"] + lot["fees"] + lot["taxes"] for lot in queue), ZERO)
        position_sources = _dedupe_sources([lot["row"]["_source"] for lot in queue])
        quote, reason = _matching_price(prices, scope, valuation_date)
        if key in incomplete_scopes:
            quote, reason = None, "Position is incomplete because purchase lots are missing."
        if reason:
            warnings.append(f"{key[0]} ({key[1]}, account {key[2] or '(unidentified)'}): {reason}")
        position = {**scope, "quantity": quantity, "cost_basis": cost_basis,
                    "average_cost": cost_basis / quantity, "current_price": None, "price_as_of": None,
                    "market_value": None, "unrealized_pnl": None, "as_of": valuation_date,
                    "is_complete": key not in incomplete_scopes, "sources": position_sources,
                    "open_lots": [{"transaction_id": lot["row"].get("id"),
                                   "quantity_remaining": lot["quantity"],
                                   "cost_basis": lot["gross"] + lot["fees"] + lot["taxes"],
                                   "source": lot["row"]["_source"]} for lot in queue]}
        if quote:
            position.update(quote)
            position["market_value"] = quantity * quote["current_price"]
            position["unrealized_pnl"] = position["market_value"] - cost_basis
        summary = summaries[key]
        if position["unrealized_pnl"] is None:
            summary["unrealized_pnl"] = None
            summary["unrealized_complete"] = False
        elif summary["unrealized_pnl"] is not None:
            summary["unrealized_pnl"] += position["unrealized_pnl"]
        summary["sources"].extend(position_sources)
        sources.extend(position_sources)
        positions.append(position)

    for key in blocked:
        summary = summaries.setdefault(key, _new_summary(key))
        summary.update(realized_pnl=None, unrealized_pnl=None, realized_complete=False, unrealized_complete=False)
    totals = {}
    metric_fields = ("realized_pnl", "fees", "taxes", "dividends", "interest", "deposits", "withdrawals", "unrealized_pnl")
    for key, summary in summaries.items():
        if key in incomplete_scopes:
            summary["realized_complete"] = False
        summary["is_complete"] = summary["realized_complete"] and summary["unrealized_complete"]
        summary["sources"] = _dedupe_sources(summary["sources"])
        currency = summary["currency"]
        total = totals.setdefault(currency, {"currency": currency, **{field: ZERO for field in metric_fields},
                                              "realized_complete": True, "unrealized_complete": True})
        for field in metric_fields:
            if field == "unrealized_pnl" and summary[field] is None:
                total[field] = None
            elif total[field] is not None and summary[field] is not None:
                total[field] += summary[field]
        total["realized_complete"] &= summary["realized_complete"]
        total["unrealized_complete"] &= summary["unrealized_complete"]
    # Zero cannot be presented as a measured profit when every row was blocked.
    for currency, total in totals.items():
        if all(summary["realized_pnl"] is None for summary in summaries.values() if summary["currency"] == currency):
            total["realized_pnl"] = None
    warnings = list(dict.fromkeys(warnings))
    if not selected and not excluded:
        warnings.append("No imported transactions match the selected filters.")
    realized_complete = not incomplete_scopes
    unrealized_complete = all(total["unrealized_complete"] for total in totals.values())
    return _json_safe({"method": "FIFO", "filters": filters, "as_of": valuation_date,
                       "totals": [totals[key] for key in sorted(totals)],
                       "by_symbol": [summaries[key] for key in sorted(summaries)],
                       "positions": sorted(positions, key=lambda item: tuple(item[field] for field in _SCOPE_FIELDS)),
                       "realized_lots": realized, "sources": _dedupe_sources(sources),
                       "warnings": warnings, "excluded_transactions": excluded,
                       "realized_complete": realized_complete, "unrealized_complete": unrealized_complete,
                       "needs_user_action": not realized_complete or not unrealized_complete or not selected,
                       "metric_definitions": {
                           "realized_pnl": "Matched FIFO sale proceeds less buy cost, with trade fees and taxes allocated once. Unmatched sales are excluded.",
                           "fees": "Fees incurred in the selected date range; allocated trade fees are already included in realized P&L and must not be deducted again.",
                           "dividends": "Net dividend cash after explicitly recorded fees and taxes; separate from trading P&L.",
                           "unrealized_pnl": "Dated price times remaining quantity less remaining cost basis; null if any open position lacks a valid price.",
                           "positions": "Positions use all available history through date_to, including purchases before date_from.",
                       }})
