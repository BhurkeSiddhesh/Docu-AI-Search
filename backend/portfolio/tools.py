"""Read-only, exact portfolio tools and a conservative model-independent query router.

Only the imported ledger supplies figures. A language model never computes, rounds,
or completes these answers, including when no document index or model is available.
"""

from collections import defaultdict
from datetime import date
from decimal import Decimal
import json
import re


TOOL_DESCRIPTIONS = {
    "calculate_realized_profit": "FIFO realized profit, grouped by currency.",
    "calculate_unrealized_profit": "Open-position profit; requires explicit price data.",
    "portfolio_profit_by_symbol": "Exact realized profit for each stock and currency.",
    "portfolio_top_winners": "Stocks with positive realized profit, by currency.",
    "portfolio_top_losers": "Stocks with negative realized profit, by currency.",
    "portfolio_open_positions": "Remaining quantities and cost basis.",
    "portfolio_fees": "Recorded fees and taxes, by currency.",
    "portfolio_dividends": "Recorded dividend income, by currency.",
    "portfolio_import_warnings": "Missing data and unsupported accounting events.",
}


def _service():
    from backend.portfolio import service
    return service


def _all_transactions():
    """Read every metadata page; a first-page match cannot establish ownership."""
    result = []
    while True:
        page = _service().list_transactions(limit=1000, offset=len(result))
        if isinstance(page, list):
            return page
        rows = page.get("transactions", [])
        result.extend(rows)
        if not rows or len(result) >= page.get("total", len(result)):
            return result


def _filters(filters=None, **kwargs):
    values = dict(filters or {})
    aliases = {"owner": "portfolio_owner", "account": "account_id"}
    for key, value in kwargs.items():
        if value is not None:
            values[aliases.get(key, key)] = value
    return values


def _money(value):
    return format(Decimal(str(value or "0")), "f")


def _by_symbol(report):
    grouped = defaultdict(Decimal)
    for row in report.get("by_symbol", []):
        grouped[(row.get("symbol", "Unknown"), row.get("currency", "Unknown"))] += Decimal(str(row.get("realized_pnl", "0")))
    return [{"symbol": symbol, "currency": currency, "realized_pnl": format(value, "f")}
            for (symbol, currency), value in sorted(grouped.items())]


def _citations(sources):
    grouped = defaultdict(list)
    for source in sources:
        key = (source.get("filename") or source.get("source_filename") or str(source.get("source_file_id", "Unknown file")),
               source.get("sheet_name") or source.get("source_sheet") or "")
        row = str(source.get("source_row_ref", ""))
        if row and row not in grouped[key]:
            grouped[key].append(row)
    citations = []
    for (filename, sheet), rows in list(grouped.items())[:12]:
        location = f"{filename}" + (f", sheet {sheet}" if sheet else "")
        if rows:
            location += ", rows " + ", ".join(rows[:24])
            if len(rows) > 24:
                location += f" (+{len(rows) - 24} more rows in the source ledger)"
        citations.append(location)
    if len(grouped) > 12:
        citations.append(f"{len(grouped) - 12} additional files in the source ledger")
    return "Sources: " + "; ".join(citations) + "." if citations else "No matching source rows are available."


def _result(report, metric="realized", limit=10):
    totals = report.get("totals", [])
    rows = _by_symbol(report)
    warnings = list(report.get("warnings", []))
    filters = report.get("filters", {})
    method = report.get("method", "FIFO")
    if metric in ("winners", "losers"):
        positive = metric == "winners"
        rows = [row for row in rows if (Decimal(row["realized_pnl"]) > 0 if positive else Decimal(row["realized_pnl"]) < 0)]
        ordered = []
        for currency in sorted({row["currency"] for row in rows}):
            ordered.extend(sorted([row for row in rows if row["currency"] == currency],
                                  key=lambda row: Decimal(row["realized_pnl"]), reverse=positive)[:limit])
        rows = ordered
    if metric in ("symbols", "winners", "losers"):
        text = "Realized profit by symbol: " + "; ".join(f"{row['symbol']}: {row['realized_pnl']} {row['currency']}" for row in rows) + "." if rows else "No matching closed trades" + (" with a realized gain." if metric == "winners" else " with a realized loss." if metric == "losers" else ".")
        numbers = rows
    elif metric == "positions":
        numbers = report.get("positions", [])
        text = "Open positions: " + "; ".join(f"{row.get('symbol', 'Unknown')}: {row.get('quantity', row.get('quantity_remaining', 'unknown'))} units, cost basis {row.get('cost_basis', 'unknown')} {row.get('currency', '')}" for row in numbers) + "." if numbers else "No open positions were calculated from the matching ledger."
    elif metric == "warnings":
        numbers = []
        text = "Portfolio import and calculation warnings." if warnings else "No calculation warnings were reported for the matching ledger."
    elif metric == "sources":
        numbers = []
        text = "These imported ledger rows support the current filtered portfolio calculation."
    else:
        field, label = {"realized": ("realized_pnl", "Realized profit"), "unrealized": ("unrealized_pnl", "Unrealized profit"), "fees": ("fees", "Recorded fees"), "dividends": ("dividends", "Dividend income")}[metric]
        numbers = [{"currency": row.get("currency"), field: row.get(field)} for row in totals]
        values = [f"{_money(row[field])} {row.get('currency', '')}" for row in totals if row.get(field) is not None]
        text = label + ": " + "; ".join(values) + "." if values else label + " cannot be determined from the available data."
        if metric == "fees":
            text += " Recorded taxes: " + ("; ".join(f"{_money(row.get('taxes'))} {row.get('currency', '')}" for row in totals) or "not available") + "."
        if metric == "realized":
            text += " Unrealized profit and dividend income are excluded."
    text += f" Method: {method}."
    if filters:
        text += " Filters: " + ", ".join(f"{key}={value}" for key, value in filters.items() if value is not None) + "."
    if len({row.get("currency") for row in totals}) > 1:
        text += " Currencies are reported separately; no exchange-rate conversion was applied."
    if warnings:
        text += "\n\nWarnings: " + "; ".join(str(warning) for warning in warnings)
    text += "\n\n" + _citations(report.get("sources", []))
    return {"answer_summary": text, "numbers": numbers, "currency": sorted({row.get("currency") for row in totals if row.get("currency")}),
            "method": method, "filters": filters, "sources": report.get("sources", []), "warnings": warnings,
            "needs_user_action": bool(report.get("needs_user_action"))}


def _failure(message, filters=None):
    return {"answer_summary": message, "numbers": [], "currency": [], "method": "FIFO", "filters": filters or {},
            "sources": [], "warnings": [message], "needs_user_action": True}


def _calculate(metric, filters=None, limit=10):
    report = _service().summary(filters=filters or {})
    if not report.get("sources") and not report.get("totals"):
        return _failure("No matching imported transactions are available, so profit cannot be determined. Open Portfolio, import a statement, review its mapping, and confirm the import.", filters)
    return _result(report, metric, limit)


def calculate_realized_profit(owner=None, account=None, broker=None, symbol=None, date_from=None, date_to=None, method="FIFO", filters=None):
    if method.upper() != "FIFO":
        return _failure("Only FIFO cost basis is supported. Request FIFO realized profit.")
    return _calculate("realized", _filters(filters, owner=owner, account=account, broker=broker, symbol=symbol, date_from=date_from, date_to=date_to))


def calculate_unrealized_profit(filters=None, **kwargs):
    return _calculate("unrealized", _filters(filters, **kwargs))


def portfolio_profit_by_symbol(filters=None, limit=10, **kwargs):
    return _calculate("symbols", _filters(filters, **kwargs), limit)


def portfolio_top_winners(filters=None, limit=10, **kwargs):
    return _calculate("winners", _filters(filters, **kwargs), limit)


def portfolio_top_losers(filters=None, limit=10, **kwargs):
    return _calculate("losers", _filters(filters, **kwargs), limit)


def portfolio_open_positions(filters=None, **kwargs):
    return _calculate("positions", _filters(filters, **kwargs))


def portfolio_fees(filters=None, **kwargs):
    return _calculate("fees", _filters(filters, **kwargs))


def portfolio_dividends(filters=None, **kwargs):
    return _calculate("dividends", _filters(filters, **kwargs))


def portfolio_import_warnings(filters=None, **kwargs):
    return _calculate("warnings", _filters(filters, **kwargs))


def is_portfolio_question(query):
    """Route calculation requests; ordinary financial document search stays RAG."""
    query = query.casefold().strip()
    if re.search(r"\b(explain|define|meaning|definition|summarize|find|search|document says|report says)\b", query) and not re.search(r"\b(my|our|i|portfolio)\b", query):
        return False
    return bool(re.search(r"\b(profit|profitable|pnl|p&l|open positions?|trades?.*(?:still )?open|dividend income|win rate|brokerage|portfolio.*warnings?|stock.*lost|top (?:winners|losers)|fees.*(?:pay|paid)|(?:pay|paid).*fees)\b", query)
                or re.search(r"\b(?:did i make more from|what files contributed to this answer|which rows support)\b", query))


def _whole_pattern(value):
    return r"(?<![\w])" + re.escape(value) + r"(?![\w])"


def _parse(query, transactions):
    original = query
    remaining = query.casefold().replace("’", "'")
    filters = {}
    if re.search(r"\b(win rate|total return|percentage|percent|roi|annualized|average|median|tax filing|wash sale|lifo|convert|converted|conversion|exchange rate|fx|benchmark|versus the market)\b|%", remaining):
        return None, None, "That metric or currency conversion is not supported. Ask for FIFO realized profit, profit by symbol, open positions, fees, or dividends with an owner, account, broker, symbol, currency, or ISO date range."
    metric = "realized"
    if "unrealized" in remaining:
        metric = "unrealized"
    elif re.search(r"\b(?:most (?:profit|profitable)|top winners?|largest (?:profit|gain)|best stock)\b", remaining):
        metric = "winners"
    elif re.search(r"\b(?:lost the most|top losers?|largest loss|worst stock)\b", remaining):
        metric = "losers"
    elif re.search(r"\b(?:by symbol|by stock|make more from)\b", remaining):
        metric = "symbols"
    elif re.search(r"\bopen\b", remaining):
        metric = "positions"
    elif re.search(r"\b(fees?|brokerage)\b", remaining):
        metric = "fees"
    elif re.search(r"\bdividends?\b", remaining):
        metric = "dividends"
    elif re.search(r"\bwarnings?\b", remaining):
        metric = "warnings"
    elif re.search(r"\b(rows|files)\b", remaining):
        metric = "sources"

    # Date filters select sales/income in the interval; the calculator retains
    # earlier buy lots when matching their cost basis.
    if "this year" in remaining:
        today = date.today()
        filters.update(date_from=f"{today.year}-01-01", date_to=today.isoformat())
        remaining = remaining.replace("this year", " ")
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", remaining)
    if dates:
        try:
            for value in dates:
                date.fromisoformat(value)
        except ValueError:
            return None, None, "Use valid ISO dates (YYYY-MM-DD)."
        if len(dates) == 2 and re.search(r"\b(?:from|between)\s+" + dates[0] + r"\s+(?:to|through|and)\s+" + dates[1], remaining):
            filters.update(date_from=dates[0], date_to=dates[1])
        elif len(dates) == 1 and re.search(r"\b(?:since|from)\s+" + dates[0], remaining):
            filters["date_from"] = dates[0]
        elif len(dates) == 1 and re.search(r"\b(?:until|through|before)\s+" + dates[0], remaining):
            if "before " + dates[0] in remaining:
                return None, None, "Use an inclusive date range: from YYYY-MM-DD to YYYY-MM-DD."
            filters["date_to"] = dates[0]
        else:
            return None, None, "Specify an inclusive date range: from YYYY-MM-DD to YYYY-MM-DD, or since YYYY-MM-DD."
        for value in dates:
            remaining = remaining.replace(value, " ")
        if filters.get("date_from", "") > filters.get("date_to", "9999"):
            return None, None, "The start date must be on or before the end date."
    years = re.findall(r"\b(?:19|20)\d{2}\b", remaining)
    if years:
        if len(years) != 1 or filters:
            return None, None, "Use one year or an explicit ISO date range."
        filters.update(date_from=f"{years[0]}-01-01", date_to=f"{years[0]}-12-31")
        remaining = remaining.replace(years[0], " ")

    owners = sorted({str(row.get("portfolio_owner", "")).strip() for row in transactions if row.get("portfolio_owner")})
    symbols = sorted({str(row.get("symbol", "")).strip() for row in transactions if row.get("symbol")})
    metadata = {"portfolio_owner": owners, "account_id": sorted({str(row["account_id"]) for row in transactions if row.get("account_id")}),
                "broker": sorted({str(row["broker"]) for row in transactions if row.get("broker")}),
                "currency": sorted({str(row["currency"]) for row in transactions if row.get("currency")})}
    # Longer metadata values win over their prefixes (e.g. two owner names).
    for field, values in metadata.items():
        found = []
        for value in sorted(values, key=len, reverse=True):
            if re.search(_whole_pattern(value.casefold()), remaining):
                found.append(value)
                remaining = re.sub(_whole_pattern(value.casefold()), " ", remaining)
        if len(found) > 1:
            return None, None, f"Specify one {field.replace('_', ' ')} at a time."
        if found:
            filters[field] = found[0]

    owner_phrase = re.search(r"\b(?:did|has)\s+(.+?)\s+(?:earn|make|earned|made)\b", remaining)
    if owner_phrase:
        name = owner_phrase.group(1).strip()
        if name not in ("i", "we", " ", ""):
            candidates = [owner for owner in owners if owner.casefold().split()[0] == name]
            if len(candidates) != 1:
                return None, None, f"I cannot identify a unique imported portfolio owner matching '{name}'. Confirm the owner during Portfolio import or specify its exact recorded name."
            filters["portfolio_owner"] = candidates[0]
            remaining = remaining[:owner_phrase.start(1)] + " " + remaining[owner_phrase.end(1):]
    if re.search(r"\b(i|my|we|our)\b", remaining) and not filters.get("portfolio_owner"):
        if len(owners) > 1:
            return None, None, "Multiple portfolio owners are imported. Specify the owner: " + ", ".join(owners) + "."
        if len(owners) == 1:
            if any(not row.get("portfolio_owner") for row in transactions):
                return None, None, "Some transactions have no portfolio owner. Specify the owner's recorded name or complete ownership metadata in Portfolio before asking for 'my' profit."
            filters["portfolio_owner"] = owners[0]

    found_symbols = []
    for symbol in sorted(symbols, key=len, reverse=True):
        # A ticker such as I or IN must not match ordinary lower-case words.
        if (len(symbol) > 2 and re.search(_whole_pattern(symbol.casefold()), remaining)) or re.search(_whole_pattern(symbol), original):
            found_symbols.append(symbol)
            remaining = re.sub(_whole_pattern(symbol.casefold()), " ", remaining)
    names = defaultdict(set)
    for row in transactions:
        if row.get("security_name") and row.get("symbol"):
            name = str(row["security_name"]).casefold().strip()
            names[name].add(row["symbol"])
            # A unique issuer's first word permits "Apple trades" only when
            # the imported security metadata actually names Apple.
            first = name.split()[0]
            if len(first) >= 4:
                names[first].add(row["symbol"])
    for name in sorted(names, key=len, reverse=True):
        if re.search(_whole_pattern(name), remaining):
            if len(names[name]) != 1:
                return None, None, f"'{name}' matches multiple imported securities. Specify the exact symbol."
            found_symbols.extend(names[name])
            remaining = re.sub(_whole_pattern(name), " ", remaining)
    found_symbols = list(dict.fromkeys(found_symbols))
    if len(found_symbols) == 1:
        filters["symbol"] = found_symbols[0]
    elif len(found_symbols) > 1:
        filters["symbols"] = found_symbols
        metric = "symbols"

    allowed = "how much profit profits profitable realized unrealized did has have i we my our make made earn earned so far total the which what stock stocks was is are most lost money show by symbol symbols trades trade came come from more or and a an in on for of to between since until through with account broker owner portfolio currency open positions position still fees fee pay paid brokerage dividend dividends income receive received top winners winner losers loser largest loss gain best worst number rows support files contributed this answer warnings warning import recorded fifo please calculate tell me all across using only at".split()
    leftovers = re.sub(r"[^\w\s]", " ", remaining)
    leftovers = re.sub(r"\b(?:" + "|".join(allowed) + r")\b", " ", leftovers)
    if leftovers.strip():
        return None, None, "I could not safely resolve this filter or metric: '" + " ".join(leftovers.split()) + "'. Use an exact imported owner, account, broker, symbol, or currency and an ISO date range. No unfiltered profit was calculated."
    return metric, filters, None


def answer_query(query):
    """Return a structured deterministic answer, or None for ordinary RAG."""
    if not is_portfolio_question(query):
        return None
    transactions = _all_transactions()
    if not transactions:
        return _failure("No portfolio transactions have been imported, so profit cannot be determined. Open Portfolio, import a CSV/XLSX or supported statement, review its column mapping, and confirm the import.")
    metric, filters, error = _parse(query, transactions)
    if error:
        return _failure(error)
    if metric == "sources" and re.search(r"\b(this answer|this profit)\b", query.casefold()):
        return _failure("To identify supporting rows, repeat the profit question with the same owner, symbol, and date filters. This chat does not retain a previous calculation's filters.")
    return _calculate(metric, filters, limit=1 if metric in ("winners", "losers") and "top" not in query.casefold() else 10)


def run_tool(name, action_input=""):
    """ReAct adapter: strict JSON inputs, never eval or silently drop filters."""
    if name not in TOOL_DESCRIPTIONS:
        return _failure("Unknown portfolio tool.")
    try:
        arguments = json.loads(action_input) if action_input and action_input.strip() not in ("None", "none") else {}
        if not isinstance(arguments, dict):
            raise ValueError("Expected a JSON object")
        allowed = {"owner", "portfolio_owner", "account", "account_id", "broker", "symbol", "symbols", "date_from", "date_to", "currency", "limit", "method"}
        if set(arguments) - allowed:
            return _failure("Unsupported portfolio filter. Use owner, account, broker, symbol, currency, or an ISO date range.")
        limit = int(arguments.pop("limit", 10))
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        method = arguments.pop("method", "FIFO")
        if method != "FIFO":
            return _failure("Only FIFO cost basis is supported.")
        metric = {"calculate_realized_profit": "realized", "calculate_unrealized_profit": "unrealized", "portfolio_profit_by_symbol": "symbols", "portfolio_top_winners": "winners", "portfolio_top_losers": "losers", "portfolio_open_positions": "positions", "portfolio_fees": "fees", "portfolio_dividends": "dividends", "portfolio_import_warnings": "warnings"}[name]
        return _calculate(metric, _filters(**arguments), limit)
    except (ValueError, TypeError):
        return _failure("Portfolio tools require a JSON object containing supported filters and an integer limit from 1 to 100.")
