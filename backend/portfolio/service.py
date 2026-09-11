"""Local, review-before-import portfolio workflow and deterministic analytics."""
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import uuid

from . import schema

logger = logging.getLogger(__name__)
MAX_FILE_BYTES = 25 * 1024 * 1024
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".pdf", ".txt", ".docx"}
_lock = threading.RLock()


class PortfolioError(ValueError):
    def __init__(self, message, status_code=400, errors=None):
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def _source(row):
    result = dict(row)
    # Full filesystem paths are internal; citations use filenames and physical rows.
    result.pop("path", None)
    for key in ("defaults", "warnings", "errors"):
        result[key] = json.loads(result.pop(key + "_json"))
    return result


def _get(conn, source_id):
    row = conn.execute("SELECT * FROM portfolio_sources WHERE id=?", (str(source_id),)).fetchone()
    if row is None:
        raise PortfolioError("Portfolio source was not found.", 404)
    return row


def _canonical_path(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _indexed_paths():
    from backend.database import get_all_files
    result = set()
    offset = 0
    while True:
        rows = get_all_files(limit=1000, offset=offset)
        result.update(_canonical_path(row["path"]) for row in rows if row.get("path"))
        if len(rows) < 1000:
            return result
        offset += len(rows)


def _read_bytes(path):
    try:
        with open(path, "rb") as handle:
            content = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        logger.warning("Portfolio source could not be read: %s", exc)
        raise PortfolioError("The source file is unavailable. Re-index it or upload it again.", 409) from exc
    if len(content) > MAX_FILE_BYTES:
        raise PortfolioError("Portfolio files must be 25 MB or smaller.", 413)
    if not content:
        raise PortfolioError("The source file is empty.", 422)
    return content


def _authorized_bytes(row):
    path = _canonical_path(row["path"])
    if row["source_kind"] == "indexed":
        if path not in _indexed_paths():
            raise PortfolioError("This file is no longer in the indexed library. Upload it or index it again.", 403)
    else:
        root = _canonical_path(schema.storage_path() / "uploads")
        if not path.startswith(root + os.sep):
            raise PortfolioError("Invalid stored portfolio source.", 403)
    return _read_bytes(path)


def _register(path, filename, content, kind):
    digest = hashlib.sha256(content).hexdigest()
    path = _canonical_path(path)
    with schema.connection() as conn:
        current = conn.execute("SELECT * FROM portfolio_sources WHERE path=?", (path,)).fetchone()
        duplicate = conn.execute("SELECT * FROM portfolio_sources WHERE content_hash=?", (digest,)).fetchone()
        if duplicate is not None and (current is None or current["id"] != duplicate["id"]):
            if current is not None:
                conn.execute("DELETE FROM portfolio_tables WHERE source_file_id=?", (current["id"],))
                conn.execute("UPDATE portfolio_sources SET content_hash=NULL, import_status='ignored', warnings_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                             (_json(["File changed to a duplicate of an existing source; old ledger rows were removed."]), current["id"]))
            # Keep provenance stable when a byte-identical renamed copy is discovered.
            if not os.path.isfile(duplicate["path"]):
                conn.execute("UPDATE portfolio_sources SET path=?,filename=?,source_kind=? WHERE id=?", (path, filename, kind, duplicate["id"]))
            return _source(_get(conn, duplicate["id"]))
        if current is not None:
            if current["content_hash"] != digest:
                conn.execute("DELETE FROM portfolio_tables WHERE source_file_id=?", (current["id"],))
                conn.execute("UPDATE portfolio_sources SET content_hash=?, import_status='discovered', warnings_json=?, errors_json='[]', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                             (digest, _json(["Source bytes changed. Previous transactions and mapping approval were removed; review this version before importing."]), current["id"]))
            return _source(_get(conn, current["id"]))
        source_id = uuid.uuid4().hex
        conn.execute("INSERT INTO portfolio_sources(id,path,filename,file_type,content_hash,source_kind) VALUES(?,?,?,?,?,?)",
                     (source_id, path, filename, Path(filename).suffix.lower().lstrip("."), digest, kind))
        return _source(_get(conn, source_id))


def upload_source(filename, content):
    """Persist bytes under a generated name; the submitted name never becomes a path."""
    filename = str(filename or "upload").replace("\\", "/").split("/")[-1][:255]
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise PortfolioError("Supported portfolio files: CSV, XLSX, PDF, TXT and DOCX.", 415)
    if len(content) > MAX_FILE_BYTES:
        raise PortfolioError("Portfolio files must be 25 MB or smaller.", 413)
    if not content:
        raise PortfolioError("The uploaded file is empty.", 422)
    with _lock:
        digest = hashlib.sha256(content).hexdigest()
        with schema.connection() as conn:
            existing = conn.execute("SELECT * FROM portfolio_sources WHERE content_hash=?", (digest,)).fetchone()
            if existing:
                return _source(existing)
        directory = schema.storage_path() / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (uuid.uuid4().hex + suffix)
        path.write_bytes(content)
        try:
            return _register(path, filename, content, "upload")
        except Exception:
            path.unlink(missing_ok=True)
            raise


def list_sources():
    with schema.connection() as conn:
        rows = conn.execute("SELECT s.*, (SELECT COUNT(*) FROM portfolio_transactions t WHERE t.source_file_id=s.id) AS transaction_count FROM portfolio_sources s ORDER BY created_at DESC,id")
        return [_source(row) for row in rows]


def discover_sources(paths=None):
    """Discover only paths already authorized by the document index."""
    with _lock:
        indexed = _indexed_paths()
        selected = indexed if paths is None else {_canonical_path(p.get("path") if isinstance(p, dict) else p) for p in paths}
        if not selected.issubset(indexed):
            raise PortfolioError("Portfolio discovery only accepts files in the indexed library.", 403)
        sources = {}
        for path in sorted(selected):
            if Path(path).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                content = _read_bytes(path)
                source = _register(path, Path(path).name, content, "indexed")
                sources[source["id"]] = source
            except PortfolioError as exc:
                logger.warning("Skipping inaccessible portfolio candidate: %s", exc)
        return list(sources.values())


def _signature(headers):
    return hashlib.sha256(_json(headers).encode("utf-8")).hexdigest()


def _scope(table, mapping, defaults):
    values = []
    for field in ("account_id", "broker"):
        if defaults.get(field):
            values.append(str(defaults[field]).strip())
            continue
        column = mapping.get(field)
        distinct = {str(r.get("values", {}).get(column, "")).strip() for r in table.get("rows", [])}
        distinct.discard("")
        values.append(next(iter(distinct)) if len(distinct) == 1 else "")
    return tuple(values)


def _refresh_and_parse(source_id):
    from .importers import parse_file
    with schema.connection() as conn:
        row = _get(conn, source_id)
        content = _authorized_bytes(row)
    refreshed = _register(row["path"], row["filename"], content, row["source_kind"])
    if refreshed["id"] != source_id:
        raise PortfolioError("This file is now a duplicate of another source. Open the existing source instead.", 409)
    with schema.connection() as conn:
        row = _get(conn, source_id)
        if conn.execute("SELECT 1 FROM portfolio_tables WHERE source_file_id=?", (source_id,)).fetchone():
            return
    # Parse a private immutable snapshot so the source cannot change midway through parsing.
    cache = schema.storage_path() / "parsing"
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = cache / (uuid.uuid4().hex + "." + row["file_type"])
    snapshot.write_bytes(content)
    try:
        tables = parse_file(str(snapshot))
        if isinstance(tables, dict):
            tables = tables.get("tables", [])
        if not tables:
            raise PortfolioError("No structured financial tables were found. Export a CSV or XLSX statement.", 422)
    except Exception as exc:
        message = str(exc) if isinstance(exc, (ValueError, PortfolioError)) else "This file could not be parsed. Try a CSV or XLSX broker export."
        with schema.connection() as conn:
            conn.execute("UPDATE portfolio_sources SET import_status='failed',errors_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (_json([{"message": message}]), source_id))
        if isinstance(exc, PortfolioError):
            raise
        logger.warning("Portfolio parsing failed", exc_info=True)
        raise PortfolioError(message, 422) from exc
    finally:
        snapshot.unlink(missing_ok=True)
    with schema.connection() as conn:
        source = _get(conn, source_id)
        defaults = json.loads(source["defaults_json"])
        warnings = json.loads(source["warnings_json"])
        for index, table in enumerate(tables):
            headers = table.get("headers", [])
            mapping = table.get("mapping", {})
            account, broker = _scope(table, mapping, defaults)
            reused = False
            if account and broker:
                template = conn.execute("SELECT mapping_json FROM portfolio_mapping_templates WHERE header_signature=? AND account_scope=? AND broker_scope=?", (_signature(headers), account, broker)).fetchone()
                if template:
                    mapping = json.loads(template["mapping_json"])
                    reused = True
            table["mapping_reused"] = reused
            table_id = uuid.uuid4().hex
            conn.execute("INSERT INTO portfolio_tables(id,source_file_id,sheet_name,table_index,header_row_index,data_start_row_index,row_count,column_count,raw_headers_json,detected_type,confidence,table_json,mapping_json,confirmed_by_user) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (table_id, source_id, table.get("sheet_name"), index, table.get("header_row_index", 1), table.get("data_start_row_index", 2), len(table.get("rows", [])), len(headers), _json(headers), table.get("detected_type", "unsupported"), float(table.get("confidence", 0)), _json(table), _json(mapping), int(reused)))
            warnings.extend(table.get("warnings", []))
        kinds = sorted({t.get("detected_type", "unsupported") for t in tables})
        conn.execute("UPDATE portfolio_sources SET import_status='needs_mapping_review',document_type=?,warnings_json=?,errors_json='[]',updated_at=CURRENT_TIMESTAMP WHERE id=?", (",".join(kinds), _json(warnings), source_id))


def _preview(conn, source_id):
    from .mapping import CANONICAL_FIELDS
    source = _source(_get(conn, source_id))
    tables = []
    for row in conn.execute("SELECT * FROM portfolio_tables WHERE source_file_id=? ORDER BY table_index", (source_id,)):
        table = json.loads(row["table_json"])
        table.update(id=row["id"], row_count=row["row_count"], column_count=row["column_count"], mapping=json.loads(row["mapping_json"]), confirmed=bool(row["confirmed_by_user"]), ignored=bool(row["ignored"]))
        table["rows"] = table.get("rows", [])[:25]
        tables.append(table)
    return {"source": source, "tables": tables, "canonical_fields": list(CANONICAL_FIELDS), "defaults": source["defaults"]}


def preview_source(source_id):
    with _lock:
        _refresh_and_parse(str(source_id))
        with schema.connection() as conn:
            return _preview(conn, str(source_id))


def confirm_mapping(source_id, payload):
    from .mapping import CANONICAL_FIELDS
    from .models import MappingConfirmation
    from pydantic import ValidationError
    try:
        payload = MappingConfirmation.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise PortfolioError("Invalid mapping confirmation.", 422, json.loads(exc.json())) from exc
    defaults = {k: str(v).strip() for k, v in payload["defaults"].items() if str(v).strip()}
    allowed_defaults = {"portfolio_owner", "account_id", "broker", "currency", "date_order"}
    if set(defaults) - allowed_defaults or any(len(v) > 200 for v in defaults.values()):
        raise PortfolioError("Invalid portfolio defaults.", 422)
    if defaults.get("date_order") not in (None, "DMY", "MDY", "YMD"):
        raise PortfolioError("Date order must be DMY, MDY or YMD.", 422)
    if defaults.get("currency") and (len(defaults["currency"]) != 3 or not defaults["currency"].isalpha()):
        raise PortfolioError("Currency must be a three-letter code.", 422)
    if "currency" in defaults:
        defaults["currency"] = defaults["currency"].upper()
    with _lock:
        _refresh_and_parse(str(source_id))
        with schema.connection() as conn:
            rows = {r["id"]: r for r in conn.execute("SELECT * FROM portfolio_tables WHERE source_file_id=?", (source_id,))}
            ids = [t["table_id"] for t in payload["tables"]]
            if len(ids) != len(set(ids)) or set(ids) != set(rows):
                raise PortfolioError("Review every table in this source exactly once, including ignored tables.", 422)
            for item in payload["tables"]:
                row = rows[item["table_id"]]
                headers = json.loads(row["raw_headers_json"])
                mapping = {k: v for k, v in item["mapping"].items() if v}
                if set(mapping) - set(CANONICAL_FIELDS) or any(v not in headers for v in mapping.values()):
                    raise PortfolioError("Mappings must use canonical fields and headers from this table.", 422)
                if not item["ignored"] and not mapping:
                    raise PortfolioError("Map the table columns or explicitly ignore the table.", 422)
            # A revised mapping supersedes the previous trusted ledger immediately.
            conn.execute("DELETE FROM portfolio_transactions WHERE source_file_id=?", (source_id,))
            for item in payload["tables"]:
                row = rows[item["table_id"]]
                mapping = {k: v for k, v in item["mapping"].items() if v}
                conn.execute("UPDATE portfolio_tables SET mapping_json=?,confirmed_by_user=1,ignored=? WHERE id=?", (_json(mapping), int(item["ignored"]), row["id"]))
                conn.execute("DELETE FROM portfolio_mappings WHERE source_table_id=?", (row["id"],))
                for canonical, header in mapping.items():
                    conn.execute("INSERT INTO portfolio_mappings(source_table_id,source_column,canonical_field,confidence,mapping_source,confirmed_by_user) VALUES(?,?,?,1.0,'user_confirmed',1)", (row["id"], header, canonical))
                table = json.loads(row["table_json"])
                account, broker = _scope(table, mapping, defaults)
                if not item["ignored"] and account and broker:
                    conn.execute("INSERT OR REPLACE INTO portfolio_mapping_templates VALUES(?,?,?,?)", (_signature(table["headers"]), account, broker, _json(mapping)))
            conn.execute("UPDATE portfolio_sources SET defaults_json=?,import_status='parsed',errors_json='[]',updated_at=CURRENT_TIMESTAMP WHERE id=?", (_json(defaults), source_id))
            return _preview(conn, source_id)


def import_source(source_id):
    from .normalizer import normalize_rows
    with _lock:
        _refresh_and_parse(str(source_id))
        with schema.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source = _get(conn, source_id)
            tables = list(conn.execute("SELECT * FROM portfolio_tables WHERE source_file_id=? ORDER BY table_index", (source_id,)))
            if source["import_status"] == "ignored":
                raise PortfolioError("Review the ignored source before importing it.", 409)
            if not tables or any(not t["confirmed_by_user"] for t in tables):
                raise PortfolioError("Confirm all table mappings before importing this source.", 409)
            defaults = json.loads(source["defaults_json"])
            warnings = json.loads(source["warnings_json"])
            errors, pending = [], []
            for row in tables:
                if row["ignored"]:
                    continue
                table = json.loads(row["table_json"])
                result = normalize_rows(table, json.loads(row["mapping_json"]), defaults)
                for key, destination in (("warnings", warnings), ("errors", errors)):
                    for issue in result.get(key, []):
                        destination.append(dict(issue, source_table_id=row["id"], source_file_id=source_id) if isinstance(issue, dict) else {"message": str(issue), "source_table_id": row["id"], "source_file_id": source_id})
                for transaction in result.get("transactions", []):
                    transaction = dict(transaction)
                    ref = str(transaction.get("source_row_ref", transaction.get("row_ref", "")))
                    if not ref:
                        errors.append({"message": "A normalized row is missing source lineage.", "source_table_id": row["id"]})
                        continue
                    transaction.update(id=uuid.uuid5(uuid.NAMESPACE_URL, row["id"] + ":" + ref).hex, source_file_id=source_id, source_table_id=row["id"], source_row_ref=ref, source_filename=source["filename"], source_sheet=row["sheet_name"], sheet_name=row["sheet_name"], source_content_hash=source["content_hash"])
                    pending.append(transaction)
            if not pending and not errors:
                errors.append({"message": "No supported transactions remain. Review table mappings and ignored tables."})
            if errors:
                # No ledger changes occur on a normalization failure; the report is persisted separately.
                conn.execute("UPDATE portfolio_sources SET import_status='failed',errors_json=?,warnings_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (_json(errors), _json(warnings), source_id))
                return {"source": _source(_get(conn, source_id)), "imported_count": 0, "warnings": warnings, "errors": errors}
            conn.execute("DELETE FROM portfolio_transactions WHERE source_file_id=?", (source_id,))
            columns = ["id", "source_file_id", "source_table_id", "source_row_ref", "account_id", "broker", "portfolio_owner", "transaction_type", "symbol", "security_name", "asset_type", "trade_date", "settlement_date", "side", "quantity", "price", "gross_amount", "fees", "taxes", "net_amount", "currency", "exchange", "isin", "cusip", "notes", "confidence"]
            sql = "INSERT INTO portfolio_transactions(" + ",".join(columns) + ",warnings_json,transaction_json) VALUES(" + ",".join("?" for _ in range(len(columns) + 2)) + ")"
            for transaction in pending:
                conn.execute(sql, [transaction.get(k) for k in columns] + [_json(transaction.get("warnings", [])), _json(transaction)])
            status = "imported_with_warnings" if warnings else "imported"
            conn.execute("UPDATE portfolio_sources SET import_status=?,warnings_json=?,errors_json='[]',updated_at=CURRENT_TIMESTAMP WHERE id=?", (status, _json(warnings), source_id))
            return {"source": _source(_get(conn, source_id)), "imported_count": len(pending), "warnings": warnings, "errors": []}


def _transactions():
    with schema.connection() as conn:
        return [json.loads(row[0]) for row in conn.execute("SELECT transaction_json FROM portfolio_transactions ORDER BY trade_date,source_file_id,source_table_id,source_row_ref")]


def list_transactions(filters=None, limit=100, offset=0):
    filters = filters or {}
    rows = _transactions()
    for key in ("portfolio_owner", "account_id", "broker", "symbol", "currency", "source_file_id"):
        value = filters.get(key) or filters.get({"portfolio_owner": "owner", "account_id": "account"}.get(key, key))
        if value:
            rows = [t for t in rows if str(t.get(key, "")).casefold() == str(value).casefold()]
    for key, lower in (("date_from", True), ("date_to", False)):
        if filters.get(key):
            rows = [t for t in rows if (t.get("trade_date", "") >= filters[key] if lower else t.get("trade_date", "") <= filters[key])]
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    return {"transactions": rows[offset:offset + limit], "total": len(rows), "limit": limit, "offset": offset}


def summary(filters=None, prices=None):
    from .calculations import calculate_portfolio
    result = calculate_portfolio(_transactions(), filters=filters, prices=prices)
    pending = [s for s in list_sources() if s["import_status"] not in ("imported", "imported_with_warnings", "ignored")]
    if pending:
        result.setdefault("warnings", []).append({"code": "sources_not_imported", "message": f"{len(pending)} source(s) still need review or import and are excluded from totals.", "source_file_ids": [s["id"] for s in pending]})
        result["needs_user_action"] = True
    return result


def warnings():
    result = []
    for source in list_sources():
        for severity, items in (("warning", source["warnings"]), ("error", source["errors"])):
            for issue in items:
                entry = dict(issue) if isinstance(issue, dict) else {"message": str(issue)}
                entry.update(source_file_id=source["id"], source_filename=source["filename"], severity=severity)
                result.append(entry)
    return result


def ignore_source(source_id):
    with _lock, schema.connection() as conn:
        _get(conn, source_id)
        conn.execute("DELETE FROM portfolio_transactions WHERE source_file_id=?", (source_id,))
        conn.execute("UPDATE portfolio_tables SET confirmed_by_user=0 WHERE source_file_id=?", (source_id,))
        conn.execute("UPDATE portfolio_sources SET import_status='ignored',updated_at=CURRENT_TIMESTAMP WHERE id=?", (source_id,))
        return _source(_get(conn, source_id))


def delete_source(source_id):
    with _lock:
        with schema.connection() as conn:
            row = _get(conn, source_id)
            conn.execute("DELETE FROM portfolio_sources WHERE id=?", (source_id,))
        if row["source_kind"] == "upload":
            path = _canonical_path(row["path"])
            root = _canonical_path(schema.storage_path() / "uploads")
            if path.startswith(root + os.sep):
                Path(path).unlink(missing_ok=True)
        return {"deleted": True, "source_id": str(source_id)}
