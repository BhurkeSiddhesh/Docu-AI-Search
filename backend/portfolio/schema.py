"""Independent local ledger; no document-index tables are changed."""
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3


def database_path():
    return Path(os.environ.get("DOCU_PORTFOLIO_DB", Path(__file__).resolve().parents[2] / "data" / "portfolio.db"))


def storage_path():
    return database_path().parent / "portfolio"


SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_sources (
 id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, filename TEXT NOT NULL,
 file_type TEXT NOT NULL, content_hash TEXT UNIQUE, source_kind TEXT NOT NULL,
 broker_guess TEXT, document_type TEXT NOT NULL DEFAULT 'unsupported',
 import_status TEXT NOT NULL DEFAULT 'discovered', defaults_json TEXT NOT NULL DEFAULT '{}',
 warnings_json TEXT NOT NULL DEFAULT '[]', errors_json TEXT NOT NULL DEFAULT '[]',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS portfolio_tables (
 id TEXT PRIMARY KEY, source_file_id TEXT NOT NULL REFERENCES portfolio_sources(id) ON DELETE CASCADE,
 sheet_name TEXT, table_index INTEGER NOT NULL, header_row_index INTEGER,
 data_start_row_index INTEGER, row_count INTEGER NOT NULL, column_count INTEGER NOT NULL,
 raw_headers_json TEXT NOT NULL, detected_type TEXT NOT NULL, confidence REAL NOT NULL,
 table_json TEXT NOT NULL, mapping_json TEXT NOT NULL DEFAULT '{}',
 confirmed_by_user INTEGER NOT NULL DEFAULT 0, ignored INTEGER NOT NULL DEFAULT 0,
 UNIQUE(source_file_id, table_index)
);
CREATE TABLE IF NOT EXISTS portfolio_mappings (
 id INTEGER PRIMARY KEY, source_table_id TEXT NOT NULL REFERENCES portfolio_tables(id) ON DELETE CASCADE,
 source_column TEXT NOT NULL, canonical_field TEXT NOT NULL, transform TEXT,
 confidence REAL NOT NULL, mapping_source TEXT NOT NULL, confirmed_by_user INTEGER NOT NULL,
 UNIQUE(source_table_id, canonical_field)
);
CREATE TABLE IF NOT EXISTS portfolio_mapping_templates (
 header_signature TEXT NOT NULL, account_scope TEXT NOT NULL, broker_scope TEXT NOT NULL,
 mapping_json TEXT NOT NULL, PRIMARY KEY(header_signature, account_scope, broker_scope)
);
CREATE TABLE IF NOT EXISTS portfolio_transactions (
 id TEXT PRIMARY KEY, source_file_id TEXT NOT NULL REFERENCES portfolio_sources(id) ON DELETE CASCADE,
 source_table_id TEXT NOT NULL REFERENCES portfolio_tables(id) ON DELETE CASCADE,
 source_row_ref TEXT NOT NULL, account_id TEXT, broker TEXT, portfolio_owner TEXT,
 transaction_type TEXT, symbol TEXT, security_name TEXT, asset_type TEXT, trade_date TEXT,
 settlement_date TEXT, side TEXT, quantity TEXT, price TEXT, gross_amount TEXT,
 fees TEXT, taxes TEXT, net_amount TEXT, currency TEXT, exchange TEXT, isin TEXT, cusip TEXT,
 notes TEXT, confidence REAL, warnings_json TEXT NOT NULL DEFAULT '[]', transaction_json TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(source_table_id, source_row_ref)
);
CREATE INDEX IF NOT EXISTS portfolio_transactions_scope
 ON portfolio_transactions(portfolio_owner, account_id, broker, currency, symbol, trade_date);
"""


@contextmanager
def connection():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
