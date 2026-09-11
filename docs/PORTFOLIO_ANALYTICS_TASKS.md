# Portfolio Analytics Roadmap

This document defines the task list for adding broker-agnostic portfolio analytics to Docu AI Search. The goal is to let a user ask questions such as "How much profit did Siddhesh earn?", "Which stock made the most profit?", or "Which trades are still open?" and get answers grounded in imported CSV, XLSX, PDF, and document exports.

The core principle is:

- Use local or cloud LLMs to understand user intent, route to tools, explain results, and help map messy files.
- Use deterministic Python code for parsing, normalization, accounting, calculations, validation, and totals.
- Keep the feature useful with small laptop-friendly models by giving them structured tools instead of asking them to perform arithmetic over raw documents.

## End State

When complete, the app should support this workflow:

1. The user indexes or imports files from one or more brokers.
2. The system discovers candidate financial files across CSV, XLSX, PDF, and text-derived formats.
3. The system classifies each file as trades, holdings, dividends, cash ledger, tax statement, or unsupported.
4. The system extracts rows or statement entries into a canonical local ledger.
5. The system maps arbitrary broker columns to canonical fields using rules, heuristics, and optional LLM assistance.
6. The user reviews low-confidence mappings before the data becomes trusted.
7. The app stores normalized transactions in SQLite with source-file and source-row lineage.
8. Deterministic tools compute realized profit, unrealized profit, open positions, profit by symbol, loss by symbol, fees, dividends, and warnings.
9. The LLM answers natural-language questions by calling these tools and explaining the computed results with citations.

Example final answer:

> Siddhesh has realized a profit of 1240.55 USD using FIFO cost basis across 3 imported broker files. The largest realized profit came from NVDA at 730.10 USD, followed by AAPL at 415.20 USD. Unrealized profit is excluded because no current price source is configured. Sources: robinhood_trades_2025.csv rows 12-30, fidelity_activity.xlsx sheet History rows 4-18.

## Non-Goals For The First Version

- Do not build tax filing or regulated financial advice.
- Do not depend on an LLM for exact arithmetic.
- Do not require cloud models.
- Do not require market-data APIs for realized P&L.
- Do not silently accept ambiguous mappings.
- Do not claim exact tax correctness for wash sales, corporate actions, or jurisdiction-specific tax rules in the MVP.

## Architecture Overview

### Existing System Role

Docu AI Search already indexes files and supports CSV/XLSX extraction as text. That is useful for search, but portfolio analytics needs structured ingestion alongside ordinary document indexing.

The new system should sit beside the existing RAG pipeline:

- Current RAG: find document snippets and answer general questions.
- New portfolio tools: parse financial files into structured tables and compute deterministic analytics.

### Proposed Modules

- `backend/portfolio/__init__.py`
- `backend/portfolio/models.py`
- `backend/portfolio/schema.py`
- `backend/portfolio/importers.py`
- `backend/portfolio/classifier.py`
- `backend/portfolio/mapping.py`
- `backend/portfolio/normalizer.py`
- `backend/portfolio/calculations.py`
- `backend/portfolio/reconciliation.py`
- `backend/portfolio/tools.py`
- `backend/portfolio/api.py`
- `backend/tests/test_portfolio_*.py`
- `frontend/src/components/PortfolioView.jsx`
- `frontend/src/components/PortfolioImportReview.jsx`
- `frontend/src/components/PortfolioSummary.jsx`

## Canonical Data Model

### Source Files

Track every imported or indexed financial source.

Required fields:

- `id`
- `path`
- `filename`
- `file_type`
- `content_hash`
- `broker_guess`
- `document_type`
- `import_status`
- `created_at`
- `updated_at`

Statuses:

- `discovered`
- `parsed`
- `needs_mapping_review`
- `imported`
- `imported_with_warnings`
- `failed`
- `ignored`

### Source Tables

For CSV/XLSX, every file may contain one or more tables.

Required fields:

- `id`
- `source_file_id`
- `sheet_name`
- `table_index`
- `header_row_index`
- `data_start_row_index`
- `row_count`
- `column_count`
- `raw_headers_json`
- `detected_type`
- `confidence`

### Column Mappings

Store how each source column maps to canonical fields.

Required fields:

- `id`
- `source_table_id`
- `source_column`
- `canonical_field`
- `transform`
- `confidence`
- `mapping_source`
- `confirmed_by_user`

Mapping sources:

- `known_broker_adapter`
- `header_heuristic`
- `value_pattern_heuristic`
- `llm_suggestion`
- `user_confirmed`

### Canonical Transactions

The normalized transaction table should support trades, dividends, fees, deposits, withdrawals, interest, taxes, transfers, and adjustments.

Required fields:

- `id`
- `source_file_id`
- `source_table_id`
- `source_row_ref`
- `account_id`
- `broker`
- `portfolio_owner`
- `transaction_type`
- `symbol`
- `security_name`
- `asset_type`
- `trade_date`
- `settlement_date`
- `side`
- `quantity`
- `price`
- `gross_amount`
- `fees`
- `taxes`
- `net_amount`
- `currency`
- `exchange`
- `isin`
- `cusip`
- `notes`
- `confidence`
- `warnings_json`
- `created_at`

Transaction types:

- `trade`
- `dividend`
- `interest`
- `fee`
- `tax`
- `deposit`
- `withdrawal`
- `transfer_in`
- `transfer_out`
- `split`
- `merger`
- `adjustment`
- `unknown`

Sides:

- `buy`
- `sell`
- `short_sell`
- `buy_to_cover`
- `unknown`

### Positions And Lots

Maintain derived tables or materialized views for holdings and tax lots.

Fields for lots:

- `id`
- `account_id`
- `symbol`
- `open_date`
- `quantity_opened`
- `quantity_remaining`
- `cost_basis`
- `currency`
- `source_transaction_id`
- `status`

Fields for realized lots:

- `id`
- `sell_transaction_id`
- `buy_lot_id`
- `symbol`
- `quantity_closed`
- `proceeds`
- `cost_basis`
- `fees_allocated`
- `realized_pnl`
- `holding_period_days`
- `currency`

## Broker-Agnostic Mapping Strategy

The system cannot assume broker files use one format. The solution should use a layered mapping approach.

### Layer 1: Known Adapters

Create adapters for common broker exports. Each adapter should include:

- Broker name.
- File signatures.
- Header aliases.
- Date formats.
- Amount sign conventions.
- Transaction-type vocabulary.
- Fees and tax column rules.
- Test fixture files.

Examples:

- Robinhood CSV.
- Fidelity CSV/XLSX.
- Schwab CSV.
- Interactive Brokers activity statement CSV.
- Zerodha tradebook / contract note exports.
- Groww / Indian broker exports if relevant.
- Generic broker fallback.

### Layer 2: Header Aliases

Maintain a dictionary of canonical fields and common header variants.

Examples:

- `symbol`: Symbol, Ticker, Instrument, Scrip, Security Symbol, ISIN Symbol.
- `quantity`: Quantity, Qty, Shares, Units, Filled Qty.
- `price`: Price, Avg Price, Trade Price, Execution Price, Rate.
- `fees`: Fees, Commission, Brokerage, Charges, Transaction Fee.
- `trade_date`: Date, Trade Date, Executed At, Transaction Date, Order Executed Time.
- `side`: Side, Action, Type, Transaction Type, Buy/Sell.
- `net_amount`: Amount, Net Amount, Net Value, Debit/Credit, Total.

### Layer 3: Value Pattern Detection

Inspect sample values to infer meaning.

Examples:

- A column with values like AAPL, TSLA, INFY, RELIANCE may be `symbol`.
- A column with dates across many rows may be `trade_date`.
- A column with BUY/SELL/Bought/Sold may be `side`.
- Decimal values close to known prices may be `price`.
- Integer or decimal share counts may be `quantity`.
- Negative/positive cash values may be `net_amount`.

### Layer 4: Cross-Column Consistency Checks

Validate candidate mappings with accounting relationships.

Examples:

- `gross_amount ~= quantity * price`
- `net_amount ~= gross_amount + fees + taxes` depending on side and broker convention.
- Buy transactions usually decrease cash.
- Sell transactions usually increase cash.
- Fees should usually be non-negative and much smaller than gross amount.

### Layer 5: LLM-Assisted Mapping

Use the LLM only to suggest mappings when rules are insufficient.

Input to the LLM should be small and structured:

- File name.
- Sheet name.
- Raw headers.
- First 5-10 sanitized rows.
- Candidate detected column types.
- Required canonical fields.

The LLM output must be JSON and must include confidence and reasoning. The backend must validate the JSON against allowed canonical fields. The user must confirm low-confidence mappings.

### Layer 6: Saved User Mappings

When a user confirms a mapping, store it by:

- Broker guess.
- File signature.
- Header set hash.
- User/account scope.

Future imports with similar headers should reuse the mapping automatically.

## Import Pipeline Tasks

### Phase 0: Design And Guardrails

- [x] Define exact MVP scope.
- [x] Decide whether the first version supports only realized P&L or both realized and unrealized.
- [x] Decide supported cost-basis methods for MVP: FIFO only, or FIFO plus average cost.
- [x] Define warnings for incomplete data.
- [x] Define what claims the app is allowed to make.
- [x] Add financial disclaimer copy for UI and API responses.
- [x] Decide whether portfolio data lives in the existing `metadata.db` or a separate `portfolio.db`.
- [x] Document privacy model: all parsing and calculations should work locally.
- [x] Document that LLMs must not receive full raw financial files unless the user chooses a cloud provider.

### Phase 1: Structured File Discovery

- [x] Add a financial-file discovery pass after indexing.
- [x] Detect candidate files by extension: `.csv`, `.xlsx`, `.xls`, `.pdf`, `.txt`.
- [x] Detect candidate files by filename keywords: trades, transactions, activity, statement, holdings, positions, dividend, contract, ledger, tax.
- [x] Detect candidate files by content keywords: buy, sell, quantity, symbol, trade date, commission, brokerage, proceeds.
- [x] Store discovered candidates in `portfolio_source_files`.
- [x] Add API endpoint: `GET /api/portfolio/sources`.
- [x] Add API endpoint: `POST /api/portfolio/sources/{id}/ignore`.
- [x] Add tests for file discovery.

### Phase 2: CSV Importer

- [x] Parse CSV with delimiter detection.
- [x] Support comma, semicolon, tab, and pipe delimiters.
- [x] Support UTF-8 with fallback encodings.
- [x] Preserve original row numbers.
- [x] Detect blank lines and multi-line notes.
- [x] Detect repeated headers.
- [x] Detect summary/footer rows.
- [x] Detect header row when metadata appears above the table.
- [x] Extract one or more candidate tables from a CSV.
- [x] Store raw table preview.
- [x] Add tests for simple CSV.
- [x] Add tests for CSV with metadata before headers.
- [x] Add tests for CSV with repeated headers.
- [x] Add tests for CSV with totals/footer rows.
- [x] Add tests for semicolon/tab-delimited files.

### Phase 3: XLSX/XLS Importer

- [x] Add `.xls` support if dependency choice is acceptable.
- [x] Iterate all worksheets.
- [x] Skip empty worksheets.
- [x] Detect tables in each sheet.
- [x] Preserve sheet name and row numbers.
- [x] Detect merged title rows above headers.
- [x] Detect multiple tables on one sheet.
- [x] Detect hidden rows/columns if supported.
- [x] Extract formulas as values where possible.
- [x] Store sheet/table previews.
- [x] Add tests for multi-sheet workbook.
- [x] Add tests for workbook with holdings sheet and trades sheet.
- [x] Add tests for multiple tables in one sheet.

### Phase 4: PDF Importer

- [ ] Start with text-based PDFs only.
- [ ] Use existing PDF text extraction for statement text.
- [ ] Add table extraction strategy if dependencies are acceptable.
- [ ] Detect broker statement sections.
- [ ] Detect transaction tables in PDF text.
- [ ] Preserve page numbers as source references.
- [ ] Mark OCR-scanned PDFs as unsupported or low-confidence in MVP.
- [ ] Add a low-confidence path for PDFs requiring user review.
- [ ] Add tests with synthetic text-based PDF statements.
- [ ] Add warnings when PDF extraction is incomplete.

### Phase 5: Document Classification

- [x] Classify tables as trade history, holdings, dividend statement, cash ledger, account summary, tax report, or unsupported.
- [x] Implement rules based on headers.
- [x] Implement rules based on sample values.
- [x] Add optional LLM classifier for ambiguous files.
- [x] Require confidence score.
- [x] Show classification result in UI.
- [x] Allow user to override classification.
- [x] Store classification in database.
- [x] Add tests for trade-history detection.
- [x] Add tests for holdings detection.
- [x] Add tests for dividend detection.
- [x] Add tests for unsupported tables.

### Phase 6: Column Mapping

- [x] Build canonical field registry.
- [x] Build header alias dictionary.
- [x] Build value-pattern detectors.
- [x] Build consistency scoring.
- [x] Generate candidate mappings with confidence.
- [x] Mark required fields for trade imports: date, symbol, side, quantity, price or amount.
- [x] Mark optional fields: fees, taxes, settlement date, currency, account, broker, security name.
- [x] Validate impossible mappings.
- [x] Add LLM-assisted mapping for ambiguous headers.
- [x] Save confirmed mappings.
- [x] Reuse confirmed mappings for similar future files.
- [x] Add tests for broker-specific mappings.
- [x] Add tests for generic mappings.
- [x] Add tests for ambiguous `Amount` / `Net` / `Value` columns.
- [x] Add tests proving low-confidence mappings are not silently imported.

### Phase 7: Normalization

- [x] Convert raw rows into canonical transactions.
- [x] Normalize dates across formats.
- [x] Normalize symbols and preserve original symbol.
- [x] Normalize buy/sell side vocabulary.
- [x] Normalize numeric formats with commas, currency symbols, parentheses, and negative signs.
- [x] Normalize currencies.
- [x] Infer missing `gross_amount` from quantity and price.
- [x] Infer missing `price` from gross amount and quantity.
- [x] Infer signs for cash amount based on side and broker convention.
- [x] Preserve all raw values in a JSON column for auditability.
- [x] Attach warnings to rows with inferred or missing values.
- [x] Reject rows missing essential data.
- [x] Add tests for date parsing.
- [x] Add tests for currency parsing.
- [x] Add tests for negative amount conventions.
- [x] Add tests for inferred price/gross amount.

### Phase 8: Database Schema

- [x] Add portfolio schema migrations.
- [x] Create `portfolio_source_files`.
- [x] Create `portfolio_source_tables`.
- [x] Create `portfolio_column_mappings`.
- [x] Create `portfolio_transactions`.
- [x] Create `portfolio_lots`.
- [x] Create `portfolio_realized_lots`.
- [x] Create `portfolio_import_warnings`.
- [x] Add indexes on symbol, date, account, broker, source file.
- [x] Add idempotent import logic using file hash and source row reference.
- [x] Add rollback behavior for failed imports.
- [x] Add tests for schema migration.
- [x] Add tests for idempotent re-import.

## Calculation Engine Tasks

### Phase 9: Cost Basis Engine

- [x] Implement FIFO lot matching.
- [x] Add average-cost method if required for the first release.
- [x] Model open lots for buys.
- [x] Close lots for sells.
- [x] Allocate fees to cost/proceeds.
- [x] Handle partial sells.
- [x] Handle selling more than available quantity with warning.
- [x] Handle short-selling as explicit unsupported or separate path.
- [x] Store realized lot rows for auditability.
- [x] Add tests for single buy/sell.
- [x] Add tests for multiple buys and one sell.
- [x] Add tests for partial sells.
- [x] Add tests for fees reducing profit.
- [x] Add tests for open positions.
- [x] Add tests for insufficient lots.

### Phase 10: Realized P&L

- [x] Compute realized profit by account.
- [x] Compute realized profit by broker.
- [x] Compute realized profit by symbol.
- [x] Compute realized profit by date range.
- [x] Compute realized profit by portfolio owner.
- [x] Compute realized loss by symbol.
- [x] Compute net realized P&L after fees.
- [x] Return source transactions supporting each result.
- [x] Add tests for profit by symbol.
- [x] Add tests for top winners.
- [x] Add tests for top losers.

### Phase 11: Unrealized P&L

- [x] Compute open positions from remaining lots.
- [x] Support current prices from an imported holdings file.
- [x] Support user-provided price overrides.
- [x] Optionally add market-price provider later.
- [x] Compute unrealized P&L by symbol.
- [x] Warn when current prices are stale or unavailable.
- [x] Add tests for imported current-price holdings.
- [x] Add tests for missing current prices.

### Phase 12: Dividends, Fees, Cash, And Total Return

- [x] Normalize dividend transactions.
- [x] Summarize dividends by symbol and date range.
- [x] Summarize fees by broker and symbol.
- [x] Summarize deposits and withdrawals.
- [x] Distinguish trading profit from total return.
- [x] Add optional total return calculation: realized P&L + dividends + unrealized P&L.
- [x] Warn when cash ledger is incomplete.
- [x] Add tests for dividend-only profit.
- [x] Add tests for total return with dividends.

### Phase 13: Advanced Accounting Later

- [ ] Add corporate action support: splits.
- [ ] Add mergers/spinoffs as unsupported-with-warning first.
- [ ] Add multi-currency conversion.
- [ ] Add realized FX gain/loss if needed.
- [ ] Add wash-sale detection only as an explicit advanced feature.
- [ ] Add tax-year summaries.
- [ ] Add jurisdiction-specific labels only when implemented.

## Tooling And Agent Integration

### Phase 14: Portfolio Tools

Add tools that small models can call.

Candidate tools:

- `discover_portfolio_files()`
- `preview_portfolio_import(source_id)`
- `confirm_portfolio_mapping(source_table_id, mapping)`
- `import_portfolio_source(source_id)`
- `calculate_realized_profit(owner=None, account=None, broker=None, symbol=None, date_from=None, date_to=None, method="FIFO")`
- `calculate_unrealized_profit(owner=None, account=None, broker=None, symbol=None)`
- `portfolio_profit_by_symbol(limit=10, date_from=None, date_to=None)`
- `portfolio_top_winners(limit=10)`
- `portfolio_top_losers(limit=10)`
- `portfolio_open_positions()`
- `portfolio_import_warnings()`
- `portfolio_explain_result(result_id)`

Each tool should return structured JSON with:

- `answer_summary`
- `numbers`
- `currency`
- `method`
- `filters`
- `sources`
- `warnings`
- `needs_user_action`

### Phase 15: Agent Prompt Changes

- [x] Add portfolio tools to the agent tool registry.
- [x] Update local model prompt to prefer portfolio tools for finance questions.
- [x] Keep tool descriptions short and concrete.
- [x] Tell the model not to do arithmetic manually when a portfolio tool exists.
- [x] Tell the model to surface warnings exactly.
- [x] Add tests proving "How much profit did Siddhesh earn?" routes to portfolio calculation after data is imported.

### Phase 16: Natural-Language Question Coverage

Supported example questions:

- "How much profit did Siddhesh earn so far?"
- "How much realized profit did I make this year?"
- "Which stock was most profitable?"
- "Which stock lost the most money?"
- "Show profit by symbol."
- "What are my open positions?"
- "How much profit came from Apple trades?"
- "Did I make more from NVDA or TSLA?"
- "What was my win rate?"
- "How much did I pay in brokerage or fees?"
- "How much dividend income did I receive?"
- "What files contributed to this answer?"
- "Which rows support this profit number?"

For each question, define:

- Required tool.
- Required data.
- Expected warnings if data is missing.
- Test fixture.
- Example response.

## API Tasks

### Phase 17: Backend Endpoints

- [x] `GET /api/portfolio/sources`
- [x] `POST /api/portfolio/discover`
- [x] `GET /api/portfolio/sources/{id}/preview`
- [x] `POST /api/portfolio/sources/{id}/mapping`
- [x] `POST /api/portfolio/sources/{id}/import`
- [x] `GET /api/portfolio/transactions`
- [x] `GET /api/portfolio/positions`
- [x] `GET /api/portfolio/pnl/realized`
- [x] `GET /api/portfolio/pnl/unrealized`
- [x] `GET /api/portfolio/pnl/by-symbol`
- [x] `GET /api/portfolio/warnings`
- [x] `DELETE /api/portfolio/sources/{id}`

All frontend calls must go through `frontend/src/lib/api.js`.

### Phase 18: Validation And Error Responses

- [x] Return structured validation errors.
- [x] Include mapping confidence.
- [x] Include row-level warning counts.
- [x] Include import-blocking errors separately from non-blocking warnings.
- [x] Do not expose full sensitive file paths unnecessarily in UI.
- [x] Preserve detailed logs locally.

## Frontend Tasks

### Phase 19: Portfolio Navigation

- [x] Add Portfolio item to sidebar.
- [x] Add Portfolio view route/state in `App.jsx`.
- [x] Keep layout consistent with existing app.
- [x] Avoid marketing-style screens; make it a practical analysis workspace.

### Phase 20: Import Review UI

- [x] Show discovered source files.
- [x] Show detected document type.
- [x] Show sheets/tables found in each file.
- [x] Show mapping confidence.
- [x] Show preview rows.
- [x] Allow user to correct mapped columns.
- [x] Allow user to mark a table as ignored.
- [x] Allow user to confirm and import.
- [x] Show row-level warning count.
- [x] Show import summary after completion.

### Phase 21: Analytics UI

- [x] Summary cards for realized P&L, unrealized P&L, total fees, dividends, open positions.
- [x] Table for profit by symbol.
- [x] Table for top winners and top losers.
- [x] Open positions table.
- [x] Source drill-down for a number.
- [x] Warning panel for incomplete/ambiguous data.
- [x] Filters for owner, account, broker, date range, symbol, cost-basis method.

### Phase 22: Search/Agent UX

- [x] Let normal search still find financial files.
- [x] Let agent mode answer portfolio questions through tools.
- [x] Show when an answer came from deterministic portfolio tools.
- [x] Include method and warnings in answer UI.
- [x] Let user open source file/row references where possible.

## Testing Strategy

### Unit Tests

- [ ] CSV table detection.
- [ ] XLSX sheet/table detection.
- [ ] Header mapping.
- [ ] Value-pattern mapping.
- [ ] LLM mapping JSON validation with mocked LLM.
- [ ] Normalization.
- [ ] FIFO lot matching.
- [ ] Profit by symbol.
- [ ] Import idempotency.
- [ ] Warning generation.

### Integration Tests

- [ ] Import one simple CSV and calculate realized profit.
- [ ] Import two broker CSVs and merge profit by symbol.
- [ ] Import one XLSX with multiple sheets.
- [ ] Import trades plus holdings and calculate unrealized P&L.
- [ ] Ask agent "Which stock was most profitable?" and verify tool-backed answer.
- [ ] Ask agent "How much profit did Siddhesh earn?" and verify it does not answer from raw RAG alone.

### Golden Fixtures

Create synthetic files under a test fixture directory, not under user data.

Fixtures:

- Simple one-broker CSV.
- CSV with odd headers.
- CSV with metadata rows.
- XLSX with multiple sheets.
- XLSX with trades and holdings.
- Text-based PDF statement.
- Ambiguous CSV requiring mapping review.
- File with dividends.
- File with fees.
- File with missing price but present gross amount.

### Manual Validation

- [ ] Import fixture files in the running app.
- [ ] Confirm mappings in UI.
- [ ] Ask example questions in agent mode.
- [ ] Verify values against hand-calculated expected results.
- [ ] Verify warnings appear when prices, fees, or mappings are missing.

## Security, Privacy, And Safety

- [ ] Keep all financial data local by default.
- [ ] Do not send full files to cloud providers automatically.
- [ ] If using cloud LLM mapping, send only small sanitized previews and require user opt-in.
- [ ] Store source lineage for every normalized row.
- [ ] Log errors without leaking full financial content.
- [ ] Treat imported financial data as sensitive.
- [ ] Add clear "not financial advice" language.
- [ ] Avoid claiming tax correctness unless explicitly implemented.

## Open-Source Contributor Plan

Good first issues:

- Add canonical field registry.
- Add CSV delimiter/header detection tests.
- Add symbol/date/number parsing helpers.
- Add simple FIFO calculator tests.
- Add synthetic fixture files.
- Add Portfolio sidebar entry.

Medium issues:

- Build generic CSV importer.
- Build XLSX multi-sheet importer.
- Build mapping confidence scorer.
- Build realized P&L endpoint.
- Build import review UI.

Advanced issues:

- Broker-specific adapters.
- PDF statement extraction.
- LLM-assisted mapping with schema validation.
- Multi-currency support.
- Corporate actions.
- Source-row drill-down UI.

## Rough Effort Estimate

MVP with CSV/XLSX, generic mapping, SQLite ledger, FIFO realized P&L, basic portfolio tools, tests, and minimal UI:

- Backend importers and mapping: 600-900 LOC.
- Database schema and persistence: 250-400 LOC.
- Calculation engine: 400-700 LOC.
- Agent/tool/API integration: 300-500 LOC.
- Frontend import/summary UI: 500-900 LOC.
- Tests and fixtures: 700-1200 LOC.

Expected MVP total: 2750-4600 LOC including tests and UI.

If only backend tools are built first with no polished UI, a useful first version could be closer to 1500-2500 LOC including tests.

Full multi-broker, PDF-aware, dividend/cash/currency/corporate-action system:

- 8000-15000+ LOC over multiple releases.

## Release Slices

### Slice 1: Deterministic CSV Proof Of Concept

Outcome:

- Import a simple CSV.
- Map known columns.
- Store transactions.
- Calculate realized FIFO P&L.
- Answer "How much profit did I make?" through a tool.

### Slice 2: Broker-Agnostic Mapping

Outcome:

- Import odd CSV headers.
- Detect likely mappings.
- Require review for ambiguity.
- Save confirmed mapping.

### Slice 3: XLSX Multi-Sheet Support

Outcome:

- Import multiple sheets.
- Detect trade sheet vs holdings sheet.
- Merge data from multiple sources.

### Slice 4: Portfolio Analytics

Outcome:

- Profit by symbol.
- Top winners.
- Top losers.
- Open positions.
- Fees and dividends.

### Slice 5: PDF And Broker Adapters

Outcome:

- Parse text-based PDF statements.
- Add specific broker adapters.
- Improve confidence and reduce review burden.

### Slice 6: Advanced Finance

Outcome:

- Currency conversion.
- Corporate actions.
- Tax-year summaries.
- Optional market data integration.

## Acceptance Criteria For MVP

- A user can import at least one CSV and one XLSX file.
- Multi-sheet XLSX files are handled.
- Ambiguous mappings are not silently accepted.
- Confirmed mappings are saved and reused.
- Realized P&L is calculated by deterministic code using FIFO.
- The system can answer "How much profit did Siddhesh earn?" from imported data.
- The system can answer "Which stock made the most profit?"
- The system can answer "Which stock lost the most money?"
- The system can show source files and rows behind each answer.
- The system warns clearly when data is incomplete.
- Unit tests cover importer, mapper, normalizer, and calculator behavior.
- Agent tests prove small-model-friendly tool routing.

