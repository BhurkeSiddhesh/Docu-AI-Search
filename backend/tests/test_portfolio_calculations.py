"""Hand-calculated accounting regressions; no network or LLM access."""

from datetime import date
from decimal import Decimal
import json
import unittest

from backend.portfolio.calculations import calculate_portfolio


def trade(identifier, side, quantity, price, day="2025-01-01", **overrides):
    gross = str(Decimal(str(quantity)) * Decimal(str(price))) if quantity is not None and price is not None else None
    row = {"id": identifier, "source_file_id": "file1", "source_table_id": "table1",
           "source_row_ref": identifier + 1, "filename": "trades.csv", "sheet_name": "Trades",
           "transaction_type": "trade", "side": side, "quantity": str(quantity) if quantity is not None else None,
           "price": str(price) if price is not None else None, "gross_amount": gross,
           "fees": "0", "taxes": "0", "currency": "USD", "symbol": "ABC",
           "account_id": "account1", "broker": "Example", "portfolio_owner": "Sam",
           "trade_date": day}
    row.update(overrides)
    return row


def cash(identifier, kind, amount, **overrides):
    row = trade(identifier, "", None, None, transaction_type=kind,
                gross_amount=str(amount), net_amount=None)
    row.update(overrides)
    return row


class TestPortfolioCalculations(unittest.TestCase):
    def test_single_sale_includes_fees_and_taxes_once(self):
        result = calculate_portfolio([
            trade(1, "buy", 10, 100, fees="2", taxes="1"),
            trade(2, "sell", 10, 120, "2025-02-01", fees="3", taxes="2"),
        ])
        self.assertEqual(result["totals"][0]["realized_pnl"], "192")
        self.assertEqual(result["totals"][0]["fees"], "5")
        self.assertEqual(result["totals"][0]["taxes"], "3")
        self.assertEqual(result["positions"], [])
        self.assertFalse(result["needs_user_action"])
        lot = result["realized_lots"][0]
        self.assertEqual(lot["cost_basis"], "1003")
        self.assertEqual(lot["proceeds"], "1195")
        self.assertEqual(lot["holding_period_days"], 31)
        self.assertEqual(lot["buy_source"]["transaction_id"], 1)
        self.assertEqual(lot["sell_source"]["transaction_id"], 2)
        json.dumps(result, allow_nan=False)

    def test_fifo_partial_close_preserves_remaining_buy_fees(self):
        result = calculate_portfolio([
            trade(1, "buy", 10, 100, fees="10"),
            trade(2, "buy", 10, 120, "2025-02-01", fees="20"),
            trade(3, "sell", 15, 150, "2025-03-01", fees="15"),
        ])
        # Buy costs closed: 1010 + 610. Net proceeds: 2250 - 15.
        self.assertEqual(result["totals"][0]["realized_pnl"], "615")
        self.assertEqual([lot["quantity_closed"] for lot in result["realized_lots"]], ["10", "5"])
        self.assertEqual([lot["realized_pnl"] for lot in result["realized_lots"]], ["480", "135"])
        position = result["positions"][0]
        self.assertEqual(position["quantity"], "5")
        self.assertEqual(position["cost_basis"], "610")
        self.assertIsNone(position["unrealized_pnl"])
        self.assertTrue(result["realized_complete"])
        self.assertFalse(result["unrealized_complete"])

    def test_all_partial_closes_consume_exact_fee_remainder(self):
        result = calculate_portfolio([
            trade(1, "buy", 3, 10, fees="1"),
            trade(2, "sell", 1, 20, "2025-01-02"),
            trade(3, "sell", 1, 20, "2025-01-03"),
            trade(4, "sell", 1, 20, "2025-01-04"),
        ])
        self.assertEqual(round(Decimal(result["totals"][0]["realized_pnl"]), 4), Decimal("29"))
        self.assertEqual(round(sum(Decimal(lot["buy_fees_allocated"]) for lot in result["realized_lots"]), 4), Decimal("1"))
        self.assertEqual(result["positions"], [])

    def test_fractional_shares_and_decimal_costs(self):
        result = calculate_portfolio([trade(1, "buy", ".125", "12.8"),
                                      trade(2, "sell", ".025", "16.4", "2025-01-02")])
        self.assertEqual(result["totals"][0]["realized_pnl"], "0.09")
        self.assertEqual(result["positions"][0]["quantity"], "0.1")
        self.assertEqual(result["positions"][0]["cost_basis"], "1.28")

    def test_missing_purchase_lots_never_turn_unmatched_proceeds_into_profit(self):
        result = calculate_portfolio([
            trade(1, "buy", 5, 10), trade(2, "sell", 8, 20, "2025-01-02", fees="8")])
        self.assertEqual(result["totals"][0]["realized_pnl"], "45")
        self.assertEqual(result["realized_lots"][0]["quantity_closed"], "5")
        self.assertEqual(result["excluded_transactions"][0]["quantity"], "3")
        self.assertEqual(result["excluded_transactions"][0]["excluded_gross_proceeds"], "60")
        self.assertFalse(result["realized_complete"])
        self.assertTrue(result["needs_user_action"])
        self.assertTrue(any("Unmatched sale proceeds are excluded" in warning for warning in result["warnings"]))

    def test_date_range_uses_earlier_buys_and_consumes_earlier_sells(self):
        rows = [trade(1, "buy", 10, 10, "2024-01-01", fees="10"),
                trade(2, "sell", 4, 20, "2024-06-01"),
                trade(3, "sell", 3, 30, "2025-02-01", fees="3"),
                trade(4, "sell", 3, 40, "2026-01-01")]
        result = calculate_portfolio(rows, {"date_from": "2025-01-01", "date_to": "2025-12-31"})
        self.assertEqual(result["totals"][0]["realized_pnl"], "54")
        self.assertEqual(result["totals"][0]["fees"], "3")
        self.assertEqual(result["positions"][0]["quantity"], "3")
        self.assertEqual(result["positions"][0]["cost_basis"], "33")
        self.assertEqual(result["positions"][0]["as_of"], "2025-12-31")
        self.assertEqual({s["transaction_id"] for s in result["sources"]}, {1, 3})

    def test_currency_account_broker_and_owner_have_separate_lots(self):
        for changed_field, changed_value in [("currency", "INR"), ("account_id", "account2"),
                                             ("broker", "Different"), ("portfolio_owner", "Other")]:
            with self.subTest(field=changed_field):
                result = calculate_portfolio([trade(1, "buy", 10, 10),
                                              trade(2, "sell", 10, 20, **{changed_field: changed_value})])
                self.assertEqual(len(result["realized_lots"]), 0)
                self.assertFalse(result["realized_complete"])
                self.assertEqual(result["positions"][0]["quantity"], "10")
                self.assertEqual(len(result["by_symbol"]), 2)

    def test_unidentified_accounts_from_different_files_do_not_match(self):
        result = calculate_portfolio([trade(1, "buy", 5, 10, account_id="", source_file_id="a"),
                                      trade(2, "sell", 5, 20, account_id="", source_file_id="b")])
        self.assertEqual(result["realized_lots"], [])
        self.assertFalse(result["realized_complete"])

    def test_explicit_account_matches_lots_across_files(self):
        result = calculate_portfolio([trade(1, "buy", 5, 10, source_file_id="a"),
                                      trade(2, "sell", 5, 20, source_file_id="b")])
        self.assertEqual(result["totals"][0]["realized_pnl"], "50")
        self.assertEqual({source["source_file_id"] for source in result["sources"]}, {"a", "b"})

    def test_filters_accept_owner_account_aliases(self):
        rows = [trade(1, "buy", 5, 10), trade(2, "sell", 5, 20),
                trade(3, "buy", 5, 10, portfolio_owner="Other")]
        result = calculate_portfolio(rows, {"owner": "sam", "account": "account1", "symbol": "abc"})
        self.assertEqual(result["totals"][0]["realized_pnl"], "50")
        self.assertEqual(result["positions"], [])

    def test_dividends_interest_fees_and_cash_are_not_trading_profit(self):
        result = calculate_portfolio([cash(1, "dividend", 25, taxes="5"),
                                      cash(2, "interest", 3), cash(3, "fee", 2),
                                      cash(4, "tax", 1), cash(5, "deposit", 1000),
                                      cash(6, "withdrawal", 50)])
        total = result["totals"][0]
        self.assertEqual(total["realized_pnl"], "0")
        self.assertEqual(total["dividends"], "20")
        self.assertEqual(total["interest"], "3")
        self.assertEqual(total["fees"], "2")
        self.assertEqual(total["taxes"], "6")
        self.assertEqual(total["deposits"], "1000")
        self.assertEqual(total["withdrawals"], "50")

    def test_standalone_fee_column_is_not_counted_twice(self):
        result = calculate_portfolio([cash(1, "fee", 3, fees="3")])
        self.assertEqual(result["totals"][0]["fees"], "3")

    def test_dated_price_computes_unrealized_after_buy_costs(self):
        result = calculate_portfolio([trade(1, "buy", 5, 10, fees="2")],
                                     {"date_to": "2025-02-01"},
                                     {"ABC": {"price": "15", "currency": "USD", "as_of": "2025-01-31"}})
        self.assertEqual(result["positions"][0]["market_value"], "75")
        self.assertEqual(result["positions"][0]["unrealized_pnl"], "23")
        self.assertEqual(result["totals"][0]["unrealized_pnl"], "23")
        self.assertFalse(result["needs_user_action"])

    def test_missing_stale_future_undated_and_wrong_currency_prices_stay_unknown(self):
        prices = [None, {"price": "15", "currency": "USD"},
                  {"price": "15", "currency": "INR", "as_of": "2025-02-01"},
                  {"price": "15", "currency": "USD", "as_of": "2025-01-01"},
                  {"price": "15", "currency": "USD", "as_of": "2025-02-02"},
                  {"price": "NaN", "currency": "USD", "as_of": "2025-02-01"}]
        for quote in prices:
            with self.subTest(quote=quote):
                result = calculate_portfolio([trade(1, "buy", 5, 10)], {"date_to": "2025-02-01"}, {"ABC": quote})
                self.assertIsNone(result["positions"][0]["unrealized_pnl"])
                self.assertIsNone(result["totals"][0]["unrealized_pnl"])
                self.assertTrue(result["warnings"])

    def test_one_missing_price_prevents_partial_unrealized_total(self):
        result = calculate_portfolio([trade(1, "buy", 5, 10), trade(2, "buy", 5, 10, symbol="XYZ")],
                                     {"date_to": "2025-02-01"},
                                     {"ABC": {"price": "15", "currency": "USD", "as_of": "2025-02-01"}})
        self.assertIsNone(result["totals"][0]["unrealized_pnl"])
        self.assertEqual(result["positions"][0]["unrealized_pnl"], "25")

    def test_short_and_corporate_actions_block_untrusted_accounting(self):
        for update in [{"side": "short_sell"}, {"side": "buy_to_cover"},
                       {"transaction_type": "split"}, {"transaction_type": "merger"}]:
            with self.subTest(update=update):
                t2 = trade(2, "sell", 10, 20)
                t2.update(update)
                result = calculate_portfolio([trade(1, "buy", 10, 10), t2])
                self.assertFalse(result["realized_complete"])
                self.assertIsNone(result["totals"][0]["realized_pnl"])
                self.assertEqual(result["positions"], [])
                self.assertEqual(result["realized_lots"], [])

    def test_invalid_trade_values_are_rejected_instead_of_reported_as_exact(self):
        for update in [{"quantity": "NaN"}, {"price": "Infinity"}, {"quantity": "-1"},
                       {"fees": "-1"}, {"currency": ""}, {"trade_date": "bad"},
                       {"gross_amount": "1000000"}]:
            with self.subTest(update=update):
                t1 = trade(1, "buy", 10, 10)
                t1.update(update)
                result = calculate_portfolio([t1])
                self.assertTrue(result["needs_user_action"])
                self.assertFalse(result["realized_complete"])
                self.assertTrue(result["excluded_transactions"])
                json.dumps(result, allow_nan=False)

    def test_invalid_filters_fail_clearly(self):
        for filters in [{"method": "AVERAGE"}, {"date_from": "bad"},
                        {"date_from": "2026-01-01", "date_to": "2025-01-01"}]:
            with self.assertRaises(ValueError):
                calculate_portfolio([], filters)

    def test_input_order_does_not_override_transaction_dates(self):
        result = calculate_portfolio([trade(2, "sell", 5, 20, "2025-02-01"),
                                      trade(1, "buy", 5, 10, "2025-01-01")])
        self.assertEqual(result["totals"][0]["realized_pnl"], "50")

    def test_no_transactions_does_not_claim_a_measured_zero(self):
        result = calculate_portfolio([])
        self.assertEqual(result["totals"], [])
        self.assertTrue(result["needs_user_action"])
        self.assertIn("No imported transactions", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()
