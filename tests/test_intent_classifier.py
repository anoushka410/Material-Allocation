"""Unit tests for nlp.intent_classifier."""
import pytest
from nlp.intent_classifier import classify_intent, extract_parameters, _keyword_classify


class TestKeywordClassify:
    def test_greeting(self):
        assert classify_intent("hello") == "greeting"
        assert classify_intent("hi") == "greeting"
        assert classify_intent("Hey there") == "greeting"

    def test_transfer_intent(self):
        assert _keyword_classify("explain the transfer recommendations") == "transfer_recommendations"
        assert _keyword_classify("show transfer recommendation details") == "transfer_recommendations"

    def test_manufacturing_intent(self):
        assert _keyword_classify("explain manufacturing decisions") == "manufacturing_plan"
        assert _keyword_classify("what items need to be manufactured today") == "manufacturing_plan"

    def test_scenario_summary(self):
        assert _keyword_classify("give me a scenario summary") == "scenario_summary"
        assert _keyword_classify("show overview results") == "scenario_summary"

    def test_total_counts(self):
        assert _keyword_classify("how many transfers are there") == "total_counts"
        assert _keyword_classify("count total recommendations") == "total_counts"

    def test_top_transfers_by_cost(self):
        assert _keyword_classify("top transfer by cost") == "top_transfers_by_cost"
        assert _keyword_classify("most expensive transfers") == "top_transfers_by_cost"

    def test_top_transfers_by_quantity(self):
        assert _keyword_classify("top transfer by quantity") == "top_transfers_by_quantity"
        assert _keyword_classify("transfers by units transferred") == "top_transfers_by_quantity"

    def test_top_manufacturing_items(self):
        assert _keyword_classify("top manufacturing by cost") == "top_manufacturing_items"
        assert _keyword_classify("most expensive manufacturing recommendation") == "top_manufacturing_items"

    def test_urgent_transfers(self):
        assert _keyword_classify("urgent transfers") == "urgent_transfers"
        assert _keyword_classify("critical transfers to prevent stockout") == "urgent_transfers"

    def test_high_cost_actions(self):
        assert _keyword_classify("most expensive actions") == "high_cost_actions"
        assert _keyword_classify("costly transfers") == "high_cost_actions"

    def test_reason_analysis(self):
        assert _keyword_classify("why were these transfers made") == "reason_analysis"
        assert _keyword_classify("explain the reason codes") == "reason_analysis"

    def test_store_activity(self):
        assert _keyword_classify("which stores are most active") == "store_activity"
        assert _keyword_classify("show store involvement") == "store_activity"

    def test_cost_breakdown(self):
        assert _keyword_classify("show cost breakdown") == "cost_breakdown"
        assert _keyword_classify("cost structure details") == "cost_breakdown"

    def test_inventory_status(self):
        assert _keyword_classify("inventory status") == "inventory_status"
        assert _keyword_classify("stock level summary") == "inventory_status"

    def test_inventory_gaps(self):
        assert _keyword_classify("inventory gaps below target") == "inventory_gaps"
        assert _keyword_classify("low inventory shortage") == "inventory_gaps"

    def test_scenario_comparison(self):
        assert _keyword_classify("compare scenarios baseline vs alternative") == "scenario_comparison"
        assert _keyword_classify("scenario comparison") == "scenario_comparison"

    def test_out_of_scope(self):
        assert _keyword_classify("what is the weather today") == "out_of_scope"
        assert _keyword_classify("tell me a joke") == "out_of_scope"

    def test_top_manufacturing_beats_explain_manufacturing(self):
        # Priority intents (like top_manufacturing_items) are checked before general intents
        # (like manufacturing_plan) in _keyword_classify. See the priority_intents list
        # in intent_classifier.py.
        assert _keyword_classify("top manufacturing actions") == "top_manufacturing_items"

    def test_top_transfers_beats_explain_transfer(self):
        assert _keyword_classify("top transfer routes by cost") == "top_transfers_by_cost"


class TestExtractParameters:
    def test_product_id_extraction(self):
        params = extract_parameters("show details for product 489")
        assert "product_489" in params["product_id"]

    def test_store_id_extraction(self):
        params = extract_parameters("transfers from store 60")
        assert "store_60" in params["store_id"]

    def test_numeric_limit_digits(self):
        params = extract_parameters("top 5 transfers")
        assert params["limit"] == 5

    def test_numeric_limit_words(self):
        params = extract_parameters("show top three transfers")
        assert params["limit"] == 3

    def test_numeric_limit_ordinal(self):
        # "show me ten transfers" — uses word number "ten"
        params = extract_parameters("show me ten transfers")
        assert params["limit"] == 10

    def test_is_all_flag(self):
        params = extract_parameters("show all transfers")
        assert params["is_all"] is True

    def test_is_detailed_flag(self):
        params = extract_parameters("explain the detailed transfer breakdown")
        assert params["is_detailed"] is True

    def test_no_limit_returns_none(self):
        params = extract_parameters("show transfer recommendations")
        assert params["limit"] is None

    def test_multiple_products(self):
        params = extract_parameters("compare product 100 and product 200")
        assert "product_100" in params["product_id"]
        assert "product_200" in params["product_id"]

    def test_sort_by_cost(self):
        params = extract_parameters("list transfers sorted by cost")
        assert params["sort_by"] == "cost"

    def test_limit_capped_at_max(self):
        """Requesting more than MAX_RESULTS should be capped at 10."""
        params = extract_parameters("top 50 transfers")
        assert params["limit"] == 10

