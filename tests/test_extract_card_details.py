import unittest

from extract_card_details import (
    CardTextExtraction,
    WebSearchPriceProvider,
    _extract_collector_fraction,
    build_price_search_query,
)


class ExtractCardDetailsTests(unittest.TestCase):
    def test_build_search_query_with_full_fields(self) -> None:
        details = CardTextExtraction(
            card_name="Charizard ex",
            collector_number="223",
            set_total="197",
            raw_name_text="Charizard ex",
            raw_number_text="223/197",
        )
        provider = WebSearchPriceProvider(engine="google")

        result = build_price_search_query(details, provider)

        self.assertEqual(result.query, "\"Charizard ex\" \"223/197\" pokemon card price")
        self.assertIn("google.com/search", result.url)
        self.assertIn("q=%22Charizard+ex%22+%22223%2F197%22+pokemon+card+price", result.url)

    def test_build_search_query_degrades_gracefully_with_missing_fields(self) -> None:
        details = CardTextExtraction(
            card_name=None,
            collector_number=None,
            set_total=None,
            raw_name_text="",
            raw_number_text="",
        )
        provider = WebSearchPriceProvider(engine="bing")

        result = build_price_search_query(details, provider)

        self.assertEqual(result.query, "pokemon card price")
        self.assertIn("bing.com/search", result.url)

    def test_extract_collector_fraction_from_noisy_text(self) -> None:
        collector_number, set_total = _extract_collector_fraction("Noisy value: 12 / 165")

        self.assertEqual(collector_number, "12")
        self.assertEqual(set_total, "165")


if __name__ == "__main__":
    unittest.main()
