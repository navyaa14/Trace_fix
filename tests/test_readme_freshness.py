
import re
import unittest
import tests._pathfix

from graph import build_support_pipeline
from simulate import generate_traces, ScenarioConfig
from attribution import FailureAttributor
from demo import accuracy


README_PATH = "README.md"

_ANCHOR_RE = re.compile(
    r"Current values:\s*"
    r"\*\*([\d.]+)%\*\*\s*all-at-once\s*/\s*"
    r"\*\*([\d.]+)%\*\*\s*binary-search agent-level accuracy,\s*"
    r"\*\*([\d.]+)%\*\*\s*failure rate,\s*"
    r"\*\*n=(\d+)\*\*\s*evaluable failed traces,\s*"
    r"\*\*(\d+)\s*tests\*\*",
    re.MULTILINE,
)


class TestReadmeFreshness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(README_PATH, encoding="utf-8") as f:
            cls.readme_text = f.read()
        match = _ANCHOR_RE.search(cls.readme_text)
        assert match is not None, (
            "Could not find the 'Current values: ...' anchor sentence in "
            "README.md's 'Latest revision' section -- either it was reworded "
            "(update _ANCHOR_RE to match) or deleted (put it back)."
        )
        cls.readme_all_at_once_pct = float(match.group(1))
        cls.readme_binary_search_pct = float(match.group(2))
        cls.readme_failure_rate_pct = float(match.group(3))
        cls.readme_n_eval = int(match.group(4))
        cls.readme_test_count = int(match.group(5))

        graph = build_support_pipeline()
        cls.traces = generate_traces(graph, n=300, config=ScenarioConfig(), seed=11)
        cls.attributor = FailureAttributor()

    def test_all_at_once_accuracy_matches_readme(self):
        acc, _, n_eval = accuracy(self.attributor.attribute_all_at_once, self.traces)
        self.assertEqual(n_eval, self.readme_n_eval)
        self.assertAlmostEqual(acc * 100, self.readme_all_at_once_pct, places=1)

    def test_binary_search_accuracy_matches_readme(self):
        acc, _, n_eval = accuracy(self.attributor.attribute_binary_search, self.traces)
        self.assertEqual(n_eval, self.readme_n_eval)
        self.assertAlmostEqual(acc * 100, self.readme_binary_search_pct, places=1)

    def test_failure_rate_matches_readme(self):
        fail_rate = sum(1 for t in self.traces if t.final_outcome_failed) / len(self.traces)
        self.assertAlmostEqual(fail_rate * 100, self.readme_failure_rate_pct, places=1)

    def test_test_count_matches_readme(self):
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir="tests", top_level_dir=".")

        def _count(s):
            total = 0
            for item in s:
                if isinstance(item, unittest.TestSuite):
                    total += _count(item)
                else:
                    total += 1
            return total

        live_count = _count(suite)
        self.assertEqual(live_count, self.readme_test_count,
                          f"README says {self.readme_test_count} tests; live discovery found "
                          f"{live_count}. Re-run the suite and update the README's 'Current "
                          f"values:' sentence, not this test.")

    def test_no_stale_bare_percentages_reintroduced_as_the_headline_claim(self):
        anchor_sentence = _ANCHOR_RE.search(self.readme_text).group(0)
        for stale in ("56%", "84%", "8.3%", "n=25", "117 tests"):
            self.assertNotIn(stale, anchor_sentence,
                              f"stale value {stale!r} reappeared in the 'Current values:' sentence")


if __name__ == "__main__":
    unittest.main()
