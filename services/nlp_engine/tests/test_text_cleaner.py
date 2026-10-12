"""
test_text_cleaner.py — Phase 2 Test Checkpoint
================================================
Tests every cleaning step in isolation AND the full `clean()` pipeline.
Uses only real job-description-like strings — no mocks needed.

Run with: python -m unittest services/nlp_engine/tests/test_text_cleaner.py -v
"""

import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp_engine.pipeline.text_cleaner import (
    clean,
    clean_batch,
    decode_html_entities,
    strip_html_tags,
    strip_urls,
    strip_emails,
    strip_bullet_markers,
    strip_noise_chars,
    normalize_whitespace,
)


# ---------------------------------------------------------------------------
# Individual Step Tests
# ---------------------------------------------------------------------------

class TestDecodeHtmlEntities(unittest.TestCase):
    """Tests for decode_html_entities()."""

    def test_ampersand(self):
        self.assertEqual(decode_html_entities("Python &amp; FastAPI"), "Python & FastAPI")

    def test_less_than_greater_than(self):
        self.assertEqual(decode_html_entities("1 &lt; 2 &gt; 0"), "1 < 2 > 0")

    def test_apostrophe_numeric(self):
        self.assertEqual(decode_html_entities("it&#39;s required"), "it's required")

    def test_nbsp(self):
        result = decode_html_entities("Python&nbsp;Developer")
        self.assertIn("Python", result)
        self.assertIn("Developer", result)

    def test_no_entities_unchanged(self):
        text = "Python FastAPI Docker"
        self.assertEqual(decode_html_entities(text), text)

    def test_empty_string(self):
        self.assertEqual(decode_html_entities(""), "")


class TestStripHtmlTags(unittest.TestCase):
    """Tests for strip_html_tags()."""

    def test_simple_tags(self):
        result = strip_html_tags("<p>We need Python developers.</p>")
        self.assertIn("We need Python developers.", result)
        self.assertNotIn("<p>", result)
        self.assertNotIn("</p>", result)

    def test_bold_tag(self):
        result = strip_html_tags("We need <b>React.js</b> skills")
        self.assertIn("React.js", result)
        self.assertNotIn("<b>", result)

    def test_nested_tags(self):
        result = strip_html_tags("<div><ul><li>Python</li><li>Docker</li></ul></div>")
        self.assertIn("Python", result)
        self.assertIn("Docker", result)
        self.assertNotIn("<", result)

    def test_self_closing_br(self):
        result = strip_html_tags("Line1<br/>Line2")
        self.assertIn("Line1", result)
        self.assertIn("Line2", result)

    def test_attributes_stripped(self):
        result = strip_html_tags('<a href="http://example.com">Click here</a>')
        self.assertNotIn("href", result)
        self.assertIn("Click here", result)

    def test_no_html_unchanged(self):
        text = "Python FastAPI Docker"
        result = strip_html_tags(text)
        self.assertIn("Python", result)
        self.assertIn("FastAPI", result)

    def test_empty_string(self):
        self.assertEqual(strip_html_tags(""), "")

    def test_only_tags_returns_empty_or_whitespace(self):
        result = strip_html_tags("<p></p><div></div>").strip()
        self.assertEqual(result, "")


class TestStripUrls(unittest.TestCase):
    """Tests for strip_urls()."""

    def test_http_url(self):
        result = strip_urls("Apply at http://jobs.example.com/python")
        self.assertNotIn("http://", result)
        self.assertNotIn("jobs.example.com", result)

    def test_https_url(self):
        result = strip_urls("Visit https://careers.company.io/apply")
        self.assertNotIn("https://", result)

    def test_www_url(self):
        result = strip_urls("See www.company.com for details")
        self.assertNotIn("www.company.com", result)

    def test_ftp_url(self):
        result = strip_urls("Download from ftp://files.server.com/doc.pdf")
        self.assertNotIn("ftp://", result)

    def test_non_url_text_preserved(self):
        result = strip_urls("We use Python, FastAPI, Docker")
        self.assertIn("Python", result)
        self.assertIn("FastAPI", result)
        self.assertIn("Docker", result)

    def test_empty_string(self):
        self.assertEqual(strip_urls("").strip(), "")


class TestStripEmails(unittest.TestCase):
    """Tests for strip_emails()."""

    def test_simple_email(self):
        result = strip_emails("Contact hr@company.com for info")
        self.assertNotIn("hr@company.com", result)
        self.assertIn("Contact", result)

    def test_email_with_dots(self):
        result = strip_emails("Email john.doe@tech-corp.io")
        self.assertNotIn("john.doe@tech-corp.io", result)

    def test_email_with_plus(self):
        result = strip_emails("Send to user+filter@gmail.com")
        self.assertNotIn("user+filter@gmail.com", result)

    def test_non_email_text_preserved(self):
        result = strip_emails("We need Python and FastAPI skills")
        self.assertIn("Python", result)
        self.assertIn("FastAPI", result)

    def test_empty_string(self):
        self.assertEqual(strip_emails("").strip(), "")


class TestStripBulletMarkers(unittest.TestCase):
    """Tests for strip_bullet_markers()."""

    def test_bullet_dot(self):
        result = strip_bullet_markers("• Python\n• Docker")
        self.assertNotIn("•", result)
        self.assertIn("Python", result)

    def test_dash_bullet(self):
        result = strip_bullet_markers("- Python\n- FastAPI")
        self.assertIn("Python", result)
        self.assertIn("FastAPI", result)

    def test_em_dash(self):
        result = strip_bullet_markers("— React.js required")
        self.assertIn("React.js", result)

    def test_arrow_bullet(self):
        result = strip_bullet_markers("► Strong Python skills")
        self.assertIn("Python", result)
        self.assertNotIn("►", result)


class TestStripNoiseChars(unittest.TestCase):
    """Tests for strip_noise_chars()."""

    def test_equal_divider(self):
        result = strip_noise_chars("=====Requirements=====")
        self.assertNotIn("=====", result)
        self.assertIn("Requirements", result)

    def test_dash_divider(self):
        result = strip_noise_chars("---Skills---")
        self.assertIn("Skills", result)

    def test_skill_names_preserved(self):
        """Critical: C++, C#, React.js, ASP.NET must NOT be damaged."""
        text = "C++ C# React.js ASP.NET Node.js"
        result = strip_noise_chars(text)
        self.assertIn("C++", result)
        self.assertIn("C#", result)
        self.assertIn("React.js", result)
        self.assertIn("ASP.NET", result)
        self.assertIn("Node.js", result)


class TestNormalizeWhitespace(unittest.TestCase):
    """Tests for normalize_whitespace()."""

    def test_multiple_spaces(self):
        self.assertEqual(normalize_whitespace("Python   FastAPI"), "Python FastAPI")

    def test_tabs(self):
        self.assertEqual(normalize_whitespace("Python\t\tFastAPI"), "Python FastAPI")

    def test_newlines(self):
        self.assertEqual(normalize_whitespace("Python\n\nFastAPI"), "Python FastAPI")

    def test_mixed_whitespace(self):
        self.assertEqual(normalize_whitespace("  Python  \n  FastAPI  "), "Python FastAPI")

    def test_leading_trailing_stripped(self):
        result = normalize_whitespace("   hello world   ")
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        self.assertEqual(normalize_whitespace(""), "")

    def test_only_whitespace(self):
        self.assertEqual(normalize_whitespace("   \n\t  "), "")


# ---------------------------------------------------------------------------
# Full Pipeline Tests (clean())
# ---------------------------------------------------------------------------

class TestCleanPipeline(unittest.TestCase):
    """End-to-end tests for the main clean() function."""

    def test_clean_html_job_description(self):
        """Real-world HTML job description should be fully cleaned."""
        raw = (
            "<div><h2>Job Requirements</h2>"
            "<ul>"
            "<li>Proficiency in <b>Python</b> and <b>FastAPI</b></li>"
            "<li>Experience with <b>Docker</b> &amp; Kubernetes</li>"
            "<li>PostgreSQL knowledge required</li>"
            "</ul>"
            "<p>Apply at https://careers.example.com or email hr@company.com</p>"
            "</div>"
        )
        result = clean(raw)

        # Skills preserved
        self.assertIn("Python", result)
        self.assertIn("FastAPI", result)
        self.assertIn("Docker", result)
        self.assertIn("Kubernetes", result)
        self.assertIn("PostgreSQL", result)

        # HTML removed
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)

        # URL removed
        self.assertNotIn("careers.example.com", result)

        # Email removed
        self.assertNotIn("hr@company.com", result)

        # Entity decoded
        self.assertIn("&", result)   # &amp; decoded to &

    def test_clean_preserves_skill_punctuation(self):
        """C++, C#, React.js, ASP.NET, Node.js must survive cleaning."""
        raw = "Required: C++, C#, React.js, ASP.NET Core, Node.js, scikit-learn"
        result = clean(raw)
        for skill in ["C++", "C#", "React.js", "ASP.NET", "Node.js", "scikit-learn"]:
            self.assertIn(skill, result, f"Skill '{skill}' was damaged by cleaner")

    def test_clean_none_returns_empty(self):
        """None input must return empty string, not raise."""
        self.assertEqual(clean(None), "")

    def test_clean_empty_string_returns_empty(self):
        self.assertEqual(clean(""), "")

    def test_clean_whitespace_only_returns_empty(self):
        self.assertEqual(clean("   \n\t  "), "")

    def test_clean_integer_input_handled(self):
        """Non-string types should be handled gracefully (no crash)."""
        result = clean(12345)
        self.assertIsInstance(result, str)

    def test_clean_already_clean_text_unchanged(self):
        """Already clean text should pass through with only whitespace normalisation."""
        text = "We need Python FastAPI and Docker skills"
        result = clean(text)
        for word in ["Python", "FastAPI", "Docker"]:
            self.assertIn(word, result)

    def test_clean_bullet_list_job_description(self):
        """Bullet-point job descriptions (common format) should clean correctly."""
        raw = (
            "Requirements:\n"
            "• 3+ years of Python experience\n"
            "• Knowledge of React.js or Vue.js\n"
            "• Familiarity with Docker & Kubernetes\n"
            "— Nice to have: Go or Rust\n"
        )
        result = clean(raw)
        for skill in ["Python", "React.js", "Vue.js", "Docker", "Kubernetes"]:
            self.assertIn(skill, result, f"Skill '{skill}' lost in bullet cleaning")
        self.assertNotIn("•", result)

    def test_clean_divider_noise(self):
        """Section dividers like ======= should be stripped."""
        raw = "====== Requirements ======\nPython, FastAPI, Docker\n======"
        result = clean(raw)
        self.assertIn("Python", result)
        self.assertNotIn("======", result)

    def test_clean_output_is_single_line_like(self):
        """Output should have no leading/trailing whitespace."""
        raw = "  <p>Python developer</p>  "
        result = clean(raw)
        self.assertEqual(result, result.strip())

    def test_clean_does_not_lowercase(self):
        """clean() must preserve original casing — NOT lowercase."""
        raw = "We need React.js, FastAPI, and Docker"
        result = clean(raw)
        self.assertIn("React.js", result)
        self.assertIn("FastAPI", result)
        self.assertIn("Docker", result)
        # Confirm it's NOT lowercased
        self.assertNotEqual(result, result.lower())


# ---------------------------------------------------------------------------
# Batch Tests (clean_batch())
# ---------------------------------------------------------------------------

class TestCleanBatch(unittest.TestCase):
    """Tests for clean_batch()."""

    def test_batch_returns_list(self):
        result = clean_batch(["Python dev", "FastAPI engineer"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_batch_none_becomes_empty(self):
        result = clean_batch(["Python dev", None, "Docker expert"])
        self.assertEqual(result[1], "")

    def test_batch_preserves_order(self):
        inputs = ["Python", "FastAPI", "Docker"]
        result = clean_batch(inputs)
        for i, skill in enumerate(inputs):
            self.assertIn(skill, result[i])

    def test_batch_empty_list(self):
        self.assertEqual(clean_batch([]), [])

    def test_batch_all_none(self):
        result = clean_batch([None, None, None])
        self.assertEqual(result, ["", "", ""])

    def test_batch_html_inputs(self):
        inputs = [
            "<p>Python &amp; FastAPI</p>",
            "<b>Docker</b> and Kubernetes",
        ]
        result = clean_batch(inputs)
        self.assertIn("Python", result[0])
        self.assertIn("FastAPI", result[0])
        self.assertIn("Docker", result[1])
        for r in result:
            self.assertNotIn("<", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
