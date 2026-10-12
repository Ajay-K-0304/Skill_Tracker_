"""
text_cleaner.py — Raw Text Cleaning Pipeline
=============================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  Accepts raw job description text (possibly with HTML, URLs, emails,
  special characters) and returns clean, normalised plain text ready
  for the spaCy NER pipeline.

Pipeline order (applied by `clean()`):
  1. Guard against None / non-string input
  2. Decode HTML entities   (&amp; → &, &lt; → <, &#39; → ')
  3. Strip HTML tags        (<b>, <p>, <br/>, ...)
  4. Strip URLs             (http/https/ftp/www...)
  5. Strip email addresses
  6. Strip bullet / list markers and lone special chars
  7. Normalize whitespace   (tabs, newlines, multiple spaces → single space)
  8. Strip leading/trailing whitespace

NOTE:
  - We do NOT lowercase here — case is preserved so the EntityRuler
    can match mixed-case variants ("React.js", "FastAPI") from the taxonomy.
    Case-insensitive lookup is handled downstream in the normalizer.
  - We do NOT remove stopwords — spaCy handles that in the NER stage.
  - This module has ZERO imports outside Python stdlib + config.

Cross-team compatibility:
  - Member 1 sends us `cleaned_description` from the DB, but we apply
    this cleaner as a safety net in case residual HTML/noise slips through.
  - The output of `clean()` is the direct input to `ner_pipeline.run()`.
"""

import html
import logging
import re
from html.parser import HTMLParser
from typing import Optional

from services.nlp_engine.config import LOG_LEVEL

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled Regex Patterns (compiled once at module load for performance)
# ---------------------------------------------------------------------------

# URLs: http, https, ftp, and bare www. links
_URL_PATTERN = re.compile(
    r"https?://\S+|ftp://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

# Email addresses
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    flags=re.IGNORECASE,
)

# Bullet / list markers: •, ◦, ▪, *, -, –, —, ·, ○, ► at start of content
_BULLET_PATTERN = re.compile(
    r"(?:^|\s)[•◦▪\*\-–—·○►]+(?=\s|$)",
    flags=re.MULTILINE,
)

# Sequences of 2+ non-alphanumeric chars (e.g. "====", "----", ">>>")
# but NOT single punctuation inside words (preserves "React.js", "C++")
_NOISE_PATTERN = re.compile(
    r"[^\w\s.+#/\-]{2,}",
)

# Multiple whitespace (spaces, tabs, newlines) → single space
_WHITESPACE_PATTERN = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# HTML Tag Stripper (stdlib HTMLParser-based — no BeautifulSoup)
# ---------------------------------------------------------------------------

class _HTMLTagStripper(HTMLParser):
    """
    Minimal HTMLParser subclass that discards all tags and collects only
    the text content between them. Handles malformed/partial HTML gracefully.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)  # we handle entities ourselves
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """Called for text nodes between tags."""
        self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        """Called for named HTML entities like &amp;"""
        # Let html.unescape handle them in a separate pass
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        """Called for numeric HTML entities like &#39;"""
        self._parts.append(f"&#{name};")

    def get_text(self) -> str:
        return " ".join(self._parts)

    @classmethod
    def strip(cls, text: str) -> str:
        """
        Strip all HTML tags from text, returning plain text with spacing.

        Args:
            text: Raw string potentially containing HTML markup.

        Returns:
            Plain text string with tags removed and replaced by spaces.
        """
        stripper = cls()
        try:
            stripper.feed(text)
            stripper.close()
            return stripper.get_text()
        except Exception:
            # Fallback: regex-based stripping for severely malformed HTML
            logger.debug("[text_cleaner] HTMLParser failed, falling back to regex tag strip.")
            return re.sub(r"<[^>]+>", " ", text)


# ---------------------------------------------------------------------------
# Individual Cleaning Steps (public — usable independently for testing)
# ---------------------------------------------------------------------------

def decode_html_entities(text: str) -> str:
    """
    Decode HTML character references and entities.

    Converts &amp; → &, &lt; → <, &gt; → >, &#39; → ', &nbsp; → space, etc.

    Args:
        text: Input string possibly containing HTML entities.

    Returns:
        String with all HTML entities decoded to their Unicode equivalents.

    Example:
        >>> decode_html_entities("Python &amp; FastAPI &lt;3")
        'Python & FastAPI <3'
    """
    return html.unescape(text)


def strip_html_tags(text: str) -> str:
    """
    Remove all HTML/XML tags from text using stdlib HTMLParser.

    Falls back to regex stripping if the parser encounters severely
    malformed HTML. Tags are replaced with a space to prevent word merging.

    Args:
        text: Input string potentially containing HTML markup.

    Returns:
        Plain text with all tags removed.

    Example:
        >>> strip_html_tags("<p>We need <b>Python</b> developers.</p>")
        'We need Python developers.'
    """
    return _HTMLTagStripper.strip(text)


def strip_urls(text: str) -> str:
    """
    Remove URLs (http/https/ftp/www) from text.

    Args:
        text: Input string potentially containing URLs.

    Returns:
        String with all URLs replaced by a single space.

    Example:
        >>> strip_urls("Apply at https://jobs.example.com/python-dev")
        'Apply at '
    """
    return _URL_PATTERN.sub(" ", text)


def strip_emails(text: str) -> str:
    """
    Remove email addresses from text.

    Args:
        text: Input string potentially containing email addresses.

    Returns:
        String with all email addresses replaced by a single space.

    Example:
        >>> strip_emails("Send CV to hr@company.com for the role.")
        'Send CV to  for the role.'
    """
    return _EMAIL_PATTERN.sub(" ", text)


def strip_bullet_markers(text: str) -> str:
    """
    Remove bullet/list markers (•, ◦, *, -, –, —, ·, ►) from text.

    These appear frequently in job postings and carry no skill signal.

    Args:
        text: Input string potentially containing bullet markers.

    Returns:
        String with bullet markers removed.
    """
    return _BULLET_PATTERN.sub(" ", text)


def strip_noise_chars(text: str) -> str:
    """
    Remove sequences of 2+ non-alphanumeric characters that carry no
    semantic content (e.g. "====", ">>>>", "---").

    Preserves single punctuation within words to keep skill names like
    "React.js", "C++", "C#", "ASP.NET" intact.

    Args:
        text: Input string potentially containing noise character sequences.

    Returns:
        String with noise sequences replaced by a single space.
    """
    return _NOISE_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace sequences (spaces, tabs, newlines, carriage
    returns) into a single space, then strip leading/trailing whitespace.

    Args:
        text: Input string with potentially irregular whitespace.

    Returns:
        Whitespace-normalized string.

    Example:
        >>> normalize_whitespace("Python   \\n  and   FastAPI")
        'Python and FastAPI'
    """
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Main Public Entry Point
# ---------------------------------------------------------------------------

def clean(text: Optional[str]) -> str:
    """
    Full cleaning pipeline for a single job description text.

    Applies all cleaning steps in the correct order:
      1. None / type guard
      2. HTML entity decoding
      3. HTML tag stripping
      4. URL removal
      5. Email removal
      6. Bullet marker removal
      7. Noise character removal
      8. Whitespace normalization

    Args:
        text: Raw job description string (may contain HTML, URLs, None).

    Returns:
        Clean plain-text string ready for the spaCy NER pipeline.
        Returns an empty string for None, non-string, or whitespace-only input.

    Example:
        >>> clean("<p>We need <b>React.js</b> &amp; Python. Apply: jobs@co.com</p>")
        'We need React.js & Python. Apply:'
    """
    # --- Step 1: Guard ---
    if text is None:
        logger.debug("[text_cleaner] Received None input, returning empty string.")
        return ""

    if not isinstance(text, str):
        logger.warning(
            f"[text_cleaner] Expected str, got {type(text).__name__}. "
            "Attempting str() conversion."
        )
        try:
            text = str(text)
        except Exception:
            return ""

    # Early exit for blank input
    if not text.strip():
        return ""

    # --- Step 2: Decode HTML entities ---
    text = decode_html_entities(text)

    # --- Step 3: Strip HTML tags ---
    text = strip_html_tags(text)

    # --- Step 4: Strip URLs ---
    text = strip_urls(text)

    # --- Step 5: Strip emails ---
    text = strip_emails(text)

    # --- Step 6: Strip bullet markers ---
    text = strip_bullet_markers(text)

    # --- Step 7: Strip noise characters ---
    text = strip_noise_chars(text)

    # --- Step 8: Normalize whitespace ---
    text = normalize_whitespace(text)

    return text


def clean_batch(texts: list[Optional[str]]) -> list[str]:
    """
    Apply `clean()` to a list of job description strings.

    Used by the NER pipeline to process a batch of job postings from
    Member 1's data in one call.

    Args:
        texts: List of raw job description strings (may contain None values).

    Returns:
        List of cleaned strings in the same order as input.
        None values and invalid inputs become empty strings.

    Example:
        >>> clean_batch(["<p>Python dev</p>", None, "FastAPI engineer"])
        ['Python dev', '', 'FastAPI engineer']
    """
    return [clean(t) for t in texts]
