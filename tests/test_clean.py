from quant_retrieval.data.clean import build_query_text, html_to_text


def test_paragraphs_become_blank_line_separated_text():
    html = "<p>First point.</p><p>Second point.</p>"
    assert html_to_text(html) == "First point.\n\nSecond point."


def test_inline_math_survives_untouched():
    # MathJax is literal text in the dump, and it is most of what makes this
    # corpus specific. Losing it would flatten half the vocabulary.
    html = "<p>Assume $dS_t = \\mu S_t dt + \\sigma S_t dW_t$ holds.</p>"
    assert "$dS_t = \\mu S_t dt + \\sigma S_t dW_t$" in html_to_text(html)


def test_display_math_survives():
    html = "<p>$$\\int_0^T \\sigma^2 ds$$</p>"
    assert "$$\\int_0^T \\sigma^2 ds$$" in html_to_text(html)


def test_code_blocks_are_fenced_and_keep_their_line_breaks():
    html = "<p>Try:</p><pre><code>import numpy as np\nnp.std(x)\n</code></pre>"
    text = html_to_text(html)
    assert "```\nimport numpy as np\nnp.std(x)\n```" in text


def test_inline_code_is_backticked():
    html = "<p>Call <code>np.std</code> on it.</p>"
    assert html_to_text(html) == "Call `np.std` on it."


def test_links_keep_their_text_and_drop_the_url():
    html = '<p>See <a href="https://example.com/very/long/path">Hull chapter 4</a>.</p>'
    text = html_to_text(html)
    assert "Hull chapter 4" in text
    assert "example.com" not in text


def test_images_are_removed():
    html = '<p>Result:</p><img src="https://i.stack.imgur.com/x.png" alt="plot">'
    assert html_to_text(html) == "Result:"


def test_list_items_are_separated():
    html = "<ul><li>delta</li><li>gamma</li></ul>"
    assert html_to_text(html) == "delta\ngamma"


def test_entities_are_unescaped():
    html = "<p>risk &amp; return &lt; 1</p>"
    assert html_to_text(html) == "risk & return < 1"


def test_empty_body_is_empty_string():
    assert html_to_text("") == ""


def test_query_text_leads_with_the_title():
    text = build_query_text("Pricing a swaption", "<p>How do I do it?</p>")
    assert text == "Pricing a swaption\n\nHow do I do it?"


def test_query_text_with_no_body_is_just_the_title():
    assert build_query_text("Pricing a swaption", "") == "Pricing a swaption"
