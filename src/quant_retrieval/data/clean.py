"""Turn post HTML into the plain text the models actually see.

Two things here are specific to this corpus and worth stating. Math is written
as MathJax, so it survives as literal text like $\\sigma^2$ inside the HTML and
we leave it alone. Code sits in <pre><code> blocks and carries real signal on a
site where half the answers are implementations, so it stays too, fenced, rather
than being flattened into the surrounding prose.

Links lose their href and keep their anchor text. A bare URL adds tokens and no
meaning to an embedding.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString

# Paragraph level tags get a blank line around them. Line level tags only end
# the current line, otherwise a five item list turns into five blank lines.
PARAGRAPH_TAGS = frozenset({"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "hr"})
LINE_TAGS = frozenset({"br", "li", "tr"})

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def html_to_text(html: str) -> str:
    """Flatten one post body to text, keeping code blocks and math."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["img", "script", "style"]):
        tag.decompose()

    # Fence code blocks before the generic walk, so their newlines are not
    # collapsed with the prose around them.
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        source = (code or pre).get_text()
        pre.replace_with(NavigableString(f"\n```\n{source.strip()}\n```\n"))

    for code in soup.find_all("code"):
        code.replace_with(NavigableString(f"`{code.get_text()}`"))

    for tag in soup.find_all(list(PARAGRAPH_TAGS)):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))

    for tag in soup.find_all(list(LINE_TAGS)):
        tag.insert_after(NavigableString("\n"))

    text = soup.get_text()
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def build_query_text(title: str, body_html: str) -> str:
    """A question becomes its title and then its body.

    The title is the strongest single signal on Stack Exchange, so it leads. If
    the body is truncated later by the tokenizer, the title is what survives.
    """
    body = html_to_text(body_html)
    title = (title or "").strip()
    if not body:
        return title
    return f"{title}\n\n{body}"
