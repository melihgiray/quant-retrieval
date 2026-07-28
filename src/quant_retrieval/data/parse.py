"""Turn the dump's XML into typed tables.

Posts.xml holds questions and answers in one file, told apart by PostTypeId
(1 is a question, 2 is an answer). The file is a few hundred MB once
unpacked, so we stream it with iterparse and drop each row as we go instead
of building a tree in memory.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

QUESTION_TYPE_ID = "1"
ANSWER_TYPE_ID = "2"

# Duplicate edges matter for splitting. Stack Exchange writes 1 for a plain
# "linked" reference and 3 for "this is a duplicate of that".
LINK_TYPE_DUPLICATE = 3

_TAG_PATTERN = re.compile(r"<([^<>]+)>")


def iter_rows(xml_path: Path) -> Iterator[dict[str, str]]:
    """Yield one dict per <row> element, with bounded memory."""
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "row":
            yield dict(elem.attrib)
            root.clear()


def parse_tags(raw: str | None) -> list[str]:
    """Read a Tags attribute.

    Older dumps write `<option-pricing><black-scholes>`, newer ones write
    `|option-pricing|black-scholes|`. Both show up depending on the dump date,
    so handle each.
    """
    if not raw:
        return []
    if raw.startswith("|"):
        return [tag for tag in raw.split("|") if tag]
    return _TAG_PATTERN.findall(raw)


def _to_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def parse_posts(xml_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split Posts.xml into a questions table and an answers table."""
    questions: list[dict] = []
    answers: list[dict] = []

    for row in iter_rows(xml_path):
        post_type = row.get("PostTypeId")
        if post_type == QUESTION_TYPE_ID:
            questions.append(
                {
                    "question_id": int(row["Id"]),
                    "title": row.get("Title", ""),
                    "body_html": row.get("Body", ""),
                    "tags": parse_tags(row.get("Tags")),
                    "score": int(row.get("Score", 0)),
                    "view_count": _to_int(row.get("ViewCount")),
                    "answer_count": int(row.get("AnswerCount", 0)),
                    "accepted_answer_id": _to_int(row.get("AcceptedAnswerId")),
                    "creation_date": row["CreationDate"],
                    "owner_user_id": _to_int(row.get("OwnerUserId")),
                }
            )
        elif post_type == ANSWER_TYPE_ID:
            answers.append(
                {
                    "answer_id": int(row["Id"]),
                    "question_id": int(row["ParentId"]),
                    "body_html": row.get("Body", ""),
                    "score": int(row.get("Score", 0)),
                    "creation_date": row["CreationDate"],
                    "owner_user_id": _to_int(row.get("OwnerUserId")),
                }
            )

    questions_df = pd.DataFrame(questions)
    answers_df = pd.DataFrame(answers)
    for frame in (questions_df, answers_df):
        if not frame.empty:
            frame["creation_date"] = pd.to_datetime(frame["creation_date"])
    return questions_df, answers_df


def parse_post_links(xml_path: Path) -> pd.DataFrame:
    """Read PostLinks.xml. One row per edge between two posts."""
    links = [
        {
            "post_id": int(row["PostId"]),
            "related_post_id": int(row["RelatedPostId"]),
            "link_type_id": int(row["LinkTypeId"]),
        }
        for row in iter_rows(xml_path)
    ]
    return pd.DataFrame(links)
