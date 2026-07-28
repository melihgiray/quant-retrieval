from pathlib import Path

import pandas as pd
import pytest

from quant_retrieval.data.parse import parse_post_links, parse_posts, parse_tags

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def parsed():
    return parse_posts(FIXTURES / "posts_sample.xml")


def test_questions_and_answers_are_separated(parsed):
    questions, answers = parsed
    assert list(questions["question_id"]) == [1, 4]
    assert list(answers["answer_id"]) == [2, 3]


def test_post_types_other_than_questions_and_answers_are_dropped(parsed):
    questions, answers = parsed
    # Row 5 is a tag wiki (PostTypeId 5) and belongs in neither table.
    assert 5 not in set(questions["question_id"])
    assert 5 not in set(answers["answer_id"])


def test_answers_keep_their_parent_question(parsed):
    _, answers = parsed
    assert set(answers.loc[answers["question_id"] == 1, "answer_id"]) == {2, 3}


def test_accepted_answer_is_read_and_missing_stays_missing(parsed):
    questions, _ = parsed
    by_id = questions.set_index("question_id")
    assert by_id.loc[1, "accepted_answer_id"] == 3
    assert pd.isna(by_id.loc[4, "accepted_answer_id"])


def test_html_body_is_unescaped_by_the_xml_reader(parsed):
    questions, _ = parsed
    body = questions.set_index("question_id").loc[1, "body_html"]
    assert "<code>swaption</code>" in body


def test_dates_become_timestamps(parsed):
    questions, _ = parsed
    assert questions["creation_date"].dtype.kind == "M"
    assert questions["creation_date"].min() == pd.Timestamp("2011-01-31 21:00:00")


def test_missing_owner_is_none_not_zero(parsed):
    questions, _ = parsed
    # Question 4 has no OwnerUserId, which means the account was deleted.
    assert pd.isna(questions.set_index("question_id").loc[4, "owner_user_id"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("|option-pricing|swaption|", ["option-pricing", "swaption"]),
        ("<volatility><estimation>", ["volatility", "estimation"]),
        ("|single|", ["single"]),
        ("", []),
        (None, []),
    ],
)
def test_parse_tags_handles_both_dump_formats(raw, expected):
    assert parse_tags(raw) == expected


def test_both_tag_formats_survive_a_real_parse(parsed):
    questions, _ = parsed
    by_id = questions.set_index("question_id")
    assert by_id.loc[1, "tags"] == ["option-pricing", "swaption"]
    assert by_id.loc[4, "tags"] == ["volatility", "estimation"]


def test_post_links_keep_the_link_type():
    links = parse_post_links(FIXTURES / "post_links_sample.xml")
    assert len(links) == 2
    duplicates = links[links["link_type_id"] == 3]
    assert list(duplicates["post_id"]) == [4]
    assert list(duplicates["related_post_id"]) == [1]
