import textwrap

import pytest

from paperqa.core import llm_parse_json


# test cases obtained from test_paperqa.py
@pytest.mark.parametrize(
    "example",
    [
        """Sure here is the json you asked for!

    {
    "example": "json"
    }

    Did you like it?""",
        '{"example": "json"}',
        """
```json
{
    "example": "json"
}
```

I have written the json you asked for.""",
        """

{
    "example": "json"
}

""",
    ],
)
def test_llm_parse_json(example: str) -> None:
    assert llm_parse_json(example) == {"example": "json"}


def test_llm_parse_json_newlines() -> None:
    """Make sure that newlines in json are preserved and escaped."""
    example = textwrap.dedent(
        """
        {
        "summary": "A line

        Another line",
        "relevance_score": 7
        }"""
    )
    assert llm_parse_json(example) == {
        "summary": "A line\n\nAnother line",
        "relevance_score": 7,
    }


# Additional cases
# Case 1: Removing Think Tags
@pytest.mark.parametrize(
    "example",
    [
        '<think> Thinking </think> I am here to help ```json {    "summary": "Lorem Ipsum",    "relevance_ score": 8 } ``` Hope this helps!'
    ],
)
def test_llm_parse_json_remove_think_tags(example: str) -> None:
    output = {"summary": "Lorem Ipsum", "relevance_score": 8}
    assert llm_parse_json(example) == output


# Case 2: JSON format with and without markdown code block
@pytest.mark.parametrize(
    "example",
    [
        '<think> Thinking </think> I am here to help ```json {    "summary": "Lorem Ipsum",    "relevance_ score": 8 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help {   "summary": "Lorem Ipsum",   "relevance_score": 8 } Hope this helps!',
    ],
)
def test_llm_parse_json_extract_json(example: str) -> None:
    output = {"summary": "Lorem Ipsum", "relevance_score": 8}
    assert llm_parse_json(example) == output


# Case 3: Relevance Score as float, string or fraction
@pytest.mark.parametrize(
    "example",
    [
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": 7.6 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": 8.4 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": "8" } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": "7.6" } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": "8/10" } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": "4/5" } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": 8/10 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help ```json {   "summary": "Lorem Ipsum",   "relevance_score": 4/5 } ``` Hope this helps!',
    ],
)
def test_llm_parse_json_relevance_score_int(example: str) -> None:
    output = {"summary": "Lorem Ipsum", "relevance_score": 8}
    assert llm_parse_json(example) == output


# Case 4: "relevance_score" key named incorrectly
@pytest.mark.parametrize(
    "example",
    [
        '<think> Thinking </think> I am here to help. ```json {    "summary": "Lorem Ipsum",    "relevance_-score": 8 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help. ```json {    "summary": "Lorem Ipsum",    "relevance_ score": 8 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help. ```json {    "summary": "Lorem Ipsum",    "score": 8 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help. ```json {    "summary": "Lorem Ipsum",    "relevance score": 8 } ``` Hope this helps!',
        '<think> Thinking </think> I am here to help. ```json {    "summary": "Lorem Ipsum",    "relevance": 8 } ``` Hope this helps!',
    ],
)
def test_llm_parse_json_relevance_score_key(example: str) -> None:
    output = {"summary": "Lorem Ipsum", "relevance_score": 8}
    assert llm_parse_json(example) == output


# Case 5: Broken JSON formatting - ','
@pytest.mark.parametrize(
    "example",
    [
        '<think> Thinking </think> I am here to help. {   "summary": "Lorem Ipsum",   "relevance_score": 8, } Hope this helps!',
        '<think> Thinking </think> I am here to help. {   "summary": "Lorem Ipsum", ,  "relevance_score": 8 } Hope this helps!',
        '<think> Thinking </think> I am here to help. { ,  "summary": "Lorem Ipsum",  "relevance_score": 8 } Hope this helps!',
    ],
)
def test_llm_parse_json_remove_extra_commas(example: str) -> None:
    output = {"summary": "Lorem Ipsum", "relevance_score": 8}
    assert llm_parse_json(example) == output


# Case 6: Non-JSON Response
@pytest.mark.parametrize(
    "example",
    [
        "<think> Thinking </think> Lorem Ipsum. Hope this helps!",
    ],
)
def test_llm_parse_json_nonjson_string(example: str) -> None:
    output = {"summary": "Lorem Ipsum. Hope this helps!"}
    assert llm_parse_json(example) == output
