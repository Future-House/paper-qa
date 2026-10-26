# paper-qa-mineru

<!-- pyml disable-num-lines 6 line-length -->

[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)](https://github.com/Future-House/paper-qa/tree/main/packages/paper-qa-mineru)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

PaperQA PDF reader that uses output already produced by
[MinerU](https://github.com/opendatalab/MinerU). It reads
`content_list.json` and the `images/` directory and does not call MinerU.

## Layout

The reader expects the MinerU output directory next to the PDF, named after
the PDF stem:

```text
papers/G7ZCDWQ4.pdf
papers/G7ZCDWQ4/content_list.json
papers/G7ZCDWQ4/images/
```

A missing `content_list.json` raises an error. The reader does not fall back
to another parser.

## Usage

```python
from paperqa import Settings
from paperqa_mineru import parse_pdf_to_pages

settings = Settings(parsing={"parse_pdf": parse_pdf_to_pages})
```

## Behavior

- Page numbers come from `page_idx`, converted to 1-based strings.
- `header`, `footer`, and `page_number` blocks are dropped from text.
- Tables contribute their `table_body` HTML to page text and to the media `text`.
- `image`, `chart`, and `table` blocks become `ParsedMedia` with the file bytes.
- Media coordinates are kept in `info["bbox"]`.

## Limits

- Only blocks present in `content_list.json` are read. Page text is not
  re-extracted from the PDF.
- Output quality depends on the MinerU run that produced the directory.
