# paper-qa-nemotron

<!-- pyml disable-num-lines 5 line-length -->

[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)](https://github.com/Future-House/paper-qa/tree/main/packages/paper-qa-nemotron)
[![tests](https://github.com/Future-House/paper-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/Future-House/paper-qa)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![PyPI Python Versions](https://img.shields.io/pypi/pyversions/paper-qa-nemotron)

PDF reading code backed by
[Nvidia's nemotron-parse VLM](https://build.nvidia.com/nvidia/nemotron-parse).

For more info on nemotron-parse, check out:

<!-- pyml disable-num-lines 9 line-length -->

- Technical blog:
  <https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/>
- Hugging Face weights: <https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1>
- Model card: <https://build.nvidia.com/nvidia/nemotron-parse/modelcard>
- API docs:
  <https://docs.nvidia.com/nim/vision-language-models/1.5.0/examples/nemotron-parse/overview.html#nemotron-parse-overview>
- Cookbook:
  <https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Parse-v1.1/build_general_usage_cookbook.ipynb>
- AWS Marketplace:
  <https://aws.amazon.com/marketplace/pp/prodview-ny2ngku2i4ge6>

## Installation

```bash
pip install paper-qa[nemotron]
# Or
pip install paper-qa-nemotron
```

If you want to prompt nemotron-parse hosted on AWS SageMaker:

```bash
pip install paper-qa-nemotron[sagemaker]
```

## Getting Started

To use nemotron-parse via the Nvidia API,
set the `NVIDIA_API_KEY` environment variable.

Then to directly access the reader:

```python
from paperqa.types import ParsedText
from paperqa_nemotron import parse_pdf_to_pages

async def main(pdf_path) -> ParsedText:
    return await parse_pdf_to_pages(pdf_path)
```

Or use the reader within PaperQA:

```python
from paperqa import Docs, PQASession, Settings

from paperqa_nemotron import parse_pdf_to_pages


async def main(pdf_path, question: str | PQASession) -> PQASession:
    settings = Settings(parsing={"parse_pdf": parse_pdf_to_pages})
    docs = Docs()
    await docs.aadd(pdf_path, settings=settings)
    return await docs.aquery(question, settings=settings)
```
