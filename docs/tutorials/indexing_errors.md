# Continuing indexing after document errors

By default, indexing stops when a document raises an unexpected error, such as
an API failure or timeout. Parsing failures represented by `ValueError` or
`ImpossibleParsingError` are already skipped.

To finish processing the other files even when a document fails, enable
`agent.index.continue_on_error`:

```bash
pqa --agent.index.continue_on_error true index ./papers
```

The same option works in Python:

```python
from paperqa import Settings
from paperqa.agents import get_directory_index
from paperqa.settings import AgentSettings, IndexSettings

settings = Settings(
    agent=AgentSettings(
        index=IndexSettings(
            paper_directory="./papers",
            continue_on_error=True,
        )
    )
)
# Configure your chat, embedding and parsing models as usual.
index = await get_directory_index(settings=settings)
report = index.indexing_report
if report is not None:
    print(report.model_dump())
```

After a build completes, the log includes the number of newly indexed files,
successfully indexed files skipped, and failed files. If there are failures,
each filename and its exception type and message appear in the final warning
summary. For example:

```text
Indexing finished with 1 failed document(s): 4 indexed, 2 skipped.
Failed document broken.pdf: TimeoutError: Request timed out
```

`index.indexing_report.failures` provides the same information for Python callers.
The report describes the current build and is not persisted. Loading an existing
index with `build=False` does not produce a build report.

Failed documents retain their existing `ERROR` marker in the persisted index.
Subsequent builds still skip those files; this option does not automatically
retry them. The summary includes these files with `previously_failed=True` and
notes that their original error is unavailable. They do not count as successfully
skipped files. A report with failures describes an incomplete index, even if the
build returns normally; if every document fails, the indexed count is zero.

This option applies to document preparation, parsing, metadata lookup, and model
calls. Index storage errors and task cancellation still interrupt the build.
An interrupted build does not emit the completed-build summary.
