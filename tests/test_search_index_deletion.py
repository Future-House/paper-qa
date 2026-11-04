from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tantivy import Document, Index, Query, SchemaBuilder

from paperqa.agents.search import SearchIndex

if TYPE_CHECKING:
    from tantivy import DocAddress, Searcher


def stored_records(index: Index) -> dict[str, dict[str, list[Any]]]:
    index.reload()
    searcher = index.searcher()
    records: dict[str, dict[str, list[Any]]] = {}
    if searcher.num_docs:
        for _, address in searcher.search(
            Query.all_query(), limit=searcher.num_docs
        ).hits:
            document = searcher.doc(address)
            location = document.get_first("file_location")
            assert isinstance(location, str)
            records[location] = document.to_dict()
    return records


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_index", [False, True])
@pytest.mark.parametrize("target", ["papers/last.txt", "x" * 80, "!!!"])
async def test_delete_stored_path(
    tmp_path: Path, existing_index: bool, target: str
) -> None:
    fields = ["file_location", "body", "title", "year"]
    documents = [
        {"file_location": target, "body": "deleted evidence", "title": "Target"},
        {
            "file_location": "keeper.txt",
            "body": "retained evidence",
            "title": "Keeper",
            "year": "2026",
        },
    ]
    index = SearchIndex(fields=fields, index_name="test", index_directory=tmp_path)
    if existing_index:
        # Populate the unchanged schema directly, as in an existing upstream index.
        schema = SchemaBuilder()
        for field in fields:
            schema.add_text_field(field, stored=True)
        native = Index(schema.build(), path=str(await index.index_filename))
        writer = native.writer()
        for document in documents:
            writer.add_document(Document.from_dict(document))
        writer.commit()
        writer.wait_merging_threads()
        del writer
    else:
        for document in documents:
            await index.add_document(document)
        await index.save_index()
    native = await index.index
    before = stored_records(native)
    assert set(before) == {target, "keeper.txt"}
    await index.delete_document(target)
    assert stored_records(native) == {"keeper.txt": before["keeper.txt"]}
    assert await index.count == 1
    await index.delete_document(target)
    await index.delete_document("missing.txt")
    assert stored_records(native) == {"keeper.txt": before["keeper.txt"]}
    await index.delete_document("keeper.txt")
    assert stored_records(native) == {}
    assert await index.count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_delete_preserves_colliding_stored_paths(
    tmp_path: Path, pending: bool
) -> None:
    index = SearchIndex(
        fields=["file_location", "body", "title", "year"],
        index_name="test",
        index_directory=tmp_path,
    )
    locations = ["a-b.txt", "a/b.txt", "A.B.txt", "other.md"]
    expected = {}
    for number, location in enumerate(locations):
        document = {
            "file_location": location,
            "body": f"Evidence {number}",
            "title": f"Title {number}",
            "year": str(2020 + number),
        }
        await index.add_document(document)
        if location != "a-b.txt":
            expected[location] = {key: [value] for key, value in document.items()}
    if not pending:
        await index.save_index()
    await index.remove_from_index("a-b.txt")
    await index.save_index()
    assert stored_records(await index.index) == expected
    assert await index.count == len(expected)
    assert set(await index.index_files) == set(expected)


@pytest.mark.asyncio
async def test_deletion_snapshot_holds_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = SearchIndex(index_name="test", index_directory=tmp_path)
    await index.add_document({"file_location": "a-b.txt", "body": "Target"})
    await index.add_document({"file_location": "a/b.txt", "body": "Keeper"})
    await index.save_index()
    competing_index = Index.open(str(await index.index_filename))
    original = index._get_deletion_query

    def locked_deletion_query(
        native: Index, location: str
    ) -> tuple[Query, Searcher, list[DocAddress]] | None:
        # A separate index writer must be excluded while the snapshot is selected.
        with pytest.raises(ValueError, match="LockBusy"):
            competing_index.writer()
        return original(native, location)

    monkeypatch.setattr(index, "_get_deletion_query", locked_deletion_query)
    await index.delete_document("a-b.txt")
    assert set(stored_records(competing_index)) == {"a/b.txt"}
    # The lock is released after committing deletion and restoration.
    writer = competing_index.writer()
    writer.rollback()
    writer.wait_merging_threads()
