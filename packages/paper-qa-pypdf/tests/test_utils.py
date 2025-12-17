import operator

from paperqa_pypdf.utils import cluster_bboxes


def test_cluster_bboxes_empty() -> None:
    assert cluster_bboxes([]) == []  # noqa: FURB115


def test_cluster_bboxes_single() -> None:
    bbox = (10.0, 20.0, 30.0, 40.0)
    results = cluster_bboxes([bbox])
    assert len(results) == 1
    assert results[0] == bbox


def test_cluster_bboxes_no_overlap_far_apart() -> None:
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (100.0, 100.0, 110.0, 110.0)
    results = cluster_bboxes([bbox1, bbox2], tolerance=50)  # noqa: FURB120
    assert len(results) == 2
    assert set(results) == {bbox1, bbox2}


def test_cluster_bboxes_overlapping() -> None:
    bbox1 = (0.0, 0.0, 20.0, 20.0)
    bbox2 = (10.0, 10.0, 30.0, 30.0)
    results = cluster_bboxes([bbox1, bbox2], tolerance=50)  # noqa: FURB120
    assert len(results) == 1
    assert results[0] == (0.0, 0.0, 30.0, 30.0), "Merged bbox should be the union"


def test_cluster_bboxes_within_tolerance() -> None:
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (15.0, 0.0, 25.0, 10.0)  # 5 units gap in x
    results = cluster_bboxes([bbox1, bbox2], tolerance=10)
    assert len(results) == 1
    assert results[0] == (0.0, 0.0, 25.0, 10.0)


def test_cluster_bboxes_outside_tolerance() -> None:
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (25.0, 0.0, 35.0, 10.0)  # 15 units gap in x
    results = cluster_bboxes([bbox1, bbox2], tolerance=10)
    assert len(results) == 2


def test_cluster_bboxes_chain_clustering() -> None:
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (15.0, 0.0, 25.0, 10.0)  # Near bbox1
    bbox3 = (30.0, 0.0, 40.0, 10.0)  # Near bbox2, far from bbox1
    results = cluster_bboxes([bbox1, bbox2, bbox3], tolerance=10)
    assert len(results) == 1
    assert results[0] == (0.0, 0.0, 40.0, 10.0)


def test_cluster_bboxes_multiple_clusters() -> None:
    # Cluster 1: top-left
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (5.0, 5.0, 15.0, 15.0)
    # Cluster 2: bottom-right (far away)
    bbox3 = (100.0, 100.0, 110.0, 110.0)
    bbox4 = (105.0, 105.0, 115.0, 115.0)

    results = cluster_bboxes([bbox1, bbox2, bbox3, bbox4], tolerance=20)
    assert len(results) == 2
    # Sort by x0 to make comparison deterministic
    result_sorted = sorted(results, key=operator.itemgetter(0))
    assert result_sorted[0] == (0.0, 0.0, 15.0, 15.0)
    assert result_sorted[1] == (100.0, 100.0, 115.0, 115.0)


def test_cluster_bboxes_vertical_proximity() -> None:
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (0.0, 15.0, 10.0, 25.0)  # 5 units gap in y
    results = cluster_bboxes([bbox1, bbox2], tolerance=10)
    assert len(results) == 1
    assert results[0] == (0.0, 0.0, 10.0, 25.0)


def test_cluster_bboxes_diagonal_proximity() -> None:
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    # Diagonal: 5 units in x, 5 units in y - should cluster with tolerance=10
    bbox2 = (15.0, 15.0, 25.0, 25.0)
    results = cluster_bboxes([bbox1, bbox2], tolerance=10)
    assert len(results) == 1


def test_cluster_bboxes_zero_tolerance() -> None:
    bbox1 = (0.0, 0.0, 10.0, 10.0)
    bbox2 = (10.0, 0.0, 20.0, 10.0)  # Touching but not overlapping
    results = cluster_bboxes([bbox1, bbox2], tolerance=0)
    assert len(results) == 1  # Touching counts as 0 distance

    bbox3 = (0.0, 0.0, 10.0, 10.0)
    bbox4 = (11.0, 0.0, 21.0, 10.0)  # 1 unit gap
    results = cluster_bboxes([bbox3, bbox4], tolerance=0)
    assert len(results) == 2


def test_cluster_bboxes_zero_tolerance_float() -> None:
    bbox1 = (0.5, 0.5, 10.5, 10.5)
    bbox2 = (10.5, 0.5, 20.5, 10.5)  # Touching
    results = cluster_bboxes([bbox1, bbox2], tolerance=0)
    assert len(results) == 1
    assert results[0] == (0.5, 0.5, 20.5, 10.5)
