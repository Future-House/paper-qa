from collections import defaultdict


def cluster_bboxes(
    bboxes: list[tuple[float, float, float, float]], tolerance: float = 50
) -> list[tuple[float, float, float, float]]:
    """Cluster nearby bounding boxes into regions using spatial proximity.

    Uses union-find to cluster bboxes based on the input tolerance,
    then computes a merged bounding box for each cluster.

    Args:
        bboxes: List of (x0, y0, x1, y1) bounding boxes.
        tolerance: Maximum distance (inclusive) between bboxes to consider them
            part of the same cluster.

    Returns:
        List of (x0, y0, x1, y1) merged bounding boxes for each cluster.
    """
    parent = list(range(len(bboxes)))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    # Cluster bboxes that are within tolerance distance
    for i, b1 in enumerate(bboxes):
        for j in range(i + 1, len(bboxes)):
            b2 = bboxes[j]
            # Distance is 0 if they overlap, otherwise there's a gap between them
            x_dist = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
            y_dist = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
            if x_dist <= tolerance and y_dist <= tolerance:
                union(i, j)

    # Group bboxes by cluster and compute merged bbox
    clusters: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for i, bbox in enumerate(bboxes):
        clusters[find(i)].append(bbox)
    return [
        (
            min(b[0] for b in cluster_bboxes),
            min(b[1] for b in cluster_bboxes),
            max(b[2] for b in cluster_bboxes),
            max(b[3] for b in cluster_bboxes),
        )
        for cluster_bboxes in clusters.values()
    ]
