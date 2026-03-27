from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.step1_shared import select_exact_minimum_vertex_cover


def _bruteforce_expected_cover(
    weighted_edges: list[tuple[int, int, float]],
) -> list[int]:
    vertices = sorted({vertex for edge in weighted_edges for vertex in edge[:2]})
    incident_counts = {vertex: 0 for vertex in vertices}
    incident_severity = {vertex: 0.0 for vertex in vertices}

    normalized_edges: list[tuple[int, int, float]] = []
    for vertex_a, vertex_b, severity in weighted_edges:
        if vertex_a == vertex_b:
            continue
        normalized_edges.append((vertex_a, vertex_b, severity))
        incident_counts[vertex_a] += 1
        incident_counts[vertex_b] += 1
        incident_severity[vertex_a] += severity
        incident_severity[vertex_b] += severity

    order_index = {vertex: idx for idx, vertex in enumerate(vertices)}

    def rank(selected_vertices: tuple[int, ...]) -> tuple[int, int, float, tuple[int, ...]]:
        return (
            len(selected_vertices),
            -sum(incident_counts[vertex] for vertex in selected_vertices),
            -sum(incident_severity[vertex] for vertex in selected_vertices),
            tuple(order_index[vertex] for vertex in selected_vertices),
        )

    best_cover: tuple[int, ...] | None = None
    for cover_size in range(len(vertices) + 1):
        for selected_vertices in combinations(vertices, cover_size):
            selected = set(selected_vertices)
            if all(
                (vertex_a in selected) or (vertex_b in selected)
                for vertex_a, vertex_b, _ in normalized_edges
            ):
                if best_cover is None or rank(selected_vertices) < rank(best_cover):
                    best_cover = selected_vertices
        if best_cover is not None:
            break

    return list(best_cover or ())


def test_select_exact_minimum_vertex_cover_matches_bruteforce_on_multiselect_case() -> None:
    weighted_edges = [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 4, 1.0),
        (2, 4, 1.0),
        (3, 4, 1.0),
        (1, 5, 1.0),
        (2, 5, 1.0),
    ]

    expected = _bruteforce_expected_cover(weighted_edges)
    selected = select_exact_minimum_vertex_cover(weighted_edges, lambda vertex: (vertex,))

    assert selected == expected
    assert len(selected) == 3


def test_select_exact_minimum_vertex_cover_uses_severity_to_break_ties() -> None:
    weighted_edges = [
        (0, 1, 10.0),
        (0, 2, 1.0),
        (1, 2, 1.0),
    ]

    selected = select_exact_minimum_vertex_cover(weighted_edges, lambda vertex: (vertex,))

    assert selected == [0, 1]


def test_select_exact_minimum_vertex_cover_uses_order_key_for_final_tie_break() -> None:
    weighted_edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 0, 1.0),
    ]

    selected = select_exact_minimum_vertex_cover(weighted_edges, lambda vertex: (vertex,))

    assert selected == [0, 2]
