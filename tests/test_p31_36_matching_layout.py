"""P31-36 layout regression tests for the Streamlit matching review."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tools" / "align_longitudinal_patient_v5_mapped.py"


def _source() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


def _function_source(name: str) -> str:
    source = _source()
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            end = getattr(node, "end_lineno", None)
            assert end is not None
            return "\n".join(lines[node.lineno - 1 : end])
    raise AssertionError(f"Function {name!r} was not found")


def test_p31_36_selected_matching_review_is_anchored_after_alignment_viewer() -> None:
    """The review card must be created after the alignment viewer and before details."""
    main = _function_source("main")
    viewer = main.index("_render_slice_viewer(")
    heading = main.index('st.header("Lesion Matching Review")', viewer)
    container = main.index("matching_visualisation = st.container(border=True)", heading)
    panel = main.index("_render_matching_panel(", container)

    assert viewer < heading < container < panel


def test_p31_36_matching_panel_routes_selected_event_into_review_container() -> None:
    panel = _function_source("_render_matching_panel")

    assert "visualisation_container=None" in panel
    assert "focus_renderer=None" in panel
    assert "with selected_visualisation:" in panel
    assert '"Select lesion correspondence / unmatched event"' in panel
    assert "focus_renderer(focus)" in panel


def test_p31_36_selected_images_and_match_evidence_stay_together() -> None:
    """IDs, selected images and matcher feature/cost evidence share the top review card."""
    panel = _function_source("_render_matching_panel")
    start = panel.index("with selected_visualisation:")
    ground_truth = panel.index("# Ground-truth agreement", start)
    review = panel[start:ground_truth]

    required = (
        'metric("BL lesion ID"',
        'metric("FU lesion ID"',
        'metric("Outcome"',
        '"Match cost"',
        "focus_renderer(focus)",
        '"Distance"',
        '"BL volume"',
        '"FU volume"',
        '"Size difference"',
        'with st.expander("Cost component details")',
    )
    for marker in required:
        assert marker in review, marker


def test_p31_36_existing_matching_and_evaluation_sections_are_preserved() -> None:
    """The layout move must not remove existing matching/evaluation interactions."""
    panel = _function_source("_render_matching_panel")

    for marker in (
        'st.markdown("#### Tracking summary")',
        'with st.expander("Matcher outcome counts")',
        'with st.expander(f"All matching events ({len(display_matches)})")',
        'st.subheader("Ground-truth evaluation")',
        "run_patient_match(",
        "_run_patient_evaluation(",
    ):
        assert marker in panel, marker
