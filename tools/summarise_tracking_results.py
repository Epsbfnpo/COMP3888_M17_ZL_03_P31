from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking_summary import (  #noqa: E402
    TrackingSummaryError,
    build_tracking_summary_from_files,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise lesion tracking outcomes per patient and optionally attach "
            "P31-18 ground-truth accuracy metrics."
        )
    )
    parser.add_argument(
        "--matches",
        required=True,
        help="Combined lesion_matches.csv produced by the matcher/pipeline.",
    )
    parser.add_argument(
        "--evaluation-summary",
        default="",
        help=(
            "Optional lesion_tracking_summary.csv from P31-18. When supplied, "
            "correct/incorrect counts and tracking accuracy are included."
        ),
    )
    parser.add_argument(
        "--out",
        default="outputs/tracking_patient_summary.csv",
        help="Output CSV containing per-patient rows and an ALL row.",
    )
    parser.add_argument(
        "--no-overall",
        action="store_true",
        help="Do not append the optional ALL/cohort row.",
    )
    return parser


def _fmt_accuracy(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{100.0 * number:.1f}%"


def main() -> None:
    args = _build_parser().parse_args()
    try:
        summary = build_tracking_summary_from_files(
            args.matches,
            evaluation_summary_path=args.evaluation_summary or None,
            include_overall=not args.no_overall,
        )
    except (FileNotFoundError, TrackingSummaryError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    print("Lesion tracking patient summary")
    print("-------------------------------")
    patient_rows = summary[summary["patient_id"] != "ALL"]
    print(f"Patients          : {len(patient_rows)}")
    for row in patient_rows.itertuples(index=False):
        print(
            f"  {row.patient_id}: BL={row.bl_lesions}, FU={row.fu_lesions}, "
            f"matched={row.matched_lesions}, appearing={row.appearing_lesions}, "
            f"disappearing={row.disappearing_lesions}, "
            f"accuracy={_fmt_accuracy(row.tracking_accuracy)}"
        )

    overall = summary[summary["patient_id"] == "ALL"]
    if not overall.empty:
        row = overall.iloc[0]
        print()
        print("Overall tested subset")
        print(f"  BL lesions       : {int(row['bl_lesions'])}")
        print(f"  FU lesions       : {int(row['fu_lesions'])}")
        print(f"  Matched lesions  : {int(row['matched_lesions'])}")
        print(f"  Appearing        : {int(row['appearing_lesions'])}")
        print(f"  Disappearing     : {int(row['disappearing_lesions'])}")
        print(f"  Tracking accuracy: {_fmt_accuracy(row['tracking_accuracy'])}")

    print()
    print(f"Summary CSV       : {out}")


if __name__ == "__main__":
    main()
