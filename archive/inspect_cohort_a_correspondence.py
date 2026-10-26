from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Add repository root to Python module search path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.cohort_a_loading import (
    CohortALoadError,
    load_pair_manifest,
    get_patient_row,
    resolve_manifest_path,
)


IMPORTANT_COLUMNS = [
    "lesion_id",
    "cog_bl",
    "cog_backpropagated",
    "img_id_bl",
    "cog_propagated",
    "cog_fu",
    "img_id_fu",
    "lesion_type",
    "topology_class",
    "merged_into",
    "volume_bl",
    "volume_fu",
    "target_lesion",
    "use_for_challenge",
    "linking_unclear",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect expert-confirmed lesion correspondence data "
            "for Cohort A."
        )
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to cohort_a_subset_pairs.csv.",
    )

    parser.add_argument(
        "--patient-id",
        default=None,
        help=(
            "Inspect one patient in detail. "
            "If omitted, all patients in the pair manifest are summarised."
        ),
    )

    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Cohort A root directory. "
            "If omitted, COHORT_A_ROOT is used when available."
        ),
    )

    return parser.parse_args()


def load_reference_csv(
    row: pd.Series,
    *,
    manifest_path: str | Path,
    data_root: str | Path | None,
) -> tuple[Path, pd.DataFrame]:

    if "reference_csv_path" not in row.index:
        raise ValueError(
            "Pair manifest does not contain 'reference_csv_path'."
        )

    raw_path = str(row["reference_csv_path"]).strip()

    if not raw_path:
        raise ValueError(
            f"Patient {row['patient_id']} has no reference CSV path."
        )

    path = resolve_manifest_path(
        raw_path,
        manifest_path=manifest_path,
        data_root=data_root,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Reference CSV does not exist: {path}"
        )

    df = pd.read_csv(path)

    return path, df


def validate_reference_format(df: pd.DataFrame) -> list[str]:
    """Return expected columns that are missing."""
    return [
        column
        for column in IMPORTANT_COLUMNS
        if column not in df.columns
    ]


def print_patient_details(
    patient_id: str,
    path: Path,
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print(f"Patient: {patient_id}")
    print(f"Reference CSV: {path}")
    print(f"Correspondence rows: {len(df)}")
    print("=" * 70)

    print()
    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    missing = validate_reference_format(df)

    if missing:
        print()
        print("WARNING: expected column(s) missing:")
        for column in missing:
            print(f"  - {column}")

    if "topology_class" in df.columns:
        print()
        print("Topology classes:")

        counts = (
            df["topology_class"]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
        )

        for topology, count in counts.items():
            print(f"  {topology}: {count}")

    if "lesion_type" in df.columns:
        print()
        print("Lesion types:")

        counts = (
            df["lesion_type"]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
        )

        for lesion_type, count in counts.items():
            print(f"  {lesion_type}: {count}")

    if "linking_unclear" in df.columns:
        unclear = (
            df["linking_unclear"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .sum()
        )

        print()
        print(f"Unclear correspondence links: {int(unclear)}")

    # Show merge relationships
    if (
        "merged_into" in df.columns
        and "lesion_id" in df.columns
    ):
        merged = df[df["merged_into"].notna()].copy()

        if not merged.empty:
            print()
            print("Merge relationships:")

            for _, row in merged.iterrows():
                print(
                    f"  lesion {row['lesion_id']}"
                    f" -> lesion {row['merged_into']}"
                )

    # Print compact manual-review table
    review_columns = [
        column
        for column in [
            "lesion_id",
            "lesion_type",
            "topology_class",
            "cog_bl",
            "cog_propagated",
            "cog_fu",
            "merged_into",
            "volume_bl",
            "volume_fu",
            "linking_unclear",
        ]
        if column in df.columns
    ]

    if review_columns:
        print()
        print("Manual review:")
        print(
            df[review_columns].to_string(
                index=False
            )
        )


def inspect_single_patient(
    manifest: pd.DataFrame,
    *,
    patient_id: str,
    manifest_path: str | Path,
    data_root: str | Path | None,
) -> None:

    row = get_patient_row(
        manifest,
        patient_id,
    )

    path, df = load_reference_csv(
        row,
        manifest_path=manifest_path,
        data_root=data_root,
    )

    print_patient_details(
        patient_id,
        path,
        df,
    )


def inspect_all_patients(
    manifest: pd.DataFrame,
    *,
    manifest_path: str | Path,
    data_root: str | Path | None,
) -> None:

    topology_counts: dict[str, int] = {}
    topology_examples: dict[str, tuple[str, object]] = {}

    successful = 0
    skipped = 0

    print(
        f"Inspecting {len(manifest)} patient(s)..."
    )

    for _, manifest_row in manifest.iterrows():

        patient_id = str(
            manifest_row["patient_id"]
        )

        try:
            _, df = load_reference_csv(
                manifest_row,
                manifest_path=manifest_path,
                data_root=data_root,
            )

        except Exception as exc:
            print(
                f"Skipped {patient_id}: {exc}"
            )
            skipped += 1
            continue

        successful += 1

        if "topology_class" not in df.columns:
            continue

        for _, lesion_row in df.iterrows():

            topology = str(
                lesion_row["topology_class"]
            ).strip()

            if not topology:
                topology = "<missing>"

            topology_counts[topology] = (
                topology_counts.get(topology, 0)
                + 1
            )

            if topology not in topology_examples:
                lesion_id = lesion_row.get(
                    "lesion_id",
                    "?",
                )

                topology_examples[topology] = (
                    patient_id,
                    lesion_id,
                )

    print()
    print("=" * 70)
    print("Cohort A expert correspondence summary")
    print("=" * 70)

    print(f"Patients inspected: {successful}")
    print(f"Patients skipped: {skipped}")

    print()
    print("Observed topology classes:")

    for topology, count in sorted(
        topology_counts.items()
    ):
        print(
            f"  {topology}: {count}"
        )

    print()
    print("Example cases:")

    for topology, example in sorted(
        topology_examples.items()
    ):
        patient_id, lesion_id = example

        print(
            f"  {topology}: "
            f"patient {patient_id}, "
            f"lesion {lesion_id}"
        )


def main() -> int:

    args = parse_args()

    try:
        manifest = load_pair_manifest(
            args.pairs
        )

        if args.patient_id:
            inspect_single_patient(
                manifest,
                patient_id=args.patient_id,
                manifest_path=args.pairs,
                data_root=args.data_root,
            )

        else:
            inspect_all_patients(
                manifest,
                manifest_path=args.pairs,
                data_root=args.data_root,
            )

    except (
        CohortALoadError,
        ValueError,
        FileNotFoundError,
    ) as exc:

        raise SystemExit(
            f"ERROR: {exc}"
        ) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())