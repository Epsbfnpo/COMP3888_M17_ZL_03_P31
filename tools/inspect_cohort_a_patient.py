from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cohort_a_loading import CohortALoadError, load_patient_pair, volume_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load BL/FU Cohort A PET/CT volumes from the pair manifest and print image metadata."
    )
    parser.add_argument("--pairs", required=True, help="Path to cohort_a_subset_pairs.csv")
    parser.add_argument("--patient-id", required=True, help="Patient identifier from the manifest")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Cohort A root. If omitted, COHORT_A_ROOT is used when set.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Inspect available modalities instead of stopping at the first missing file.",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    try:
        pair = load_patient_pair(
            args.pairs,
            args.patient_id,
            data_root=args.data_root,
            modalities=("ct", "pet"),
            allow_missing=args.allow_missing,
        )
    except CohortALoadError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    payload = {
        "patient_id": pair.patient_id,
        "baseline": {
            "scan_id": pair.baseline.scan_id,
            "ct": volume_summary(pair.baseline.ct),
            "pet": volume_summary(pair.baseline.pet),
        },
        "followup": {
            "scan_id": pair.followup.scan_id,
            "ct": volume_summary(pair.followup.ct),
            "pet": volume_summary(pair.followup.pet),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
