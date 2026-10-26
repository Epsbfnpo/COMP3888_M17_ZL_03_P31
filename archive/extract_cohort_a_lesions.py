from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cohort_a_loading import CohortALoadError, load_patient_pair
from src.lesion_components import LesionMaskError, extract_individual_lesions, lesion_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load BL/FU lesion masks, split connected lesions, and write temporary lesion IDs."
    )
    parser.add_argument("--pairs", required=True, help="Path to cohort_a_subset_pairs.csv")
    parser.add_argument("--patient-id", required=True, help="Patient identifier from the manifest")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Cohort A root. If omitted, COHORT_A_ROOT is used when set.",
    )
    parser.add_argument("--out", default="outputs/cohort_a_lesions.csv", help="Output lesion CSV")
    parser.add_argument("--connectivity", choices=[6, 18, 26], type=int, default=18)
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    try:
        pair = load_patient_pair(
            args.pairs,
            args.patient_id,
            data_root=args.data_root,
            modalities=("lesion_mask",),
        )
        if pair.baseline.lesion_mask is None or pair.followup.lesion_mask is None:
            raise RuntimeError("BL/FU lesion masks were unexpectedly not loaded.")
        bl = extract_individual_lesions(
            pair.baseline.lesion_mask,
            patient_id=pair.patient_id,
            timepoint="BL",
            connectivity=args.connectivity,
        )
        fu = extract_individual_lesions(
            pair.followup.lesion_mask,
            patient_id=pair.patient_id,
            timepoint="FU",
            connectivity=args.connectivity,
        )
    except (CohortALoadError, LesionMaskError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    rows = lesion_rows([bl, fu])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"Patient: {pair.patient_id}")
    print(f"BL lesions: {bl.lesion_count}")
    print(f"FU lesions: {fu.lesion_count}")
    print(f"Wrote {len(rows)} lesion row(s): {out}")


if __name__ == "__main__":
    main()
