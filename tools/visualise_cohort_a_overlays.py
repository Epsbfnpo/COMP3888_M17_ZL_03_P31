from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
#add the repository root to Python's module search path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import matplotlib.pyplot as plt
from src.cohort_a_loading import load_pair_manifest, load_patient_pair
from src.lesion_visualization import render_patient_comparison

def parse_args() -> argparse.Namespace:
    #define the commandline options
    parser = argparse.ArgumentParser(
        description="Create BL/FU Cohort A CT/PET lesion-overlay figures."
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="outputs/P31-9")
    parser.add_argument("--modality", choices=("ct", "pet"), default="ct")
    parser.add_argument("--patient-id", action="append")
    parser.add_argument("--max-patients", type=int, default=3)
    parser.add_argument("--baseline-slice", type=int)
    parser.add_argument("--followup-slice", type=int)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    #load manifest and select requested patients, if no patient id provided, use the first max_patients entries
    manifest = load_pair_manifest(args.pairs)
    patient_ids = (
            args.patient_id
            or manifest["patient_id"].astype(str).tolist()[:args.max_patients]
    )
    # create the output directory if not exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient_id in patient_ids:
        #load bl/fu images and lesion masks for one patient
        pair = load_patient_pair(
            args.pairs,
            patient_id,
            data_root=args.data_root,
            modalities=(args.modality, "lesion_mask"),
        )
        figure_path = (
                out_dir / f"{patient_id}_{args.modality}_overlay.png"
        )
        #create and save bl/fu comparison figure
        fig, result = render_patient_comparison(
            pair,
            modality=args.modality,
            baseline_slice=args.baseline_slice,
            followup_slice=args.followup_slice,
            output_path=figure_path,
        )
        #release matplotlib figure from memory
        plt.close(fig)
        #store res for manual inspection table
        rows.append(
            {
                "patient_id": patient_id,
                "modality": args.modality.upper(),
                "baseline_slice": result.baseline.slice_index,
                "followup_slice": result.followup.slice_index,
                "baseline_lesion_voxels":
                    result.baseline.total_lesion_voxels,
                "followup_lesion_voxels":
                    result.followup.total_lesion_voxels,
                "image_mask_alignment": "PASS",
                "manual_review": "PENDING",
                "notes": "",
                "figure": figure_path.name,
            }
        )
        print(f"Created {figure_path}")
    #write csv file for record manual review res
    review_path = out_dir / "manual_inspection.csv"
    with review_path.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Created {review_path}; visually review each figure "
        "and update manual_review/notes."
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())