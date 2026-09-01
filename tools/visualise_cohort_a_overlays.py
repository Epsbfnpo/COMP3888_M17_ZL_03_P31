from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import matplotlib.pyplot as plt

#add repository root to Python's module search path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cohort_a_loading import load_pair_manifest, load_patient_pair
from src.lesion_visualization import (
    VisualizationError,
    render_patient_comparison,
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create BL/FU Cohort A CT/PET lesion-overlay figures."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to cohort_a_subset_pairs.csv.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing the Cohort A imaging data.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/P31-9",
        help="Directory used to save generated figures.",
    )
    parser.add_argument(
        "--modality",
        choices=("ct", "pet"),
        default="ct",
        help="Imaging modality to visualise.",
    )
    parser.add_argument(
        "--patient-id",
        action="append",
        help=(
            "Patient ID to process. This option can be supplied multiple times. "
            "If omitted, the first --max-patients patients are processed."
        ),
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=3,
        help="Maximum number of patients to process when --patient-id is omitted.",
    )
    parser.add_argument(
        "--baseline-slice",
        type=int,
        help="Optional baseline axial slice index.",
    )
    parser.add_argument(
        "--followup-slice",
        type=int,
        help="Optional follow-up axial slice index.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate baseline and follow-up lesion-overlay figures."""
    args = parse_args()
    #Load patient-pair manifest.
    manifest = load_pair_manifest(args.pairs)
    #Process explicitly requested patients when patient-id is supplied
    #Otherwise process the first max-patients entries in the manifest
    patient_ids = (
            args.patient_id
            or manifest["patient_id"].astype(str).tolist()[: args.max_patients]
    )
    #create output directory if it not exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "modality",
        "baseline_slice",
        "followup_slice",
        "baseline_lesion_voxels",
        "followup_lesion_voxels",
        "image_mask_alignment",
        "manual_review",
        "notes",
        "figure",
    ]
    successful_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, str]] = []
    for patient_id in patient_ids:
        print(f"Processing patient {patient_id}...")
        try:
            #load selected image modality and lesion masks for both
            #bl and fu timepoints
            pair = load_patient_pair(
                args.pairs,
                patient_id,
                data_root=args.data_root,
                modalities=(args.modality, "lesion_mask"),
            )
            figure_path = (
                    out_dir / f"{patient_id}_{args.modality}_overlay.png"
            )
            #render and save bl/fu comparison figure
            fig, result = render_patient_comparison(
                pair,
                modality=args.modality,
                baseline_slice=args.baseline_slice,
                followup_slice=args.followup_slice,
                output_path=figure_path,
            )
        except VisualizationError as error:
            #An empty lesion mask or another visualisation problem should not
            #stop the processing of the remaining patients
            message = str(error)
            print(f"Skipped patient {patient_id}: {message}")
            skipped_rows.append(
                {
                    "patient_id": patient_id,
                    "reason": message,
                }
            )
            continue
        except Exception as error:
            #record unexpected patient-specific errors and continue processing
            #remaining patients
            message = f"{type(error).__name__}: {error}"
            print(f"Failed patient {patient_id}: {message}")
            skipped_rows.append(
                {
                    "patient_id": patient_id,
                    "reason": message,
                }
            )
            continue
        else:
            #release the matplotlib figure after it saved
            plt.close(fig)
            successful_rows.append(
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
    #write the inspection records for successfully processed patients
    review_path = out_dir / "manual_inspection.csv"
    with review_path.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(successful_rows)
    print(f"Created {review_path}")
    #write the skipped patient id and their failure reasons
    skipped_path = out_dir / "skipped_patients.csv"
    with skipped_path.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient_id", "reason"],
        )
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"Created {skipped_path}")
    #print a final processing summary
    print()
    print("Processing complete.")
    print(f"Successful patients: {len(successful_rows)}")
    print(f"Skipped patients: {len(skipped_rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
