from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lesion_pair_costs import (
    PairCostConfig,
    export_cost_matrices,
    export_pair_costs,
    generate_lesion_pair_costs,
    load_aligned_feature_csvs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate within-patient BL/FU lesion candidate costs from aligned "
            "lesion-feature CSV files. This tool scores candidate edges only; "
            "it does not perform lesion assignment."
        )
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="One or more CSV files produced by extract_aligned_lesion_features.py.",
    )
    parser.add_argument(
        "--out-pairs",
        default="outputs/lesion_pair_costs.csv",
        help="Long-form candidate-edge CSV output.",
    )
    parser.add_argument(
        "--matrix-dir",
        default="outputs/lesion_pair_cost_matrices",
        help="Directory for one BL x FU labelled cost matrix CSV per patient.",
    )
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.5)
    parser.add_argument("--pet-weight", type=float, default=0.25)
    parser.add_argument(
        "--distance-scale-mm",
        type=float,
        default=50.0,
        help="Distance divisor used to put mm distances on a cost scale.",
    )
    parser.add_argument(
        "--pet-feature",
        choices=["none", "suvmean", "suvmax", "pet_mean", "pet_max"],
        default="suvmax",
        help=(
            "Optional PET feature. SUV columns are safest when available. Raw "
            "PET mean/max are used only when BL/FU pet_units match."
        ),
    )
    parser.add_argument(
        "--max-distance-mm",
        type=float,
        help=(
            "Optional spatial gate. BL/FU pairs farther apart than this are "
            "not candidate edges and appear as inf in the cost matrix."
        ),
    )
    args = parser.parse_args()

    features = load_aligned_feature_csvs(args.features)
    config = PairCostConfig(
        distance_weight=args.distance_weight,
        size_weight=args.size_weight,
        pet_weight=args.pet_weight,
        distance_scale_mm=args.distance_scale_mm,
        pet_feature=args.pet_feature,
        max_distance_mm=args.max_distance_mm,
    )
    result = generate_lesion_pair_costs(features, config)

    pair_path = export_pair_costs(result.pair_costs, args.out_pairs)
    matrix_paths = export_cost_matrices(result.matrices, args.matrix_dir)

    print(f"Wrote {len(result.pair_costs)} candidate lesion pair(s): {pair_path}")
    print(f"Wrote {len(matrix_paths)} patient cost matrix file(s): {args.matrix_dir}")

    if result.matrices:
        for patient_id, matrix in sorted(result.matrices.items()):
            finite = int((matrix.values < float("inf")).sum())
            print(
                f"  {patient_id}: BL={len(matrix.bl_lesion_ids)}, "
                f"FU={len(matrix.fu_lesion_ids)}, candidates={finite}, "
                f"matrix={matrix.shape}"
            )


if __name__ == "__main__":
    main()
