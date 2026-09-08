"""Small self-contained demonstration of lesion-pair cost generation."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lesion_pair_costs import (
    PairCostConfig,
    export_cost_matrices,
    export_pair_costs,
    generate_lesion_pair_costs,
)


def main() -> None:
    rows = [
        #Patient A: two BL lesions and two FU lesions.
        dict(patient_id="demo_A", timepoint="BL", lesion_id="demo_A_BL_L001", volume_ml=2.0,
             centroid_x_mm=0, centroid_y_mm=0, centroid_z_mm=0, coordinate_space="FU_RAS_mm", suvmax=8.0),
        dict(patient_id="demo_A", timepoint="BL", lesion_id="demo_A_BL_L002", volume_ml=5.0,
             centroid_x_mm=50, centroid_y_mm=0, centroid_z_mm=0, coordinate_space="FU_RAS_mm", suvmax=12.0),
        dict(patient_id="demo_A", timepoint="FU", lesion_id="demo_A_FU_L001", volume_ml=2.2,
             centroid_x_mm=3, centroid_y_mm=4, centroid_z_mm=0, coordinate_space="FU_RAS_mm", suvmax=9.0),
        dict(patient_id="demo_A", timepoint="FU", lesion_id="demo_A_FU_L002", volume_ml=4.5,
             centroid_x_mm=52, centroid_y_mm=0, centroid_z_mm=0, coordinate_space="FU_RAS_mm", suvmax=11.0),
        # Patient B: two BL candidates are both allowed to point to one FU lesion.
        dict(patient_id="demo_B", timepoint="BL", lesion_id="demo_B_BL_L001", volume_ml=1.0,
             centroid_x_mm=0, centroid_y_mm=0, centroid_z_mm=0, coordinate_space="FU_RAS_mm", suvmax=None),
        dict(patient_id="demo_B", timepoint="BL", lesion_id="demo_B_BL_L002", volume_ml=1.2,
             centroid_x_mm=4, centroid_y_mm=0, centroid_z_mm=0, coordinate_space="FU_RAS_mm", suvmax=None),
        dict(patient_id="demo_B", timepoint="FU", lesion_id="demo_B_FU_L001", volume_ml=2.1,
             centroid_x_mm=2, centroid_y_mm=0, centroid_z_mm=0, coordinate_space="FU_RAS_mm", suvmax=None),
    ]
    features = pd.DataFrame(rows)
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(
            distance_weight=1.0,
            size_weight=0.5,
            pet_weight=0.25,
            pet_feature="suvmax",
            distance_scale_mm=50.0,
        ),
    )

    out_root = Path("outputs/demo_lesion_pair_costs")
    pair_path = export_pair_costs(result.pair_costs, out_root / "lesion_pair_costs.csv")
    matrix_paths = export_cost_matrices(result.matrices, out_root / "matrices")

    print(result.pair_costs.to_string(index=False))
    print(f"\nWrote pair costs: {pair_path}")
    for path in matrix_paths:
        print(f"Wrote matrix: {path}")


if __name__ == "__main__":
    main()
