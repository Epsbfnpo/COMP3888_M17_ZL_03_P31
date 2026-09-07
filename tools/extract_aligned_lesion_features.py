from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cohort_a_loading import load_nifti_volume
from src.lesion_features import extract_aligned_lesion_features, export_lesion_features

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patient-id', required=True)
    parser.add_argument('--bl-mask', required=True, help='Registered BL lesion mask in FU space')
    parser.add_argument('--fu-mask', required=True, help='FU lesion mask on FU CT grid')
    parser.add_argument('--fu-reference', required=True, help='Fixed FU CT used for registration')
    parser.add_argument('--bl-pet', help='Optional registered BL PET on FU CT grid')
    parser.add_argument('--fu-pet', help='Optional FU PET resampled onto FU CT grid')
    parser.add_argument('--bl-pet-units', default='unknown', help='SUV only if already SUV calibrated')
    parser.add_argument('--fu-pet-units', default='unknown', help='SUV only if already SUV calibrated')
    parser.add_argument('--connectivity', type=int, choices=[6, 18, 26], default=18)
    parser.add_argument('--out', default='outputs/aligned_lesion_features.csv')
    args = parser.parse_args()
    rows = extract_aligned_lesion_features(
        load_nifti_volume(args.bl_mask, preserve_dtype=True),
        load_nifti_volume(args.fu_mask, preserve_dtype=True),
        reference=load_nifti_volume(args.fu_reference), patient_id=args.patient_id,
        baseline_pet=load_nifti_volume(args.bl_pet) if args.bl_pet else None,
        followup_pet=load_nifti_volume(args.fu_pet) if args.fu_pet else None,
        baseline_pet_units=args.bl_pet_units, followup_pet_units=args.fu_pet_units,
        connectivity=args.connectivity,
    )
    export_lesion_features(rows, args.out)
    print(f'Wrote {len(rows)} lesion features: {args.out}')

if __name__ == '__main__':
    main()
