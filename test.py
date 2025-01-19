from pathlib import Path

from brainles_preprocessing.defacing import QuickshearDefacer
from brainles_preprocessing.brain_extraction import HDBetExtractor
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)


from pathlib import Path
from typing import List

# Function to run the pipeline for all subjects in a folder
def process_all_subjects_in_folder(data_folder: Path, output_folder: Path):
    # List all subject directories in the data folder
    subject_dirs = [d for d in data_folder.iterdir() if d.is_dir()]
    
    for subject_dir in subject_dirs:
        print(f"Processing subject: {subject_dir.name}")
        
        # Define file paths for the current subject
        t1c_file = subject_dir / "T1C.nii.gz"
        t1_file = subject_dir / "T1.nii.gz"
        fla_file = subject_dir / "FLAIR.nii.gz"
        t2_file = subject_dir / "T2.nii.gz"
        
        # Output paths
        t1c_normalized_skull_output_path = output_folder / f"{subject_dir.name}_t1c_normalized_skull.nii.gz"
        t1c_normalized_bet_output_path = output_folder / f"{subject_dir.name}_t1c_normalized_bet.nii.gz"
        t1c_normalized_defaced_output_path = output_folder / f"{subject_dir.name}_t1c_normalized_defaced.nii.gz"
        t1c_bet_mask = output_folder / f"{subject_dir.name}_t1c_bet_mask.nii.gz"
        t1c_defacing_mask = output_folder / f"{subject_dir.name}_t1c_defacing_mask.nii.gz"

        t1_normalized_bet_output_path = output_folder / f"{subject_dir.name}_t1_normalized_bet.nii.gz"
        fla_normalized_bet_output_path = output_folder / f"{subject_dir.name}_fla_normalized_bet.nii.gz"
        t2_normalized_bet_output_path = output_folder / f"{subject_dir.name}_t2_normalized_bet.nii.gz"

        # Create a PercentileNormalizer
        percentile_normalizer = PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )

        # Define the center modality
        center = CenterModality(
            modality_name="t1c",
            input_path=t1c_file,
            normalizer=percentile_normalizer,
            normalized_skull_output_path=t1c_normalized_skull_output_path,
            normalized_bet_output_path=t1c_normalized_bet_output_path,
            normalized_defaced_output_path=t1c_normalized_defaced_output_path,
            bet_mask_output_path=t1c_bet_mask,
            defacing_mask_output_path=t1c_defacing_mask,
        )

        # Define the moving modalities
        moving_modalities = [
            Modality(
                modality_name="t1",
                input_path=t1_file,
                normalizer=percentile_normalizer,
                normalized_bet_output_path=t1_normalized_bet_output_path,
            ),
            Modality(
                modality_name="t2",
                input_path=t2_file,
                normalizer=percentile_normalizer,
                normalized_bet_output_path=t2_normalized_bet_output_path,
            ),
            Modality(
                modality_name="flair",
                input_path=fla_file,
                normalizer=percentile_normalizer,
                normalized_bet_output_path=fla_normalized_bet_output_path,
            ),
        ]

        # Initialize the Preprocessor
        preprocessor = Preprocessor(
            center_modality=center,
            moving_modalities=moving_modalities,
            registrator=ANTsRegistrator(),
            brain_extractor=HDBetExtractor(),
            defacer=QuickshearDefacer(),
            limit_cuda_visible_devices="0",  # Adjust as per your GPU
        )

        # Run the preprocessor pipeline for the current subject
        preprocessor.run()
        print(f"Finished processing subject: {subject_dir.name}")

# Specify the data folder and output folder paths
data_folder = Path("/mnt/lustre-grete/usr/u12402/MRI_GBM")  # Folder with all subjects
output_folder = Path("output")  # Folder to save the results

# Ensure the output folder exists
if not output_folder.exists():
    output_folder.mkdir(parents=True)

# Process all subjects in the folder
process_all_subjects_in_folder(data_folder, output_folder)
