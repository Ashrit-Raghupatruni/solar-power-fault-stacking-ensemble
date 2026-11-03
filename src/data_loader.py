"""
Data verification and acquisition module for the Zenodo benchmark.
Validates the presence of raw dataset CSV files and provides structured download assistance.
"""
from pathlib import Path
from src.config import DATA_DIR, STATION_FILES

ZENODO_RECORD_URL = "https://zenodo.org/records/20565892"
DOI = "10.5281/zenodo.20565892"

def verify_dataset_files(raise_error: bool = False) -> bool:
    """
    Checks if station1_normal_features.csv, station2_medium_features.csv,
    and station3_high_features.csv exist in Database/controlled-fault-benchmark-v1.0/.
    """
    missing = []
    for st_id, p in STATION_FILES.items():
        if not p.exists():
            missing.append(p.name)
            
    if missing:
        msg = (
            f"\n[DATASET NOTICE] Missing {len(missing)} raw benchmark file(s) in {DATA_DIR}:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + f"\n\nTo download the official benchmark (1.81M records, 94.3 MB ZIP):\n"
            + f"  1. Visit: {ZENODO_RECORD_URL}\n"
            + f"  2. Extract CSV files into: {DATA_DIR}\n"
        )
        if raise_error:
            raise FileNotFoundError(msg)
        else:
            print(msg)
            return False
            
    print(f"All 3 station benchmark CSV files verified in {DATA_DIR}!")
    return True

if __name__ == "__main__":
    verify_dataset_files()
