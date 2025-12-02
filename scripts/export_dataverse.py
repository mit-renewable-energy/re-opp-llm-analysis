#!/usr/bin/env python3
"""
Export datasets for Harvard Dataverse publication.

Generates:
- raw_data.zip: All input data files with documentation
- analysis_results.csv: Final opposition/support scores
- validation_results.json: Human validation accuracy labels (summary only)
"""

import json
import pickle
import shutil
import zipfile
from pathlib import Path


def export_validation_results(input_path: Path, output_path: Path) -> None:
    """Export validation results with only plant_code and accuracy_summary."""
    with open(input_path, "r") as f:
        data = json.load(f)

    simplified = [
        {
            "plant_code": entry["plant_code"],
            "accuracy_summary": entry["accuracy_summary"]
        }
        for entry in data
    ]

    with open(output_path, "w") as f:
        json.dump(simplified, f, indent=2)

    print(f"Exported {len(simplified)} validation records to {output_path}")


def export_analysis_results(input_path: Path, output_path: Path) -> None:
    """Export analysis dataset as CSV."""
    with open(input_path, "rb") as f:
        df = pickle.load(f)

    # Drop geometry column if present (not CSV-compatible)
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} records ({len(df.columns)} columns) to {output_path}")


def export_raw_data(raw_dir: Path, output_path: Path) -> None:
    """Create zip of raw data directory."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in raw_dir.rglob("*"):
            if file_path.is_file():
                # Skip extracted folders if zip exists, and skip large pickle files
                if file_path.suffix == ".pkl":
                    continue
                arcname = file_path.relative_to(raw_dir)
                zf.write(file_path, arcname)
                print(f"  Added: {arcname}")

    print(f"Created {output_path}")


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "dataverse_export"
    output_dir.mkdir(exist_ok=True)

    print("Exporting datasets for Harvard Dataverse...\n")

    # Export validation results (simplified)
    print("1. Validation results:")
    export_validation_results(
        base_dir / "data/final/validation_human_labels.json",
        output_dir / "validation_results.json"
    )

    # Export analysis results as CSV
    print("\n2. Analysis results:")
    try:
        export_analysis_results(
            base_dir / "data/final/complete_analysis_dataset.pkl",
            output_dir / "analysis_results.csv"
        )
    except Exception as e:
        print(f"  Error exporting analysis results: {e}")
        print("  Make sure geopandas is installed: pip install geopandas")

    # Export raw data as zip
    print("\n3. Raw data:")
    export_raw_data(
        base_dir / "data/raw",
        output_dir / "raw_data.zip"
    )

    print(f"\nExport complete. Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
