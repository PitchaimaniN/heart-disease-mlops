"""Download and prepare the Cleveland Heart Disease dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target_raw",
]

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"


def download_raw_data() -> None:
    """Download a fresh copy of the original, headerless UCI CSV."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URL, RAW_DATA_PATH)
    print(f"Downloaded original data to: {RAW_DATA_PATH}")


def load_raw_data() -> pd.DataFrame:
    """Read the source CSV and convert its missing markers and columns."""
    raw_data = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=RAW_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )

    for column in RAW_COLUMNS:
        raw_data[column] = pd.to_numeric(raw_data[column], errors="coerce")

    return raw_data


def report_missing_values(raw_data: pd.DataFrame) -> None:
    """Report missing counts, source lines, and affected patient context."""
    print("\nRAW DATA - missing-value audit")
    print("In the original CSV, '?' means that a value was not recorded.")
    print(
        f"Raw rows: {len(raw_data):,} | "
        f"Raw columns: {len(raw_data.columns)}"
    )

    missing_summary = pd.DataFrame(
        {
            "Missing values": raw_data.isna().sum(),
            "Missing (%)": (raw_data.isna().mean() * 100).round(2),
        }
    )
    print("\nMissing values by column:")
    print(missing_summary.to_string())

    missing_cells = []
    patient_row_indices = []
    for row_index, (_, row) in enumerate(raw_data.iterrows()):
        for feature in RAW_COLUMNS:
            if pd.isna(row[feature]):
                missing_cells.append(
                    {
                        "Raw data line": row_index + 1,
                        "Feature": feature,
                    }
                )
                patient_row_indices.append(row_index)

    missing_locations = pd.DataFrame(missing_cells)

    if missing_locations.empty:
        print("\nNo missing cells found in the downloaded raw data.")
        return

    print(
        "\nExact missing-cell locations "
        "(line numbers are 1-based patient records; the source has no header):"
    )
    print(missing_locations.to_string(index=False))

    patient_row_indices = sorted(set(patient_row_indices))
    affected_records = raw_data.loc[
        patient_row_indices, ["age", "ca", "thal", "target_raw"]
    ].copy()
    affected_records.insert(
        0,
        "Raw data line",
        [row_index + 1 for row_index in patient_row_indices],
    )

    print(
        "\nAffected patient context "
        "(age, missing-value features, and original target):"
    )
    print(affected_records.to_string(index=False))


def clean_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop unknown targets and map the original 0-4 outcome to binary."""
    rows_before_cleaning = len(raw_data)
    cleaned_data = raw_data.dropna(subset=["target_raw"]).copy()
    rows_removed = rows_before_cleaning - len(cleaned_data)

    cleaned_data["target"] = (cleaned_data["target_raw"] > 0).astype(int)
    cleaned_data = cleaned_data.drop(columns=["target_raw"])

    return cleaned_data, rows_removed


def save_clean_data(cleaned_data: pd.DataFrame) -> None:
    """Save the prepared dataset and keep missing predictors for imputation."""
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(CLEAN_DATA_PATH, index=False)

    print("\nCLEANED DATA")
    print(f"Saved raw data to: {RAW_DATA_PATH}")
    print(f"Saved cleaned data to: {CLEAN_DATA_PATH}")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")


def main() -> None:
    """Run the download, audit, cleaning, and save workflow."""
    print("Heart Disease dataset preparation")
    print(
        "This process downloads the source and overwrites "
        "the raw and cleaned CSVs."
    )

    print("\nStep 1/4: Downloading the source data.")
    download_raw_data()

    print("\nStep 2/4: Reading the raw data and auditing missing values.")
    raw_data = load_raw_data()
    report_missing_values(raw_data)

    print("\nStep 3/4: Cleaning the target and preparing the binary label.")
    cleaned_data, rows_removed = clean_data(raw_data)
    print(
        f"Rows retained: {len(cleaned_data):,} | "
        f"Rows removed for missing target: {rows_removed:,}"
    )
    print(
        "Target mapping: 0 = no heart disease; "
        "original values 1-4 = disease (1)."
    )
    print(
        "Missing predictor values are retained for imputation "
        "during model training."
    )

    print("\nStep 4/4: Saving the cleaned dataset.")
    save_clean_data(cleaned_data)

    cleaned_missing_summary = pd.DataFrame(
        {
            "Missing values": cleaned_data.isna().sum(),
            "Missing (%)": (cleaned_data.isna().mean() * 100).round(2),
        }
    )
    print("\nMissing values in the cleaned dataset:")
    print(cleaned_missing_summary.to_string())
    print("\nFirst five cleaned rows:")
    print(cleaned_data.head().to_string(index=False))


if __name__ == "__main__":
    main()
