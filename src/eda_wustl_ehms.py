"""Quick EDA on WUSTL-EHMS-2020: class balance, feature groups, missing values."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "wustl-ehms-2020.csv"

BIOMETRIC_COLS = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]


def main():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    print("\nLabel distribution:")
    print(df["Label"].value_counts())

    print("\nAttack Category distribution:")
    print(df["Attack Category"].value_counts())

    print("\nMissing values per column (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nBiometric feature summary (normal vs attack):")
    for col in BIOMETRIC_COLS:
        if col in df.columns:
            grouped = df.groupby("Label")[col].mean()
            print(f"  {col}: normal={grouped.get(0, float('nan')):.3f}  attack={grouped.get(1, float('nan')):.3f}")

    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "wustl_ehms_2020.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved cleaned copy to {out_path}")


if __name__ == "__main__":
    main()
