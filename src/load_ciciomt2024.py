"""Load and consolidate CICIoMT2024 WiFi/MQTT attack CSVs into one labeled dataframe.

Each source file = one attack class, already train/test split by CIC.
Labels are recovered from filenames since the CSVs carry no label column.
"""
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "CICIoMT2024" / "WiFI_and_MQTT" / "attacks" / "CSV"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Coarse category mapping (for the five-class taxonomy: Benign, MQTT, Recon, Spoofing, TCP_IP DoS/DDoS)
CATEGORY_RULES = [
    (r"^Benign", "Benign"),
    (r"^ARP_Spoofing", "Spoofing"),
    (r"^MQTT", "MQTT"),
    (r"^Recon", "Recon"),
    (r"^TCP_IP-DDoS", "TCP_IP-DDoS"),
    (r"^TCP_IP-DoS", "TCP_IP-DoS"),
]


def infer_label_and_category(filename: str) -> tuple[str, str]:
    """From e.g. 'TCP_IP-DDoS-ICMP3_train.pcap.csv' -> label 'TCP_IP-DDoS-ICMP', category 'TCP_IP-DDoS'."""
    stem = filename.replace(".pcap.csv", "")
    stem = re.sub(r"_(train|test)$", "", stem)
    # strip trailing numeric chunk id, e.g. ICMP3 -> ICMP, SYN1 -> SYN (but keep e.g. Connect_Flood intact)
    label = re.sub(r"(\d+)$", "", stem)
    category = None
    for pattern, cat in CATEGORY_RULES:
        if re.match(pattern, label):
            category = cat
            break
    if category is None:
        category = "Other"
    return label, category


def load_split(split: str) -> pd.DataFrame:
    split_dir = RAW_DIR / split
    frames = []
    for csv_path in sorted(split_dir.glob("*.csv")):
        label, category = infer_label_and_category(csv_path.name)
        df = pd.read_csv(csv_path)
        df["label"] = label
        df["category"] = category
        df["split"] = split
        df["source_file"] = csv_path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    train_df = load_split("train")
    test_df = load_split("test")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Combined: {full_df.shape}")
    print("\nRows per coarse category (combined):")
    print(full_df["category"].value_counts())
    print("\nRows per fine-grained label (combined):")
    print(full_df["label"].value_counts())

    out_path = OUT_DIR / "ciciomt2024_wifi_mqtt_consolidated.parquet"
    full_df.to_parquet(out_path, index=False)
    print(f"\nSaved consolidated dataframe to {out_path}")


if __name__ == "__main__":
    main()
