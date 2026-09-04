"""Extract windowed flow-style features from CICIoMT2024 Bluetooth (BLE/HCI) PCAPs.

BLE captures are HCI-layer events, not IP flows, so there's no 5-tuple to group by.
Instead we aggregate packets into fixed-size tumbling windows (like a sliding flow
window) and compute statistics analogous to the WiFi/MQTT CICFlowMeter schema
(packet counts, byte size stats, inter-arrival time stats) plus BLE-specific
HCI event-type distribution features.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import PcapReader

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "CICIoMT2024" / "Bluetooth" / "attacks" / "pcap"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 200  # packets per window


def packet_event_type(pkt) -> str:
    """Best-effort HCI event/type label from the layer names present on the packet."""
    layers = [l.__name__ for l in pkt.layers()]
    for name in ("HCI_LE_Meta_Advertising_Reports", "HCI_Event_Command_Complete",
                 "HCI_Event_Disconnection_Complete", "HCI_Event_Number_Of_Completed_Packets",
                 "HCI_ACL_Hdr", "HCI_Event_LE_Meta"):
        if name in layers:
            return name
    return layers[-1] if layers else "Unknown"


def extract_windows(pcap_path: Path) -> pd.DataFrame:
    lengths, times, types = [], [], []
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            lengths.append(len(pkt))
            times.append(float(pkt.time))
            types.append(packet_event_type(pkt))

    n = len(lengths)
    rows = []
    for start in range(0, n, WINDOW_SIZE):
        end = min(start + WINDOW_SIZE, n)
        if end - start < WINDOW_SIZE // 4:  # drop tiny trailing window
            continue
        win_len = np.array(lengths[start:end], dtype=float)
        win_time = np.array(times[start:end], dtype=float)
        win_types = types[start:end]
        iat = np.diff(win_time) if len(win_time) > 1 else np.array([0.0])

        type_counts = pd.Series(win_types).value_counts(normalize=True)

        row = {
            "packet_count": end - start,
            "duration": win_time[-1] - win_time[0] if len(win_time) > 1 else 0.0,
            "byte_sum": win_len.sum(),
            "byte_min": win_len.min(),
            "byte_max": win_len.max(),
            "byte_avg": win_len.mean(),
            "byte_std": win_len.std(),
            "iat_avg": iat.mean(),
            "iat_std": iat.std(),
            "iat_max": iat.max(),
            "iat_min": iat.min(),
        }
        for hci_type in ("HCI_LE_Meta_Advertising_Reports", "HCI_Event_Command_Complete",
                          "HCI_Event_Disconnection_Complete", "HCI_Event_Number_Of_Completed_Packets",
                          "HCI_ACL_Hdr", "HCI_Event_LE_Meta"):
            row[f"pct_{hci_type}"] = type_counts.get(hci_type, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    frames = []
    for split in ("train", "test"):
        split_dir = RAW_DIR / split
        for pcap_path in sorted(split_dir.glob("*.pcap")):
            label = "Benign" if "Benign" in pcap_path.stem else "DoS"
            print(f"Processing {pcap_path.name} ...")
            df = extract_windows(pcap_path)
            df["label"] = label
            df["category"] = f"Bluetooth_{label}"
            df["split"] = split
            df["source_file"] = pcap_path.name
            df["protocol"] = "Bluetooth"
            frames.append(df)
            print(f"  -> {len(df)} windows extracted")

    full_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal windows: {len(full_df)}")
    print(full_df.groupby(["split", "label"]).size())

    out_path = OUT_DIR / "ciciomt2024_bluetooth_consolidated.parquet"
    full_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
