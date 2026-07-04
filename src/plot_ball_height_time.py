import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = PROJECT_ROOT / "outputs" / "ball_positions.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "ball_height_time_graph.png"

FPS = 30
NET_HEIGHT = 2.43


def extract_frame_number(frame_name):
    match = re.findall(r"\d+", str(frame_name))

    if not match:
        return None

    return int(match[-1])


def main():
    print(f"Reading CSV file: {CSV_FILE}")

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Cannot find file: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    print("CSV columns:")
    print(df.columns.tolist())

    print(f"Total rows in CSV: {len(df)}")

    if "status" in df.columns:
        print("Status counts:")
        print(df["status"].value_counts())

        df = df[df["status"] == "ok"].copy()

    print(f"Rows after keeping status ok: {len(df)}")

    if "Z" not in df.columns:
        raise RuntimeError("Z column not found in ball_positions.csv")

    df["Z"] = pd.to_numeric(df["Z"], errors="coerce")
    df = df.dropna(subset=["Z"])

    print(f"Valid Z rows: {len(df)}")

    if len(df) == 0:
        print("No valid Z values found.")
        print("Open outputs/ball_positions.csv and check whether X, Y, Z values are empty.")
        return

    df["frame_number"] = df["frame"].apply(extract_frame_number)

    if df["frame_number"].isna().all():
        print("Could not extract frame numbers from frame names.")
        print("Using row index as time instead.")
        df["time_seconds"] = range(len(df))
        df["time_seconds"] = df["time_seconds"] / FPS
    else:
        df = df.dropna(subset=["frame_number"])
        df = df.sort_values("frame_number")
        df["time_seconds"] = df["frame_number"] / FPS

    print("\nFirst few height values:")
    print(df[["frame", "time_seconds", "Z"]].head(10).to_string(index=False))

    # plt.figure(figsize=(12, 6))

    # plt.plot(
    #     df["time_seconds"],
    #     df["Z"],
    #     color="black",
    #     linewidth=2,
    #     marker="o",
    #     markersize=4,
    #     label="Height"
    # )

    # plt.axhline(
    #     NET_HEIGHT,
    #     color="red",
    #     linestyle="--",
    #     linewidth=2,
    #     label="Men's net height 2.43 m"
    # )

    # plt.xlabel("Time (s)")
    # plt.ylabel("Ball height (m)")
    # plt.title("Ball Height over Time")
    # plt.grid(True, alpha=0.3)
    # plt.legend()

    # y_max = max(5, df["Z"].max() + 0.5)
    # plt.ylim(0, y_max)

    # plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

    # print(f"\nSaved graph successfully: {OUTPUT_FILE}")

    # plt.show()

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["time_seconds"],
        df["Z"],
        color="black",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Ball height Z"
    )

    plt.axhline(
        0,
        color="gray",
        linestyle="-",
        linewidth=1.5,
        label="Floor level Z=0"
    )

    plt.axhline(
        NET_HEIGHT,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Men's net height 2.43 m"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Ball height Z (m)")
    plt.title("Ball Height over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    y_min = df["Z"].min() - 0.5
    y_max = max(NET_HEIGHT + 0.5, df["Z"].max() + 0.5)

    plt.ylim(y_min, y_max)

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()