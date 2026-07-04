import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = PROJECT_ROOT / "outputs" / "ball_positions.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "ball_xyz_court_net_clear.png"


COURT_LENGTH = 18
COURT_WIDTH = 9
NET_HEIGHT = 2.43


def draw_3d_court_with_net(ax):
    # Court boundary
    ax.plot([0, 18], [0, 0], [0, 0], color="black", linewidth=2)
    ax.plot([0, 18], [9, 9], [0, 0], color="black", linewidth=2)
    ax.plot([0, 0], [0, 9], [0, 0], color="black", linewidth=2)
    ax.plot([18, 18], [0, 9], [0, 0], color="black", linewidth=2)

    # Center line / net line
    ax.plot([9, 9], [0, 9], [0, 0], color="black", linestyle="--", linewidth=2)

    # Attack lines
    ax.plot([3, 3], [0, 9], [0, 0], color="gray", linestyle=":", linewidth=2)
    ax.plot([15, 15], [0, 9], [0, 0], color="gray", linestyle=":", linewidth=2)

    # Net
    ax.plot([9, 9], [0, 9], [NET_HEIGHT, NET_HEIGHT], color="blue", linewidth=4)
    ax.plot([9, 9], [0, 0], [0, NET_HEIGHT], color="blue", linewidth=2)
    ax.plot([9, 9], [9, 9], [0, NET_HEIGHT], color="blue", linewidth=2)

    # Net mesh guide lines
    for y in [1.5, 3, 4.5, 6, 7.5]:
        ax.plot([9, 9], [y, y], [0, NET_HEIGHT], color="blue", linestyle="--", linewidth=0.7)

    for z in [0.5, 1.0, 1.5, 2.0]:
        ax.plot([9, 9], [0, 9], [z, z], color="blue", linestyle="--", linewidth=0.7)

    # Labels
    ax.text(9, 9.3, NET_HEIGHT, "Net 2.43 m", color="blue", fontsize=10)
    ax.text(3, 9.3, 0, "Attack line", fontsize=8)
    ax.text(15, 9.3, 0, "Attack line", fontsize=8)

    ax.set_xlim(0, COURT_LENGTH)
    ax.set_ylim(0, COURT_WIDTH)
    ax.set_zlim(0, 6)

    ax.set_xlabel("X court length in meters")
    ax.set_ylabel("Y court width in meters")
    ax.set_zlabel("Z height in meters")

    ax.set_title("3D Volleyball Ball Trajectory with Court and Net")


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Cannot find CSV file: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    df = df[df["status"] == "ok"].copy()

    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["Z"] = pd.to_numeric(df["Z"], errors="coerce")

    df = df.dropna(subset=["X", "Y", "Z"])

    # Remove unrealistic triangulation results
    df = df[
        (df["X"] >= -2) & (df["X"] <= 20) &
        (df["Y"] >= -2) & (df["Y"] <= 11) &
        (df["Z"] >= 0) & (df["Z"] <= 8)
    ]

    if len(df) == 0:
        raise RuntimeError("No valid X, Y, Z ball positions found after filtering.")

    xs = df["X"].values
    ys = df["Y"].values
    zs = df["Z"].values

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    draw_3d_court_with_net(ax)

    # Main ball trajectory
    ax.plot(
        xs,
        ys,
        zs,
        color="red",
        marker="o",
        markersize=6,
        linewidth=3,
        label="Ball trajectory"
    )

    # Shadow projection on floor
    ax.plot(
        xs,
        ys,
        [0] * len(xs),
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Floor projection"
    )

    # Vertical height guide lines
    for x, y, z in zip(xs, ys, zs):
        ax.plot(
            [x, x],
            [y, y],
            [0, z],
            color="red",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5
        )

    # Start and end points
    ax.scatter(xs[0], ys[0], zs[0], color="green", s=120, label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="purple", s=140, label="End")

    ax.text(
        xs[0],
        ys[0],
        zs[0],
        "Start",
        color="green",
        fontsize=10
    )

    ax.text(
        xs[-1],
        ys[-1],
        zs[-1],
        f"End\nX={xs[-1]:.2f}\nY={ys[-1]:.2f}\nZ={zs[-1]:.2f}",
        color="purple",
        fontsize=10
    )

    # Better viewing angle
    ax.view_init(elev=28, azim=-55)

    # Make 3D proportions easier to see
    ax.set_box_aspect((18, 9, 6))

    ax.legend()

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved clear 3D graph: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()