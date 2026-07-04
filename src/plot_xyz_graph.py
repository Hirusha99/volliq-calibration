import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = PROJECT_ROOT / "outputs" / "ball_positions.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "ball_xyz_court_net_graph.png"


COURT_LENGTH = 18
COURT_WIDTH = 9

# Change this based on your match
# Men's volleyball: 2.43
# Women's volleyball: 2.24
NET_HEIGHT = 2.43


def draw_3d_court_with_net(ax):
    # Court boundary on floor
    ax.plot([0, COURT_LENGTH], [0, 0], [0, 0], linewidth=2)
    ax.plot([0, COURT_LENGTH], [COURT_WIDTH, COURT_WIDTH], [0, 0], linewidth=2)
    ax.plot([0, 0], [0, COURT_WIDTH], [0, 0], linewidth=2)
    ax.plot([COURT_LENGTH, COURT_LENGTH], [0, COURT_WIDTH], [0, 0], linewidth=2)

    # Center line / net line on floor
    ax.plot([9, 9], [0, COURT_WIDTH], [0, 0], linestyle="--", linewidth=2)

    # Attack lines
    ax.plot([3, 3], [0, COURT_WIDTH], [0, 0], linestyle=":", linewidth=1.5)
    ax.plot([15, 15], [0, COURT_WIDTH], [0, 0], linestyle=":", linewidth=1.5)

    # Net top line
    ax.plot([9, 9], [0, COURT_WIDTH], [NET_HEIGHT, NET_HEIGHT], linewidth=3)

    # Net side poles
    ax.plot([9, 9], [0, 0], [0, NET_HEIGHT], linewidth=2)
    ax.plot([9, 9], [COURT_WIDTH, COURT_WIDTH], [0, NET_HEIGHT], linewidth=2)

    # Net vertical guide lines
    for y in [1.5, 3, 4.5, 6, 7.5]:
        ax.plot([9, 9], [y, y], [0, NET_HEIGHT], linestyle="--", linewidth=0.7)

    # Net horizontal guide lines
    for z in [0.5, 1.0, 1.5, 2.0]:
        ax.plot([9, 9], [0, COURT_WIDTH], [z, z], linestyle="--", linewidth=0.7)

    # Labels
    ax.text(9, COURT_WIDTH + 0.3, NET_HEIGHT, "Net", fontsize=10)
    ax.text(3, COURT_WIDTH + 0.3, 0, "Attack line", fontsize=8)
    ax.text(15, COURT_WIDTH + 0.3, 0, "Attack line", fontsize=8)

    # Axis limits
    ax.set_xlim(0, COURT_LENGTH)
    ax.set_ylim(0, COURT_WIDTH)
    ax.set_zlim(0, 6)

    ax.set_xlabel("X coordinate / court length (m)")
    ax.set_ylabel("Y coordinate / court width (m)")
    ax.set_zlabel("Z coordinate / height (m)")

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

    if len(df) == 0:
        raise RuntimeError("No valid X, Y, Z ball positions found.")

    xs = df["X"].values
    ys = df["Y"].values
    zs = df["Z"].values

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    draw_3d_court_with_net(ax)

    # Ball trajectory
    ax.plot(xs, ys, zs, marker="o", markersize=4, linewidth=1.5, label="Ball trajectory")

    # Start and end points
    ax.scatter(xs[0], ys[0], zs[0], s=80, label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], s=100, label="End")

    ax.text(
        xs[-1],
        ys[-1],
        zs[-1],
        f"End\nX={xs[-1]:.2f}\nY={ys[-1]:.2f}\nZ={zs[-1]:.2f}",
        fontsize=8
    )

    ax.legend()

    # Better viewing angle
    ax.view_init(elev=25, azim=-60)

    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved 3D court-net graph: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()