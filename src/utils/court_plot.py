import matplotlib.pyplot as plt


def draw_court_base():
    court_length = 18
    court_width = 9

    plt.plot([0, 18, 18, 0, 0], [0, 0, 9, 9, 0], linewidth=2)

    plt.axvline(9, linestyle="--", linewidth=1)
    plt.axvline(3, linestyle=":", linewidth=1)
    plt.axvline(15, linestyle=":", linewidth=1)

    plt.text(9, 9.2, "Net line", ha="center")
    plt.text(3, 9.2, "Attack line", ha="center")
    plt.text(15, 9.2, "Attack line", ha="center")

    plt.xlim(-0.5, court_length + 0.5)
    plt.ylim(-0.5, court_width + 0.8)
    plt.xlabel("Court X length in meters")
    plt.ylabel("Court Y width in meters")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)


def save_single_position_map(x, y, z, output_path):
    plt.figure(figsize=(10, 5))
    draw_court_base()
    plt.scatter([x], [y], s=90)
    plt.text(x, y, f"Ball Z={z:.2f}m", fontsize=9)
    plt.title("Ball Position on 2D Volleyball Court")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_trajectory_map(xs, ys, zs, output_path):
    plt.figure(figsize=(10, 5))
    draw_court_base()
    plt.plot(xs, ys, marker="o", markersize=3)
    if len(xs) > 0:
        plt.scatter([xs[-1]], [ys[-1]], s=90)
        plt.text(xs[-1], ys[-1], f"Latest Z={zs[-1]:.2f}m", fontsize=9)
    plt.title("Ball Trajectory on 2D Volleyball Court")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()