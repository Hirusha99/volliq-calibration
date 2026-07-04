import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from utils.camera_utils import triangulate_two_views
from utils.court_plot import save_single_position_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = PROJECT_ROOT / "calibration" / "camera_params.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CAM1_IMAGE = PROJECT_ROOT / "data" / "cam1_frames" / "frame_0050.jpg"
CAM2_IMAGE = PROJECT_ROOT / "data" / "cam2_frames" / "frame_0050.jpg"


clicked_points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        img = param
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click ball center", img)


def get_clicked_point(image_path):
    global clicked_points
    clicked_points = []

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    display = image.copy()

    cv2.imshow("Click ball center", display)
    cv2.setMouseCallback("Click ball center", click_event, display)

    print(f"Click the ball center in: {image_path}")
    print("Press ENTER after clicking.")

    while True:
        key = cv2.waitKey(1)

        if key == 13:
            break

        if key == 27:
            raise RuntimeError("Cancelled by user.")

    cv2.destroyAllWindows()

    if not clicked_points:
        raise RuntimeError("No point clicked.")

    return clicked_points[-1]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    calibration = np.load(CALIBRATION_FILE)
    P1 = calibration["P1"]
    P2 = calibration["P2"]

    point1 = get_clicked_point(CAM1_IMAGE)
    point2 = get_clicked_point(CAM2_IMAGE)

    X, Y, Z = triangulate_two_views(P1, P2, point1, point2)

    print("Ball 3D position:")
    print(f"X = {X:.3f} m")
    print(f"Y = {Y:.3f} m")
    print(f"Z = {Z:.3f} m")

    save_single_position_map(float(X), float(Y), float(Z), OUTPUT_DIR / "manual_court_map.png")

    pd.DataFrame([{
        "cam1_u": point1[0],
        "cam1_v": point1[1],
        "cam2_u": point2[0],
        "cam2_v": point2[1],
        "X": X,
        "Y": Y,
        "Z": Z
    }]).to_csv(OUTPUT_DIR / "manual_ball_position.csv", index=False)

    print(f"Saved: {OUTPUT_DIR / 'manual_court_map.png'}")
    print(f"Saved: {OUTPUT_DIR / 'manual_ball_position.csv'}")


if __name__ == "__main__":
    main()