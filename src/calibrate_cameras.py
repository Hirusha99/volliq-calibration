import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from utils.camera_utils import (
    load_calibration_points,
    find_best_focal_length
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAM1_FRAME_DIR = PROJECT_ROOT / "data" / "cam1_frames"
CAM2_FRAME_DIR = PROJECT_ROOT / "data" / "cam2_frames"

COURT_POINTS = PROJECT_ROOT / "calibration" / "court_points_3d.csv"
CAM1_POINTS = PROJECT_ROOT / "calibration" / "cam1_points_2d.csv"
CAM2_POINTS = PROJECT_ROOT / "calibration" / "cam2_points_2d.csv"

OUTPUT_FILE = PROJECT_ROOT / "calibration" / "camera_params.npz"


def first_image_in_folder(folder):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []

    for ext in extensions:
        files.extend(sorted(folder.glob(ext)))

    if not files:
        raise FileNotFoundError(f"No image found in {folder}")

    return files[0]


def print_point_errors(camera_name, point_names, point_errors):
    print()
    print(f"{camera_name} point-wise reprojection errors:")

    for name, error in zip(point_names, point_errors):
        print(f"{name}: {error:.2f} pixels")


def main():
    cam1_image_path = first_image_in_folder(CAM1_FRAME_DIR)
    cam2_image_path = first_image_in_folder(CAM2_FRAME_DIR)

    cam1_img = cv2.imread(str(cam1_image_path))
    cam2_img = cv2.imread(str(cam2_image_path))

    if cam1_img is None:
        raise RuntimeError(f"Could not read {cam1_image_path}")

    if cam2_img is None:
        raise RuntimeError(f"Could not read {cam2_image_path}")

    h1, w1 = cam1_img.shape[:2]
    h2, w2 = cam2_img.shape[:2]

    obj1, img1, point_names1 = load_calibration_points(COURT_POINTS, CAM1_POINTS)
    obj2, img2, point_names2 = load_calibration_points(COURT_POINTS, CAM2_POINTS)

    focal_lengths = list(range(800, 4001, 100))

    print()
    print("Testing focal lengths for Camera 1")
    best_cam1, results_cam1 = find_best_focal_length(
        object_points=obj1,
        image_points=img1,
        image_width=w1,
        image_height=h1,
        focal_lengths=focal_lengths,
        use_ransac=False
    )

    print()
    print("Testing focal lengths for Camera 2")
    best_cam2, results_cam2 = find_best_focal_length(
        object_points=obj2,
        image_points=img2,
        image_width=w2,
        image_height=h2,
        focal_lengths=focal_lengths,
        use_ransac=False
    )

    print_point_errors(
        "Camera 1",
        point_names1,
        best_cam1["point_errors"]
    )

    print_point_errors(
        "Camera 2",
        point_names2,
        best_cam2["point_errors"]
    )

    np.savez(
        OUTPUT_FILE,

        focal_length_cam1=best_cam1["focal_length"],
        K1=best_cam1["camera_matrix"],
        dist1=best_cam1["dist_coeffs"],
        rvec1=best_cam1["rvec"],
        tvec1=best_cam1["tvec"],
        R1=best_cam1["rotation_matrix"],
        P1=best_cam1["projection_matrix"],
        reprojection_error_cam1=best_cam1["mean_error"],

        focal_length_cam2=best_cam2["focal_length"],
        K2=best_cam2["camera_matrix"],
        dist2=best_cam2["dist_coeffs"],
        rvec2=best_cam2["rvec"],
        tvec2=best_cam2["tvec"],
        R2=best_cam2["rotation_matrix"],
        P2=best_cam2["projection_matrix"],
        reprojection_error_cam2=best_cam2["mean_error"]
    )

    print()
    print("Camera calibration completed.")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Camera 1 best focal length: {best_cam1['focal_length']}")
    print(f"Camera 1 mean reprojection error: {best_cam1['mean_error']:.2f} pixels")
    print(f"Camera 2 best focal length: {best_cam2['focal_length']}")
    print(f"Camera 2 mean reprojection error: {best_cam2['mean_error']:.2f} pixels")

    print()
    print("Important:")
    print("If reprojection error is still high, check point-wise errors.")
    print("Re-click points with very large error.")
    print("Try adding net-top points with real Z height.")
    print("For best accuracy, use real chessboard camera calibration.")


if __name__ == "__main__":
    main()