import cv2
import numpy as np
import pandas as pd


def load_calibration_points(court_csv, image_csv):
    court_df = pd.read_csv(court_csv)
    image_df = pd.read_csv(image_csv)

    merged = pd.merge(court_df, image_df, on="point_name", how="inner")

    if len(merged) < 6:
        raise ValueError("At least 6 matching court and image points are recommended for solvePnP.")

    object_points = merged[["X", "Y", "Z"]].values.astype(np.float32)
    image_points = merged[["u", "v"]].values.astype(np.float32)

    point_names = merged["point_name"].tolist()

    return object_points, image_points, point_names


def estimate_camera_matrix(image_width, image_height, focal_length=None):
    if focal_length is None:
        focal_length = max(image_width, image_height)

    camera_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    return camera_matrix, dist_coeffs


def solve_camera_pose(object_points, image_points, camera_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP failed. Check your calibration points.")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    projection_matrix = camera_matrix @ np.hstack((rotation_matrix, tvec))

    return rvec, tvec, rotation_matrix, projection_matrix


def solve_camera_pose_ransac(object_points, image_points, camera_matrix, dist_coeffs):
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        iterationsCount=1000,
        reprojectionError=8.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnPRansac failed. Check your calibration points.")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    projection_matrix = camera_matrix @ np.hstack((rotation_matrix, tvec))

    return rvec, tvec, rotation_matrix, projection_matrix, inliers


def reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(image_points - projected_points, axis=1)

    return float(np.mean(errors)), errors, projected_points


def find_best_focal_length(
    object_points,
    image_points,
    image_width,
    image_height,
    focal_lengths=None,
    use_ransac=False
):
    if focal_lengths is None:
        focal_lengths = list(range(800, 4001, 100))

    best_result = None
    all_results = []

    for focal_length in focal_lengths:
        try:
            camera_matrix, dist_coeffs = estimate_camera_matrix(
                image_width,
                image_height,
                focal_length=focal_length
            )

            if use_ransac:
                rvec, tvec, rotation_matrix, projection_matrix, inliers = solve_camera_pose_ransac(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs
                )
            else:
                rvec, tvec, rotation_matrix, projection_matrix = solve_camera_pose(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs
                )
                inliers = None

            mean_error, point_errors, projected_points = reprojection_error(
                object_points,
                image_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )

            result = {
                "focal_length": focal_length,
                "mean_error": mean_error,
                "point_errors": point_errors,
                "projected_points": projected_points,
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "rvec": rvec,
                "tvec": tvec,
                "rotation_matrix": rotation_matrix,
                "projection_matrix": projection_matrix,
                "inliers": inliers
            }

            all_results.append(result)

            if best_result is None or mean_error < best_result["mean_error"]:
                best_result = result

            print(f"Focal length {focal_length}: reprojection error = {mean_error:.2f} pixels")

        except Exception as e:
            print(f"Focal length {focal_length}: failed - {e}")

    if best_result is None:
        raise RuntimeError("No valid focal length found.")

    print()
    print("Best focal length result")
    print(f"Best focal length: {best_result['focal_length']}")
    print(f"Best reprojection error: {best_result['mean_error']:.2f} pixels")

    return best_result, all_results


def triangulate_two_views(P1, P2, point1, point2):
    pts1 = np.array(point1, dtype=np.float64).reshape(2, 1)
    pts2 = np.array(point2, dtype=np.float64).reshape(2, 1)

    point_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    if abs(point_4d[3, 0]) < 1e-9:
        raise RuntimeError("Triangulation failed because homogeneous coordinate is close to zero.")

    point_3d = point_4d[:3, 0] / point_4d[3, 0]

    return point_3d