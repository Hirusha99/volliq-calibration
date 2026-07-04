import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.camera_utils import triangulate_two_views
from utils.yolo_ball_detector import YOLOBallDetector
from utils.court_plot import save_single_position_map, save_trajectory_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAM1_FRAME_DIR = PROJECT_ROOT / "data" / "cam1_frames"
CAM2_FRAME_DIR = PROJECT_ROOT / "data" / "cam2_frames"

CALIBRATION_FILE = PROJECT_ROOT / "calibration" / "camera_params.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def list_frame_pairs(cam1_dir, cam2_dir):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    cam1_files = {
        p.name: p for p in cam1_dir.iterdir()
        if p.suffix.lower() in extensions
    }

    cam2_files = {
        p.name: p for p in cam2_dir.iterdir()
        if p.suffix.lower() in extensions
    }

    common_names = sorted(set(cam1_files.keys()) & set(cam2_files.keys()))

    pairs = [(name, cam1_files[name], cam2_files[name]) for name in common_names]

    return pairs


def draw_detection(image, detection, label):
    if detection is None:
        return image

    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    conf = detection["confidence"]

    out = image.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        out,
        f"{label} {conf:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained YOLO ball detection model.")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--ball_class_name", default=None, help="Optional class name, for example ball or volleyball.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--class_id", type=int, default=None, help="YOLO class ID. Example: 0 for volleyball.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            "Calibration file not found. Run python src/calibrate_cameras.py first."
        )

    calibration = np.load(CALIBRATION_FILE)
    P1 = calibration["P1"]
    P2 = calibration["P2"]

    detector = YOLOBallDetector(
        model_path=args.model,
        confidence=args.confidence,
        ball_class_name=args.ball_class_name,
        imgsz=args.imgsz,
        class_id=args.class_id
    )

    frame_pairs = list_frame_pairs(CAM1_FRAME_DIR, CAM2_FRAME_DIR)

    if not frame_pairs:
        raise FileNotFoundError("No matching frame names found in cam1_frames and cam2_frames.")

    rows = []
    xs = []
    ys = []
    zs = []

    detection_output_dir = OUTPUT_DIR / "detections"
    detection_output_dir.mkdir(exist_ok=True)

    for frame_name, cam1_path, cam2_path in tqdm(frame_pairs):
        img1 = cv2.imread(str(cam1_path))
        img2 = cv2.imread(str(cam2_path))

        if img1 is None or img2 is None:
            continue

        point1, det1 = detector.detect_ball_center(img1)
        point2, det2 = detector.detect_ball_center(img2)

        if point1 is None or point2 is None:
            rows.append({
                "frame": frame_name,
                "status": "skipped_ball_not_visible_in_both_cameras",
                "cam1_u": None,
                "cam1_v": None,
                "cam2_u": None,
                "cam2_v": None,
                "X": None,
                "Y": None,
                "Z": None
            })
            continue

        try:
            X, Y, Z = triangulate_two_views(P1, P2, point1, point2)
            status = "ok"
        except Exception as e:
            X, Y, Z = None, None, None
            status = f"triangulation_failed: {e}"

        rows.append({
            "frame": frame_name,
            "status": status,
            "cam1_u": point1[0],
            "cam1_v": point1[1],
            "cam2_u": point2[0],
            "cam2_v": point2[1],
            "X": X,
            "Y": Y,
            "Z": Z
        })

        if status == "ok":
            xs.append(float(X))
            ys.append(float(Y))
            zs.append(float(Z))

            save_single_position_map(
                float(X),
                float(Y),
                float(Z),
                OUTPUT_DIR / "court_map.png"
            )

        debug1 = draw_detection(img1, det1, "ball")
        debug2 = draw_detection(img2, det2, "ball")

        combined_height = min(debug1.shape[0], debug2.shape[0])
        debug1_resized = cv2.resize(debug1, (int(debug1.shape[1] * combined_height / debug1.shape[0]), combined_height))
        debug2_resized = cv2.resize(debug2, (int(debug2.shape[1] * combined_height / debug2.shape[0]), combined_height))
        combined = np.hstack([debug1_resized, debug2_resized])

        cv2.imwrite(str(detection_output_dir / frame_name), combined)

    output_csv = OUTPUT_DIR / "ball_positions.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    if len(xs) > 0:
        save_trajectory_map(xs, ys, zs, OUTPUT_DIR / "court_trajectory.png")

    print("Processing completed.")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved detection images: {detection_output_dir}")
    print(f"Saved latest court map: {OUTPUT_DIR / 'court_map.png'}")
    print(f"Saved trajectory map: {OUTPUT_DIR / 'court_trajectory.png'}")


if __name__ == "__main__":
    main()