import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAM1_FRAME_DIR = PROJECT_ROOT / "data" / "cam1_frames"
CAM2_FRAME_DIR = PROJECT_ROOT / "data" / "cam2_frames"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "threshold_test"


def draw_boxes(image, results, target_class_name="sports ball"):
    output = image.copy()
    names = results.names

    detected_count = 0

    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))

        if cls_name.lower() != target_class_name.lower():
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        detected_count += 1

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{cls_name} {conf:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cv2.circle(output, (center_x, center_y), 4, (0, 0, 255), -1)

    return output, detected_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--frame", required=True, help="Example: frame_0050.jpg")
    parser.add_argument("--class_name", default="sports ball")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cam1_path = CAM1_FRAME_DIR / args.frame
    cam2_path = CAM2_FRAME_DIR / args.frame

    if not cam1_path.exists():
        raise FileNotFoundError(f"Camera 1 frame not found: {cam1_path}")

    if not cam2_path.exists():
        raise FileNotFoundError(f"Camera 2 frame not found: {cam2_path}")

    img1 = cv2.imread(str(cam1_path))
    img2 = cv2.imread(str(cam2_path))

    if img1 is None:
        raise RuntimeError(f"Could not read image: {cam1_path}")

    if img2 is None:
        raise RuntimeError(f"Could not read image: {cam2_path}")

    model = YOLO(args.model)

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60
    ]

    for conf in thresholds:
        result1 = model(img1, conf=conf, verbose=False)[0]
        result2 = model(img2, conf=conf, verbose=False)[0]

        out1, count1 = draw_boxes(img1, result1, args.class_name)
        out2, count2 = draw_boxes(img2, result2, args.class_name)

        h = min(out1.shape[0], out2.shape[0])

        out1 = cv2.resize(out1, (int(out1.shape[1] * h / out1.shape[0]), h))
        out2 = cv2.resize(out2, (int(out2.shape[1] * h / out2.shape[0]), h))

        combined = cv2.hconcat([out1, out2])

        output_file = OUTPUT_DIR / f"{args.frame.replace('.', '_')}_conf_{conf:.2f}.jpg"
        cv2.imwrite(str(output_file), combined)

        print(
            f"Confidence {conf:.2f}: "
            f"Camera 1 sports ball detections = {count1}, "
            f"Camera 2 sports ball detections = {count2}, "
            f"saved = {output_file}"
        )


if __name__ == "__main__":
    main()