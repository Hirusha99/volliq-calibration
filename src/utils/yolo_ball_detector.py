from ultralytics import YOLO


class YOLOBallDetector:
    def __init__(self, model_path, confidence=0.25, ball_class_name=None, imgsz=640, class_id=None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.ball_class_name = ball_class_name
        self.imgsz = imgsz
        self.class_id = class_id

    def detect_ball_center(self, image):
        results = self.model(
            image,
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=self.class_id,
            verbose=False
        )[0]

        best_box = None
        best_conf = -1.0

        names = results.names

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))

            if self.ball_class_name is not None:
                if cls_name.lower() != self.ball_class_name.lower():
                    continue

            if conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            return None, None

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().tolist()

        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2

        detection = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": best_conf
        }

        return (center_u, center_v), detection