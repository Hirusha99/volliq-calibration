import cv2
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POINT_NAMES = [
    "corner_left_near",
    "corner_right_near",
    "corner_left_far",
    "corner_right_far",
    "center_left",
    "center_right",
    "attack_left_near",
    "attack_right_near",
    "attack_left_far",
    "attack_right_far"
]


class PointClicker:
    def __init__(self, image_path, display_width=1280):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))

        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        self.original_height, self.original_width = self.image.shape[:2]

        self.scale = display_width / self.original_width
        self.display_width = int(self.original_width * self.scale)
        self.display_height = int(self.original_height * self.scale)

        self.display_base = cv2.resize(
            self.image,
            (self.display_width, self.display_height)
        )

        self.display = self.display_base.copy()
        self.points = []
        self.current_index = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.current_index >= len(POINT_NAMES):
            return

        point_name = POINT_NAMES[self.current_index]

        original_x = int(x / self.scale)
        original_y = int(y / self.scale)

        self.points.append((point_name, original_x, original_y))

        cv2.circle(self.display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            self.display,
            point_name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        self.current_index += 1
        self.show_instruction()

    def show_instruction(self):
        temp = self.display.copy()

        if self.current_index < len(POINT_NAMES):
            text = f"Click: {POINT_NAMES[self.current_index]}"
        else:
            text = "All points clicked. Press ENTER to save."

        cv2.putText(
            temp,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow("Calibration Point Clicker", temp)

    def run(self):
        cv2.namedWindow("Calibration Point Clicker", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Point Clicker", self.display)
        cv2.setMouseCallback("Calibration Point Clicker", self.mouse_callback)

        self.show_instruction()

        while True:
            key = cv2.waitKey(1)

            if key == 13:
                break

            if key == 27:
                raise RuntimeError("Cancelled by user.")

        cv2.destroyAllWindows()

        return self.points


def main():
    image_path = input("Enter image path: ").strip()
    output_csv = input("Enter output CSV path, for example calibration/cam1_points_2d.csv: ").strip()

    clicker = PointClicker(Path(image_path), display_width=1280)
    points = clicker.run()

    df = pd.DataFrame(points, columns=["point_name", "u", "v"])
    df.to_csv(PROJECT_ROOT / output_csv, index=False)

    print(f"Saved: {PROJECT_ROOT / output_csv}")


if __name__ == "__main__":
    main()