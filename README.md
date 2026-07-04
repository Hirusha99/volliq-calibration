# Volleyball Ball 3D Tracking and 2D Court Mapping

This project detects a volleyball from two synchronized camera views, estimates the ball's 3D court position using triangulation, and maps the ball location onto a 2D volleyball court.

## Folder structure

```text
volleyball_ball_3d_project/
├── data/
│   ├── cam1_frames/
│   └── cam2_frames/
├── calibration/
│   ├── court_points_3d.csv
│   ├── cam1_points_2d.csv
│   └── cam2_points_2d.csv
├── outputs/
├── src/
│   ├── calibrate_cameras.py
│   ├── detect_and_map.py
│   └── utils/
│       ├── camera_utils.py
│       ├── court_plot.py
│       └── yolo_ball_detector.py
├── requirements.txt
└── README.md
```

## Step 1: Install packages

```bash
pip install -r requirements.txt
```

## Step 2: Add your frames

Put synchronized frames here:

```text
data/cam1_frames/
data/cam2_frames/
```

The same timestamp frames must have the same filename.

Example:

```text
data/cam1_frames/frame_0001.jpg
data/cam2_frames/frame_0001.jpg
```

## Step 3: Prepare calibration CSV files

You must mark the same court points in both camera views.

Use real volleyball court coordinates in meters.

Example court coordinate system:

```text
X direction = court length, 0 to 18 m
Y direction = court width, 0 to 9 m
Z direction = height, floor is 0
```

CSV files are included as templates.

## Step 4: Calibrate cameras

```bash
python src/calibrate_cameras.py
```

This creates:

```text
calibration/camera_params.npz
```

## Step 5: Add your trained YOLO ball model

Put your trained volleyball detection model in the project folder, for example:

```text
ball_model.pt
```

The model should detect the volleyball class.

## Step 6: Run detection, triangulation, and 2D mapping

```bash
python src/detect_and_map.py --model ball_model.pt
```

Outputs are saved in:

```text
outputs/ball_positions.csv
outputs/court_map.png
outputs/court_trajectory.png
```

## Important notes

This project requires calibrated cameras.

If the camera moves, zoom changes, video is cropped, or resolution changes, calibration must be repeated.

If the ball is visible in only one camera for a frame, the project skips that frame because 3D triangulation requires two views.