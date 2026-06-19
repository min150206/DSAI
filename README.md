# Vietnam Traffic Sign Detection

YOLO project for detecting Vietnamese traffic signs in images and videos.

## What's inside (note that on Github is only source code)

- `predict_video.py` - plays a video with detection boxes on screen.
- `Demo/demo_code_local.py` - runs detection on an image or video and saves the result.
- `Demo/datatest/` - put test images and videos here.
- `Demo/output/` - results are saved here.
- `Demo/best.pt` - model used by `Demo/demo_code_local.py`.
- `System/train_yolo11_final/weights/best.pt` - model used by `predict_video.py`.
- `System/train_local.py` - train a new model.

## Install

Open a terminal in this project folder and install the needed packages:

```bash
pip install ultralytics opencv-python pyyaml torch
```

If you have an NVIDIA GPU, install the correct CUDA version of PyTorch from:

```text
https://pytorch.org/get-started/locally/
```

## Download a video

Download any `.mp4` video that you want to test.

Rename it to:

```text
demo_video.mp4
```

Place it here:

```text
Demo/datatest/demo_video.mp4
```

The project already has this example path in the code.

## Run video detection on screen

This command opens the video and shows detection boxes while it plays:

```bash
python predict_video.py
```

Press `q` to stop the video.

Note: `predict_video.py` uses GPU with `device = 0`. If you do not have an NVIDIA GPU, open `predict_video.py` and change this line:

```python
results = model.predict(frame, conf=CONF, verbose=False, device = 0)[0]
```

to:

```python
results = model.predict(frame, conf=CONF, verbose=False)[0]
```

## Run image example and save output

This command runs the default image example:

```bash
python Demo/demo_code_local.py
```

Default input:

```text
Demo/datatest/test_img_8.jpg
```

Default output:

```text
Demo/output/result_test_img_8.jpg
```

## Use your own image

Put your image in:

```text
Demo/datatest/
```

Then open `Demo/demo_code_local.py` and change:

```python
IMAGE_SOURCE = PROJECT_ROOT / "Demo" / "datatest" / "test_img_8.jpg"
```

Example:

```python
IMAGE_SOURCE = PROJECT_ROOT / "Demo" / "datatest" / "my_image.jpg"
```

Run:

```bash
python Demo/demo_code_local.py
```

## Save a processed video

In `Demo/demo_code_local.py`, change the selected source in `main()` from image to video:

```python
source_path = VIDEO_SOURCE
if not source_path.exists():
    raise FileNotFoundError(f"Video file not found: {source_path}")
run_on_video(model, source_path)
```

Then run:

```bash
python Demo/demo_code_local.py
```

The saved video will be created in:

```text
Demo/output/result_demo_video.mp4
```

## Train a new model

Dataset files are in:

```text
System/train/
System/valid/
System/test/
System/data.yaml
```

Run training:

```bash
python System/train_local.py
```

Training results will be saved in a new folder like:

```text
System/train_yolo11_final_YYYYMMDD_HHMMSS/
```

The best trained model will be inside:

```text
System/train_yolo11_final_YYYYMMDD_HHMMSS/weights/best.pt
```

## Quick example

1. Download a traffic video.
2. Rename it to `demo_video.mp4`.
3. Put it in `Demo/datatest/`.
4. Run:

```bash
python predict_video.py
```

5. Watch the detection window. Press `q` when finished.
