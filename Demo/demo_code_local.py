from pathlib import Path

import cv2
from ultralytics import YOLO


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "Demo" / "best.pt"
OUTPUT_FOLDER = PROJECT_ROOT / "Demo" / "output"
CONFIDENCE = 0.5

# Default sample files
VIDEO_SOURCE = PROJECT_ROOT / "Demo" / "datatest" / "demo_video.mp4" #changes video dir and img dir here
IMAGE_SOURCE = PROJECT_ROOT / "Demo" / "datatest" / "test_img_8.jpg"


def get_source_type(source: str) -> str:
    """Return whether the input is an image, video, or unsupported file."""
    if isinstance(source, int):
        return "video"

    ext = Path(source).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return "image"
    if ext in {".mp4", ".avi", ".mkv", ".mov"}:
        return "video"
    return "unknown"


def run_on_image(model, image_path: str) -> None:
    """Run inference on a single image and save the annotated result."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model(frame, conf=CONFIDENCE, verbose=False)
    annotated_frame = results[0].plot()

    output_path = OUTPUT_FOLDER / f"result_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"Image saved to: {output_path}")

    cv2.imshow("Detection Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(model, video_path: str) -> None:
    """Run inference on a video and save the annotated output."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = OUTPUT_FOLDER / f"result_{Path(video_path).name}"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        frame_count += 1
        if frame_count % 20 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    writer.release()
    print("Video processing completed.")


def main() -> None:
    """Load the model and run inference on the selected source."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    
    #----------------------------------------------
    source_path = IMAGE_SOURCE
    if not source_path.exists():
        raise FileNotFoundError(f"Image file not found: {source_path}")
    run_on_image(model, source_path)
        
    #----------------------------------------------
    
'''
    print("Choose input type:")
    print("1. Image")
    print("2. Video")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        source_path = IMAGE_SOURCE
        if not source_path.exists():
            raise FileNotFoundError(f"Image file not found: {source_path}")
        run_on_image(model, source_path)
    elif choice == "2":
        source_path = VIDEO_SOURCE
        if not source_path.exists():
            raise FileNotFoundError(f"Video file not found: {source_path}")
        run_on_video(model, source_path)
    else:
        raise ValueError("Invalid choice. Please enter 1 for image or 2 for video.")
'''

if __name__ == "__main__":
    main()


#Detect ảnh nha, con nếu muốn chọn 1 trong 2 option thi bỏ cái comment cam và đóng comment cái khung xanh đi (trong main())
#Code này là nhận data và trả về file data output chứ k phải trực tiếp như code ở ngoài folder chính
