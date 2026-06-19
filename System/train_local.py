import os
import zipfile
from datetime import datetime
from pathlib import Path
import torch

import yaml
from ultralytics import YOLO

#In dataset zip files contains:
#<train> folder contain images and label for train
#<valid> contains images and label for validation
#<test> contains images and label for test
#<data.yaml> contain config dataset for YOLO


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
DATA_ZIP_PATH = SYSTEM_DIR / "data_set.zip"
DATASET_DIR = SYSTEM_DIR / "dataset"
RUN_NAME = f"train_yolo11_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
PROJECT_OUTPUT_DIR = SYSTEM_DIR / RUN_NAME

# Training config here
EPOCHS = 34
IMGSZ = 640
BATCH = 4  # Num of img processed together in 1 training step, larger values are faster but take more VRAM, patch = 6,8,16... with higher GPU VRAM (depending on your hardware)
DEVICE = "auto"
MODEL_NAME = "yolo11l.pt"
WORKERS = 4




CODE_MAPPING = {
    "P.102": "No Entry",
    "P.103a": "No Cars",
    "P.103b": "No Right Turn Cars",
    "P.103c": "No Left Turn Cars",
    "P.104": "No Motorcycles",
    "P.106a": "No Trucks",
    "P.106b": "No Heavy Trucks",
    "P.107a": "No Passenger Cars",
    "P.112": "No Pedestrians",
    "P.115": "Limit Weight",
    "P.117": "Limit Height",
    "P.123a": "No Left Turn",
    "P.123b": "No Right Turn",
    "P.124a": "No U-Turn",
    "P.124b": "No U-Turn Cars",
    "P.124c": "No Left Turn & U-Turn",
    "P.125": "No Overtaking",
    "P.127": "Speed Limit",
    "P.128": "End Speed Limit",
    "P.130": "No Stopping/Parking",
    "P.131a": "No Parking",
    "P.137": "No Left/Right Turn",
    "P.245a": "Go Slow",
    "R.301a": "Go Straight",
    "R.301b": "Turn Right Only",
    "R.301c": "Turn Left Only",
    "R.301d": "Must Turn Right",
    "R.301e": "Must Turn Left",
    "R.302a": "Keep Right",
    "R.302b": "Keep Left",
    "R.303": "Roundabout",
    "R.407a": "One Way",
    "R.409": "U-Turn Allowed",
    "R.425": "Hospital",
    "R.434": "Bus Stop",
    "S.509a": "Stop Warning",
    "W.201a": "Curve Left",
    "W.201b": "Curve Right",
    "W.202a": "Sharp Turn Left",
    "W.202b": "Sharp Turn Right",
    "W.203a": "Narrow Road",
    "W.203b": "Narrow Road (Right)",
    "W.203c": "Narrow Road (Left)",
    "W.205a": "Intersection",
    "W.205b": "Intersection",
    "W.205d": "Intersection",
    "W.207a": "Intersection",
    "W.207b": "Intersection",
    "W.207c": "Intersection",
    "W.208": "Priority Intersection",
    "W.209": "Merge Traffic",
    "W.210": "Railroad Crossing",
    "W.219": "Children Crossing",
    "W.221b": "Bumpy Road",
    "W.224": "Pedestrian Crossing",
    "W.225": "Danger: Children",
    "W.227": "Construction",
    "W.233": "Other Danger",
    "W.235": "Slippery Road",
    "W.245a": "Go Slow Warning",
    "DP.135": "End of Restriction",
}





def extract_dataset_if_needed():
    """Extract the dataset archive if the dataset folder does not exist yet."""
    if (SYSTEM_DIR / "data.yaml").exists():
        print(f"Dataset already exists at: {SYSTEM_DIR}")
        return SYSTEM_DIR

    if DATASET_DIR.exists() and (DATASET_DIR / "data.yaml").exists():
        print(f"Dataset already exists at: {DATASET_DIR}")
        return DATASET_DIR

    if not DATA_ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset archive was not found: {DATA_ZIP_PATH}")

    print(f"Extracting dataset from: {DATA_ZIP_PATH}")
    with zipfile.ZipFile(DATA_ZIP_PATH, "r") as archive:
        archive.extractall(SYSTEM_DIR)

    if (SYSTEM_DIR / "data.yaml").exists():
        return SYSTEM_DIR

    if DATASET_DIR.exists() and (DATASET_DIR / "data.yaml").exists():
        return DATASET_DIR

    candidates = [p for p in SYSTEM_DIR.iterdir() if p.is_dir() and p.name not in {"__pycache__", "train_yolo11_final"}]
    if len(candidates) == 1:
        return candidates[0]

    return SYSTEM_DIR





def find_data_yaml(base_dir: Path) -> Path:
    """Search for data.yaml inside the extracted dataset folder."""
    for root, _, files in os.walk(base_dir):
        if "data.yaml" in files:
            return Path(root) / "data.yaml"

    raise FileNotFoundError(f"Could not find data.yaml inside: {base_dir}")





def update_class_names(data_yaml_path: Path) -> None:
    """Replace the original class names with English names for readability."""
    with open(data_yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names", [])
    if isinstance(names, list):
        updated_names = {}
        for index, name in enumerate(names):
            clean_name = str(name).strip()
            updated_names[index] = CODE_MAPPING.get(clean_name, clean_name)
        data["names"] = updated_names
    elif isinstance(names, dict):
        updated_names = {}
        for index, name in names.items():
            clean_name = str(name).strip()
            updated_names[index] = CODE_MAPPING.get(clean_name, clean_name)
        data["names"] = updated_names

    with open(data_yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    print(f"Updated class names in: {data_yaml_path}")





def prepare_data_yaml(data_yaml_path: Path) -> Path:
    """Make sure the local dataset paths in data.yaml match the current folder structure."""
    with open(data_yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in: {data_yaml_path}")

    base_dir = data_yaml_path.parent
    expected_pairs = [
        ("train", base_dir / "train" / "images"),
        ("val", base_dir / "valid" / "images"),
        ("test", base_dir / "test" / "images"),
    ]

    for key, expected_path in expected_pairs:
        if expected_path.exists() and key in data:
            data[key] = str(expected_path.relative_to(base_dir)).replace("\\", "/")

    with open(data_yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    print(f"Prepared dataset paths in: {data_yaml_path}")
    return data_yaml_path





def resume_training(checkpoint_path: str):
   
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    print(f"Resuming training from: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    model.train(resume=True)

    print("Resume training finished.")





def main():
    
    #Data extract and processing
    dataset_root = extract_dataset_if_needed()
    data_yaml_path = find_data_yaml(dataset_root)
    update_class_names(data_yaml_path)
    prepare_data_yaml(data_yaml_path)
    
    
    
    
    
    #resume_training('./train_yolo11_final/weights/last.pt')  #insert the dir of last.pt here






    if DEVICE is None or DEVICE == "auto":
        if torch.cuda.is_available():
            device = 0
        else:
            device = "cpu"
    else:
        device = DEVICE

    print("Training configuration:")
    print(f"- Project root: {PROJECT_ROOT}")
    print(f"- Output folder: {PROJECT_OUTPUT_DIR}")
    print(f"- Dataset YAML: {data_yaml_path}")
    print(f"- Device: {device}")
    print(f"- Epochs: {EPOCHS}")

    PROJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(data_yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(SYSTEM_DIR),
        name=RUN_NAME,
        device=device,
        optimizer="auto",
        plots=True,
        save=True,
        exist_ok=True,
        workers=WORKERS,
        amp=True,
    )

    print("Training finished.")
    print(f"Results are stored in: {PROJECT_OUTPUT_DIR}")
    
    



if __name__ == "__main__":
    main()
