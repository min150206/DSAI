# GTSRB YOLOv8

## Sturcture

```
gtsrb_yolo/
├── train.py          ← main script
├── predict.py        ← inference after train run
├── README.md
├── data/
│   └── gtsrb-german-traffic-sign/   ← extract datasheet here
│       ├── Train.csv
│       ├── Test.csv
│       └── Train/
│           ├── 0/
│           ├── 1/
│           └── ...
├── dataset/          ← self-created after convert (skip)
├── gtsrb.yaml        ← automatically self-created (skip)
└── runs/             ← train & predict results (skip)
```

---

## Env setup

```bash
# 2.  PyTorch (chọn 1 trong 2)
# With GPU NVIDIA:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 #more details visit torch website
# Without GPU:
pip install torch torchvision

# 3. install remaining library
pip install ultralytics scikit-learn pillow pandas pyyaml
```

---

## Dataset
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
2. Download & extract to `data/gtsrb-german-traffic-sign/`

---

## Run

```bash
# Train
python train.py


python predict.py --source ./dataset/images/val


python predict.py --source ./my_image.jpg --conf 0.4

# Show (Needs GUI/screen)
python predict.py --source ./my_image.jpg --show
```

---

## Custom vars train.py

| Var | Defaults | Notes |
|------|----------|---------|
| `MODEL_SIZE` | `"s"` | n / s / m / l / x |
| `EPOCHS` | `30` | Tăng lên 50–100 để accuracy cao hơn |
| `BATCH` | `16` | Giảm xuống 8 nếu out of memory |
| `device` | `0` | Đổi thành `"cpu"` nếu không có GPU |

---

## Results after train

- Best weights: `./runs/gtsrb_v1/weights/best.pt`
- Metrics & charts: `./runs/gtsrb_v1/`
- Predict pics: `./runs/predict/`
