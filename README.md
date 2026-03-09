# Bee Pose Estimation — MLOps Project

Визначення пози бджіл на прилітній дошці вулика за допомогою YOLO Pose з відстеженням експериментів через MLflow (DagsHub).

## Задача

Fine-tuning YOLOv11n-pose на датасеті бджіл:
- **400 зображень** з 8 вуликів (50 кадрів кожний)
- **2 ключові точки** на бджолу: голова та жало
- Формат анотацій: YOLO Pose (`class cx cy w h kp1x kp1y kp2x kp2y`)

## Структура проєкту

```
├── src/
│   ├── config.py          # Конфігурація, credentials, DVC setup
│   ├── train.py           # Тренування з CLI аргументами
│   ├── data.py            # Train/val split по вуликах
│   └── tracking.py        # MLflow логування (DagsHub)
├── notebooks/
│   └── 01_eda.ipynb       # Розвідувальний аналіз даних
├── data/raw/pose/         # Сирі дані (DVC)
├── runs/                  # Результати YOLO (не в git)
└── mlruns/                # MLflow local fallback (не в git)
```

## Встановлення

```bash
git clone https://github.com/spiowm/mlops-project.git
cd mlops-project
uv sync
```

### Credentials

Створити `.env` (шаблон: `.env.example`):
```
DAGSHUB_TOKEN=your_token
DAGSHUB_USER=your_username
DAGSHUB_REPO=your_repo_name
```

### Датасет (DVC)

```bash
python src/config.py   # налаштовує DVC credentials
dvc pull               # завантажує датасет (~360 MB)
```

## Запуск тренування

```bash
python src/train.py --epochs 50 --batch 16

# З іншими гіперпараметрами
python src/train.py --epochs 100 --lr0 0.005 --optimizer SGD
```

### CLI аргументи

| Аргумент | Default | Опис |
|----------|---------|------|
| `--epochs` | 50 | Кількість епох |
| `--batch` | 16 | Розмір батчу |
| `--imgsz` | 640 | Розмір зображення |
| `--lr0` | 0.01 | Initial learning rate |
| `--optimizer` | AdamW | SGD / Adam / AdamW |
| `--model` | yolo11n-pose.pt | Pretrained модель |
| `--force-split` | — | Перестворити train/val split |

## Google Colab

1. Додати `DAGSHUB_TOKEN`, `DAGSHUB_USER`, `DAGSHUB_REPO` у Colab Secrets (🔑)

```bash
!git clone https://github.com/spiowm/mlops-project.git
%cd mlops-project
!pip install uv
!uv sync
!uv run python src/config.py
!uv run dvc pull
!uv run python src/train.py --epochs 50 --batch 16
```

## Технології

- **YOLO v11** (ultralytics) — pose estimation
- **MLflow** → **DagsHub** — experiment tracking
- **DVC** → **DagsHub Storage** — версіонування датасету
- **Python 3.12**, uv