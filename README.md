# Bee Pose Estimation — MLOps Project

Визначення пози бджіл на прилітній дошці вулика за допомогою YOLO Pose з автоматизованим відстеженням експериментів через MLflow та W&B.

## Задача

Fine-tuning YOLOv11n-pose на датасеті бджіл:
- **400 зображень** з 8 вуликів (50 кадрів кожний)
- **2 ключові точки** на бджолу: голова та жало
- Формат анотацій: YOLO Pose (`class cx cy w h kp1x kp1y kp2x kp2y`)

## Структура проєкту

```
├── src/
│   ├── train.py           # Тренування з CLI аргументами
│   ├── data.py            # Train/val split по вуликах
│   ├── tracking.py        # MLflow + W&B логування
│   └── setup_colab.py     # Підготовка середовища Colab
├── notebooks/
│   └── 01_eda.ipynb       # Розвідувальний аналіз даних
├── data/raw/pose/         # Сирі дані (не в git)
├── models/pretrained/     # Pretrained ваги (не в git)
├── runs/                  # Результати YOLO (не в git)
└── mlruns/                # MLflow tracking store (не в git)
```

## Встановлення

```bash
# Клонування та створення середовища
git clone <repo-url>
cd mlops_project
uv sync  # або: pip install ultralytics mlflow wandb pyyaml
```

## Запуск тренування

```bash
# Локально (тест без GPU)
python src/train.py --no-wandb --epochs 2 --batch 2 --imgsz 320

# Повне тренування
python src/train.py --epochs 50 --batch 16 --lr0 0.01

# З іншими гіперпараметрами
python src/train.py --epochs 50 --lr0 0.005 --optimizer SGD --run-name "sgd-0.005"
```

### CLI аргументи

| Аргумент | Default | Опис |
|----------|---------|------|
| `--epochs` | 50 | Кількість епох |
| `--batch` | 16 | Розмір батчу |
| `--imgsz` | 640 | Розмір зображення |
| `--lr0` | 0.01 | Initial learning rate |
| `--optimizer` | AdamW | SGD / Adam / AdamW |
| `--model` | models/pretrained/yolo11n-pose.pt | Pretrained ваги |
| `--no-wandb` | — | Вимкнути W&B |
| `--force-split` | — | Перестворити train/val split |

## Запуск у Google Colab

```python
!git clone <repo-url>
%cd mlops_project
!python src/setup_colab.py           # монтує Drive, розпаковує датасет
!wandb login                          # опціонально
!python src/train.py --epochs 50 --batch 16 --lr0 0.01
```

Датасет завантажується автоматично з Google Drive.

## MLflow UI

```bash
mlflow ui --backend-store-uri mlruns/
# Відкрити: http://127.0.0.1:5000
```

## Технології

- **YOLO v11** (ultralytics) — pose estimation
- **MLflow** — experiment tracking (параметри, метрики, артефакти)
- **W&B** — хмарний трекінг експериментів
- **Python 3.12**, uv