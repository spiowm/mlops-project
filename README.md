# Bee Pose Estimation — MLOps Project

Визначення пози бджіл на прилітній дошці вулика за допомогою **YOLO11 Pose** із відстеженням експериментів через **MLflow → DagsHub** та версіонуванням даних через **DVC**.

## Задача

Fine-tuning YOLOv11n-pose на датасеті бджіл:
- **~400 зображень** з 8 вуликів (50 кадрів на кожний)
- **2 ключові точки** на бджолу: голова та жало
- Формат анотацій: YOLO Pose (`class cx cy w h kp1x kp1y kp2x kp2y`)

---

## Структура проєкту

```
mlops_project/
├── config/
│   ├── config.yaml          # Hydra конфіг HPO (повний пошук)
│   └── config_test.yaml     # Hydra конфіг HPO (мінімальний тест)
│
├── src/
│   ├── config.py            # PROJECT_ROOT, DagsHub credentials
│   ├── prepare.py           # Розбивка датасету train/val по вуликах
│   ├── train.py             # Одиночне тренування YOLO + MLflow
│   └── optimize.py          # HPO: Optuna + Hydra + MLflow nested runs
│
├── models/
│   ├── pretrained/          # Базові ваги YOLO (не в git — великі файли)
│   │   └── yolo11n-pose.pt
│   ├── best.pt              # Найкраща модель з train.py
│   └── best_model_hpo.pt    # Найкраща модель після HPO
│
├── data/
│   ├── raw/pose/            # Сирий датасет (керується DVC)
│   └── split/               # Train/val split (генерується prepare.py)
│
├── docs/
│   └── lab3_hpo_theory.md   # Теорія HPO та схема (укр.)
│
├── notebooks/
│   └── 01_eda.ipynb         # Розвідувальний аналіз даних
│
├── runs/                    # Результати YOLO runs (не в git)
├── mlruns/                  # MLflow local fallback (не в git)
├── params.yaml              # Параметри пайплайну (DVC-tracked)
├── dvc.yaml                 # DVC стадії пайплайну
└── pyproject.toml           # Python залежності (uv)
```

---

## Встановлення

```bash
git clone https://github.com/spiowm/mlops-project.git
cd mlops-project
uv sync
```

### Credentials (`.env`)

Скопіювати `.env.example` → `.env` та заповнити:
```
DAGSHUB_TOKEN=your_dagshub_token
DAGSHUB_USER=your_dagshub_username
DAGSHUB_REPO=your_dagshub_repo_name
```

Без credentials: MLflow логує локально (`mlruns/`), DVC не підключений до remote.

### Датасет (DVC)

```bash
python src/config.py   # налаштовує DVC credentials з .env
dvc pull               # завантажує датасет (~360 MB)
```

---

## Способи запуску

### 1. Підготовка даних (обов'язково першим кроком)

```bash
python src/prepare.py
# або через DVC:
dvc repro prepare
```

Розбиває `data/raw/pose/` на `data/split/train/` та `data/split/val/` за ID вуликів.

---

### 2. Одиночне тренування

```bash
python src/train.py
```

| Аргумент | Default | Опис |
|----------|---------|------|
| `--epochs` | 1 | Кількість епох |
| `--batch` | 16 | Розмір батчу |
| `--imgsz` | 320 | Розмір зображення |
| `--lr0` | 0.01 | Initial learning rate |
| `--patience` | 20 | Early stopping patience |
| `--optimizer` | auto | `SGD` / `Adam` / `AdamW` / `auto` |
| `--model` | `models/pretrained/yolo11n-pose.pt` | Шлях до базової моделі |

```bash
# Приклади
python src/train.py --epochs 50 --batch 16 --imgsz 640
python src/train.py --epochs 100 --lr0 0.005 --optimizer SGD
```

---

### 3. Гіперпараметрична оптимізація (HPO)

Детальна теорія → [`docs/lab3_hpo_theory.md`](docs/lab3_hpo_theory.md)

#### 🧪 Мінімальний тест (локально, ~5 хвилин)

Перевірка що pipeline запускається без помилок:
```bash
python src/optimize.py --config-name config_test
```
→ 3 trials × 1 епоха, підбираються `lr0` та `batch`.

#### 🚀 Повний HPO (на GPU / Colab)

```bash
# TPE sampler (Bayesian, рекомендований)
python src/optimize.py

# Random sampler (для порівняння з TPE)
python src/optimize.py hpo.sampler=random

# Перевизначення з CLI
python src/optimize.py hpo.n_trials=30 hpo.trial_epochs=10

# Через DVC pipeline
dvc repro optimize
```

#### Що відбувається під час HPO:

1. Запускається **20 trials** (коротке тренування ~5 епох кожен)
2. Optuna TPE обирає наступні параметри на основі попередніх результатів
3. Кожен trial → **child MLflow run** (параметри + mAP50-95)
4. Найкраща конфігурація → **повне тренування** (epochs з `params.yaml`)
5. Результат: `models/best_model_hpo.pt` + артефакти на **DagsHub MLflow**

---

### 4. Повний DVC пайплайн

```bash
dvc repro          # prepare → train → optimize
dvc repro prepare  # тільки підготовка даних
dvc repro train    # тільки тренування
dvc repro optimize # тільки HPO
```

---

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

```bash
# Клонування та встановлення
!git clone https://github.com/spiowm/mlops-project.git
%cd mlops-project
!pip install uv -q
!uv sync

# Credentials (або через Colab Secrets → os.environ)
# Створити .env вручну або через secrets

# Дані
!uv run python src/config.py
!uv run dvc pull

# Тренування
!uv run python src/train.py --epochs 50 --batch 16 --imgsz 640

# HPO (повний пошук)
!uv run python src/optimize.py hpo.n_trials=20 hpo.trial_epochs=5
```

---

## Відстеження експериментів (MLflow → DagsHub)

Результати всіх runs зберігаються на **DagsHub**:
```
https://dagshub.com/<DAGSHUB_USER>/<DAGSHUB_REPO>/mlflow
```

Експерименти:
- **`bee-pose-estimation`** — звичайні тренування (`train.py`)
- **`HPO_YOLO_Pose`** — HPO runs з nested trials (`optimize.py`)

---

## Технології

| | Інструмент | Роль |
|--|---|---|
| 🤖 | YOLO v11 (ultralytics) | Pose estimation модель |
| 📊 | MLflow + DagsHub | Experiment tracking, артефакти |
| 🔢 | Optuna | HPO: підбір гіперпараметрів |
| ⚙️ | Hydra | Управління конфігами |
| 📦 | DVC + DagsHub | Версіонування датасету |
| 🐍 | Python 3.12, uv | Середовище та залежності |