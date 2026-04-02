# Bee Pose Estimation — MLOps Project

Fine-tuning **YOLO11 Pose** на датасеті бджіл (голова + жало) з повним MLOps-стеком:
відстеження експериментів → HPO → CI/CD → Continuous Training оркестрація.

---

## Завдання

Визначення пози бджіл на прилітній дошці вулика:

- **~400 зображень** з 8 вуликів (50 кадрів на кожний)
- **2 ключові точки** на бджолу: голова та жало
- Формат: YOLO Pose (`class cx cy w h kp1x kp1y kp2x kp2y`)

---

## Структура проєкту

```
mlops_project/
│
├── .github/workflows/
│   └── ci.yaml                    # GitHub Actions: lint + DAG tests + Docker + train
│
├── config/
│   ├── config.yaml                # Hydra конфіг HPO (повний пошук, 20 trials)
│   └── config_test.yaml           # Hydra конфіг HPO (мінімальний тест, 3 trials)
│
├── src/
│   ├── config.py                  # PROJECT_ROOT, DagsHub credentials (.env)
│   ├── prepare.py                 # Розбивка датасету train/val по ID вуликів
│   ├── train.py                   # Одиночне тренування YOLO + MLflow loggin
│   └── optimize.py                # HPO: Optuna + Hydra + MLflow nested runs
│
├── scripts/
│   ├── ci_train.py                # CI тренування (1 epoch, CPU, smoke test)
│   ├── compare_metrics.py         # Порівняння з baseline (Δ метрик)
│   └── generate_report.py         # CML Markdown звіт для PR
│
├── dags/                          # ← ЛР5: Оркестрація
│   ├── __init__.py
│   └── yolo_training_pipeline.py  # Prefect Flow: CT пайплайн
│
├── tests/
│   ├── conftest.py                # Shared fixtures (шляхи, пороги)
│   ├── test_data.py               # Pre-train: структура даних, мітки YOLO Pose
│   ├── test_config.py             # Pre-train: pyproject.toml, params.yaml
│   ├── test_artifacts.py          # Post-train: Quality Gate (mAP50 ≥ поріг)
│   └── test_dag.py                # ← ЛР5: 12 тестів цілісності Prefect Flow
│
├── models/
│   ├── pretrained/                # Базові ваги YOLO (не в git, DVC або вручну)
│   ├── best.pt                    # Найкраща модель з train.py
│   └── best_model_hpo.pt          # Найкраща модель після HPO
│
├── data/
│   ├── raw/pose/                  # Сирий датасет (версіонується DVC)
│   └── split/                     # Train/val split (генерується prepare.py)
│
├── baseline/
│   └── metrics.json               # Еталонні метрики для порівняння в PR і CT
│
├── tasks/docs/                    # Теорія та документація до лабораторних
│   ├── lab3_hpo_theory.md         # Теорія + відповіді на питання ЛР3 (HPO)
│   ├── lab4_cicd_theory.md        # Теорія + відповіді на питання ЛР4 (CI/CD)
│   └── lab5_orchestration_theory.md  # Теорія + відповіді на питання ЛР5 (CT)
│
├── notebooks/
│   └── 01_eda.ipynb               # Розвідувальний аналіз даних (EDA)
│
├── Dockerfile                     # ← ЛР5: Multi-stage build
├── runs/                          # Результати YOLO runs (не в git)
├── mlruns/                        # MLflow local fallback (не в git)
├── params.yaml                    # Параметри пайплайну (DVC-tracked)
├── dvc.yaml                       # DVC стадії: prepare → train → optimize
└── pyproject.toml                 # Python залежності (uv) + pytest/ruff конфіг
```

---

## Встановлення

```bash
git clone https://github.com/spiowm/mlops-project.git
cd mlops-project
uv sync
```

### Credentials (`.env`)

```bash
cp .env.example .env
```

Заповнити `.env`:
```env
DAGSHUB_TOKEN=your_dagshub_token
DAGSHUB_USER=your_dagshub_username
DAGSHUB_REPO=your_dagshub_repo_name
```

Без credentials — MLflow логує локально (`mlruns/`), DVC не підключений до remote.

### Датасет (DVC)

```bash
uv run dvc pull        # завантажує датасет (~360 MB) з DagsHub
```

---

## Способи запуску

### 1. Підготовка даних (обов'язковий перший крок)

```bash
uv run python src/prepare.py
# або через DVC:
uv run dvc repro prepare
```

Розбиває `data/raw/pose/` на `data/split/train/` та `data/split/val/` по ID вуликів.

---

### 2. Одиночне тренування

```bash
uv run python src/train.py
```

| Аргумент | Default | Опис |
|----------|---------|------|
| `--epochs` | 1 | Кількість епох |
| `--batch` | 16 | Розмір батчу |
| `--imgsz` | 320 | Розмір зображення |
| `--lr0` | 0.01 | Initial learning rate |
| `--patience` | 20 | Early stopping |
| `--optimizer` | auto | `SGD` / `Adam` / `AdamW` / `auto` |
| `--model` | `models/pretrained/yolo11n-pose.pt` | Шлях до базової моделі |

```bash
# Приклади
uv run python src/train.py --epochs 50 --imgsz 640
uv run python src/train.py --epochs 100 --lr0 0.005 --optimizer SGD
```

Результати → MLflow (`bee-pose-estimation` experiment) + `runs/train/`

---

### 3. Гіперпараметрична оптимізація (HPO)

> Детальна теорія + відповіді на питання → [`tasks/docs/lab3_hpo_theory.md`](tasks/docs/lab3_hpo_theory.md)

#### Мінімальний тест (~5 хвилин, CPU)

```bash
uv run python src/optimize.py --config-name config_test
```

→ 3 trials × 1 epoch, підбирає `lr0` і `batch`.

#### Повний HPO (GPU рекомендовано)

```bash
# TPE sampler (Bayesian, рекомендований)
uv run python src/optimize.py

# Random sampler (для порівняння)
uv run python src/optimize.py hpo.sampler=random

# Перевизначити параметри з CLI (Hydra)
uv run python src/optimize.py hpo.n_trials=30 hpo.trial_epochs=10
```

**Що відбувається:**

1. Запускається **20 trials** (5 epochs кожен)
2. Optuna TPE підбирає наступні параметри на основі результатів
3. Кожен trial → **child MLflow run** (params + mAP50-95)
4. Найкраща конфігурація → **повне тренування** (epochs з `params.yaml`)
5. Результат: `models/best_model_hpo.pt` + артефакти в **DagsHub MLflow**

---

### 4. Continuous Training Pipeline (ЛР5)

> Детальна теорія + інструкції → [`tasks/docs/lab5_orchestration_theory.md`](tasks/docs/lab5_orchestration_theory.md)

CT Pipeline — оркестрований пайплайн (Prefect Flow), що запускається за розкладом або вручну і автоматично: перевіряє дані → готує → тестує → навчає → перевіряє якість → реєструє або сповіщає.

#### Запуск (без UI)

```bash
# Smoke запуск (threshold=0.0, завжди реєструє)
uv run python dags/yolo_training_pipeline.py

# З параметрами
CT_EPOCHS=1 CT_IMGSZ=320 CT_MAP50_THRESHOLD=0.0 \
  uv run python dags/yolo_training_pipeline.py
```

#### Запуск з Prefect UI

```bash
# Термінал 1
uv run prefect server start  # UI на http://localhost:4200

# Термінал 2
uv run python dags/yolo_training_pipeline.py
```

#### Параметри CT Pipeline

| Env змінна | Default | Опис |
|-----------|---------|------|
| `CT_EPOCHS` | `1` | Epochs тренування |
| `CT_IMGSZ` | `320` | Розмір зображення |
| `CT_MAP50_THRESHOLD` | `0.0` | Мінімальний mAP50 для реєстрації |

#### Що робить пайплайн крок за кроком

```
1. check_data_freshness   → Перевіряє data/split/, повертає статус
2. prepare_data           → src/prepare.py (якщо даних немає)
3. run_pretrain_tests     → pytest tests/test_data.py tests/test_config.py
4. train_model            → scripts/ci_train.py → metrics.json
5. evaluate_model         → читає metrics.json, повертає mAP50
6. check_quality_gate     → mAP50 ≥ CT_MAP50_THRESHOLD?
   ├─ PASSED → register_model → оновлює baseline/metrics.json
   └─ FAILED → notify_failure → Prefect artifact зі звітом
```

---

### 5. Повний DVC пайплайн

```bash
uv run dvc repro           # prepare → train → optimize
uv run dvc repro prepare   # тільки підготовка
uv run dvc repro train     # тільки тренування
uv run dvc repro optimize  # тільки HPO
```

---

## Тестування

```bash
# Pre-train (валідація даних та конфігурації)
uv run pytest tests/test_data.py tests/test_config.py -v

# Post-train (Quality Gate)
MAP50_THRESHOLD=0.0 uv run pytest tests/test_artifacts.py -v

# DAG/Flow тести (ЛР5)
uv run pytest tests/test_dag.py -v

# Всі тести
uv run pytest -v

# Лінтинг
uv run ruff check src/ scripts/ tests/ dags/
```

---

## CI/CD (GitHub Actions)

> Детальна теорія → [`tasks/docs/lab4_cicd_theory.md`](tasks/docs/lab4_cicd_theory.md)

При кожному `push` та `pull_request` автоматично:

```
┌─────────┐  ┌────────────┐  ┌─────────────┐  ┌────────────────────┐
│  Lint   │  │ DAG Tests  │  │Docker Build │  │   Train & Test     │
│ (ruff)  │  │(test_dag.py│  │(multi-stage)│  │  1 epoch CPU       │
│ dags/ ✓ │  │ 12 tests)  │  │+ smoke run  │  │  Quality Gate      │
└────┬────┘  └─────┬──────┘  └─────────────┘  │  CML Report (PR)   │
     │             │                           └────────────────────┘
     └─────────────┴──────────────────────────────────────┬──────────
                                                          │ needs: lint, dag-tests, pre-train-tests
```

### Jobs у CI

| Job | Що робить | Тригер |
|-----|-----------|--------|
| `lint` | ruff check src/ scripts/ tests/ dags/ | всі |
| `dag-tests` | pytest tests/test_dag.py (12 тестів) | всі |
| `docker-build` | docker build + smoke run | всі |
| `pre-train-tests` | dvc pull → prepare → pytest test_data + test_config | всі |
| `train-and-test` | CI train → Quality Gate → CML report | після lint + dag-tests + pre-train |

### Налаштування GitHub Secrets

`Settings → Secrets and variables → Actions`:

| Secret | Призначення |
|--------|------------|
| `DAGSHUB_USER` | DVC remote + MLflow auth |
| `DAGSHUB_TOKEN` | DVC remote + MLflow auth |

---

## Docker (ЛР5)

```bash
# Збудувати
docker build --target runtime -t yolo-pose:local .

# Запустити CI training
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  yolo-pose:local \
  python scripts/ci_train.py

# Запустити CT pipeline
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/baseline:/app/baseline \
  -e CT_EPOCHS=1 -e CT_MAP50_THRESHOLD=0.0 \
  yolo-pose:local \
  python dags/yolo_training_pipeline.py
```

---

## Відстеження експериментів (MLflow → DagsHub)

```
https://dagshub.com/<DAGSHUB_USER>/<DAGSHUB_REPO>/mlflow
```

| Experiment | Скрипт | Що зберігається |
|-----------|--------|----------------|
| `bee-pose-estimation` | `train.py` | params, metrics, confusion matrix |
| `HPO_YOLO_Pose` | `optimize.py` | nested runs (parent study + child trials) |

---

## Google Colab

```python
!git clone https://github.com/spiowm/mlops-project.git
%cd mlops-project
!pip install uv -q && uv sync

# Credentials через Colab Secrets або вручну
import os
os.environ["DAGSHUB_TOKEN"] = "..."
os.environ["DAGSHUB_USER"] = "..."
os.environ["DAGSHUB_REPO"] = "..."

# Дані
!uv run dvc pull

# Тренування
!uv run python src/train.py --epochs 50 --imgsz 640

# HPO (повний пошук)
!uv run python src/optimize.py hpo.n_trials=20 hpo.trial_epochs=5
```

---

## Технології

| | Інструмент | Роль | ЛР |
|--|---|---|---|
| 🤖 | YOLO11 (ultralytics) | Pose estimation модель | ЛР1 |
| 📦 | DVC + DagsHub | Версіонування датасету | ЛР2 |
| 📊 | MLflow + DagsHub | Experiment tracking | ЛР2 |
| 🔢 | Optuna | HPO: підбір гіперпараметрів | ЛР3 |
| ⚙️ | Hydra | Управління конфігураціями | ЛР3 |
| 🔄 | GitHub Actions + CML | CI/CD, звіти в PR | ЛР4 |
| 🧪 | pytest + ruff | Тестування та лінтинг | ЛР4 |
| 🎯 | Prefect | Continuous Training оркестрація | ЛР5 |
| 🐳 | Docker (multi-stage) | Контейнеризація, CI perевірка | ЛР5 |
| 🐍 | Python 3.12, uv | Середовище та залежності | всі |