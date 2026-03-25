# Гіперпараметрична оптимізація YOLO Pose — теорія та схема проєкту

## 1. Що таке гіперпараметри і чому їх потрібно підбирати?

**Параметри моделі** — ваги нейронної мережі, що змінюються в процесі навчання (автоматично).

**Гіперпараметри** — налаштування, які ми задаємо *до* навчання і які суттєво впливають на результат:

| Гіперпараметр | Що означає |
|---|---|
| `lr0` | Швидкість навчання на початку (learning rate) |
| `lrf` | Коефіцієнт фінальної learning rate відносно початкової |
| `batch` | Кількість зображень за одну ітерацію |
| `optimizer` | Алгоритм оптимізації (SGD, Adam, AdamW) |
| `mosaic` | Інтенсивність аугментації mosaic (0–1) |
| `degrees` | Кут випадкового повороту при аугментації |
| `imgsz` | Розмір вхідного зображення |

Ручний підбір цих параметрів неефективний — треба автоматична оптимізація.

---

## 2. Методи пошуку гіперпараметрів

```
Grid Search:   ██ ██ ██ ██   — перебір усіх комбінацій (повільно)
Random Search: ·  ·  · ·  ·  — випадковий відбір (швидше, але без гарантій)
Bayesian/TPE:  ▲▲  ▲  ▲▲▲   — розумний пошук, навчається на попередніх trial-ах
```

### Tree-structured Parzen Estimator (TPE) — алгоритм Optuna за замовчуванням

Optuna ділить вже перевірені набори параметрів на "гарні" (top-N%) та "погані", будує для кожної групи розподіл, і наступний trial обирає з регіону де `p(гарний) >> p(поганий)`. Тобто він **навчається** де шукати ефективніше.

---

## 3. Інструменти в цьому проєкті

| Інструмент | Роль |
|---|---|
| **Optuna** | Проводить HPO: управляє trials, обирає параметри |
| **YOLO (ultralytics)** | Тренується з заданими гіперпараметрами в кожному trial |
| **Hydra** | Зберігає конфіги у YAML, дозволяє перевизначати з CLI |
| **MLflow → DagsHub** | Логує всі trial-и (параметри, метрики, артефакти) |
| **DVC** | Версіонує датасет, може запустити весь пайплайн |

---

## 4. Схема роботи HPO пайплайну

```
params.yaml / config.yaml
        │
        ▼
  src/optimize.py (Hydra entry point)
        │
        ├─── MLflow parent run "hpo_study_tpe_*"
        │          │
        │    ┌─────┴──────────────────────────────┐
        │    │  Optuna Study (TPE або Random)      │
        │    │                                    │
        │    │  trial_000 ─── YOLO.train(1-5ep) ──┼── MLflow child run
        │    │  trial_001 ─── YOLO.train(1-5ep) ──┼── MLflow child run
        │    │  ...                               │
        │    │  trial_019 ─── YOLO.train(1-5ep) ──┼── MLflow child run
        │    └────────────────────────────────────┘
        │          │
        │    best_params (best trial)
        │          │
        │    YOLO.train(повні epochs із params.yaml)
        │          │
        └─── models/best_model_hpo.pt  ──── artifact у parent run
```

**Ключова ідея budget optimization**: замість повного тренування (~50-100 епох) кожна спроба використовує лише 1–10 "пробних" епох. Це дозволяє перевірити 20+ конфігурацій за прийнятний час, а повне тренування відбувається лише для переможця.

---

## 5. Структура файлів проєкту

```
mlops_project/
├── config/
│   ├── config.yaml          # Основна конфігурація (повний пошук)
│   └── config_test.yaml     # Мінімальна конфігурація (швидкий тест)
│
├── src/
│   ├── config.py            # PROJECT_ROOT, DagsHub credentials
│   ├── prepare.py           # Розбивка даних train/val
│   ├── train.py             # Одиночне тренування YOLO + MLflow
│   └── optimize.py          # HPO: Optuna + Hydra + MLflow nested runs
│
├── models/
│   ├── pretrained/          # Базові ваги (yolo11n-pose.pt, тощо)
│   ├── best.pt              # Найкраща модель з train.py
│   └── best_model_hpo.pt    # Найкраща модель після HPO
│
├── data/
│   ├── raw/pose/            # Сирі дані (DVC)
│   └── split/               # Train/val split (dvc repro prepare)
│
├── runs/                    # Артефакти YOLO (ваги, графіки) — не в git
├── mlruns/                  # MLflow local fallback — не в git
├── params.yaml              # Параметри пайплайну (DVC)
├── dvc.yaml                 # DVC pipeline stages
└── pyproject.toml           # Python dependencies (uv)
```

---

## 6. Конфіги: мінімальний тест vs. повний пошук

### 🧪 Мінімальний конфіг для локального тесту

Файл: `config/config_test.yaml`  
**Ціль**: перевірити що увесь пайплайн запускається без помилок (3 trials × 1 epoca ≈ 5 хвилин).

```yaml
seed: 42

mlflow:
  experiment_name: "HPO_YOLO_Pose"

data:
  dataset_yaml: "data/split/dataset.yaml"

model:
  weights: "models/pretrained/yolo11n-pose.pt"
  imgsz: 320
  patience: 1

hpo:
  n_trials: 3
  sampler: "tpe"
  direction: "maximize"
  metric: "metrics/mAP50-95(P)"
  trial_epochs: 1       # 1 епоха — тільки smoke test
  log_trial_weights: false  # не завантажувати ваги при тесті

  # Мінімальний пошуковий простір
  lr0:
    low: 0.005
    high: 0.05
  batch:
    choices: [8, 16]

hydra:
  run:
    dir: .
  output_subdir: null
```

**Запуск:**
```bash
python src/optimize.py --config-name config_test
```

---

### 🚀 Повний конфіг для справжнього пошуку (Colab)

Файл: `config/config.yaml`  
**Ціль**: реальний підбір гіперпараметрів. Рекомендовано запускати на Colab/GPU.

```yaml
seed: 42

mlflow:
  experiment_name: "HPO_YOLO_Pose"

data:
  dataset_yaml: "data/split/dataset.yaml"

model:
  weights: "models/pretrained/yolo11n-pose.pt"
  imgsz: 640          # базовий розмір (override через imgsz search space)
  patience: 5

hpo:
  n_trials: 20
  sampler: "tpe"      # tpe | random
  direction: "maximize"
  metric: "metrics/mAP50-95(P)"
  trial_epochs: 5     # короткі тренування для порівняння
  log_trial_weights: true   # зберігати ваги кожного trial на DagsHub

  # Повний пошуковий простір
  lr0:
    low: 0.001
    high: 0.1
  lrf:
    low: 0.001
    high: 0.1
  batch:
    choices: [8, 16, 32]
  optimizer:
    choices: ["SGD", "Adam", "AdamW"]
  mosaic:
    low: 0.0
    high: 1.0
  degrees:
    low: 0.0
    high: 30.0
  imgsz:
    choices: [320, 640, 1280, 1920]  # від базового до максимального

hydra:
  run:
    dir: .
  output_subdir: null
```

> **Примітка щодо `imgsz`**: більший розмір зображення (1280, 1920) суттєво збільшує час тренування і вимоги до GPU пам'яті. На Colab з T4 рекомендується починати з 320–640. На A100 можна пробувати 1280+.

---

## 7. Повний список команд запуску

```bash
# --- Мінімальний тест (локально, ~5 хвилин) ---
python src/optimize.py --config-name config_test

# --- Стандартний HPO (TPE) ---
python src/optimize.py

# --- HPO з Random sampler для порівняння ---
python src/optimize.py hpo.sampler=random hpo.n_trials=20

# --- Перевизначити будь-який параметр з CLI ---
python src/optimize.py hpo.n_trials=30 hpo.trial_epochs=10

# --- Через DVC (якщо змінили params.yaml → hpo секцію) ---
dvc repro optimize
```

---

## 8. Що бачити в MLflow (DagsHub)

Після запуску на **DagsHub → MLflow**:

1. Відкрити `https://dagshub.com/<user>/<repo>/mlflow`
2. Обрати experiment **HPO_YOLO_Pose**
3. Знайти запуск `hpo_study_tpe_*` (parent run)
4. Розгорнути → побачити child runs `trial_000`, `trial_001`, ...
5. Кожен trial: параметри + метрика `trial_score` (mAP50-95)
6. Parent run містить: `best_params.json`, `config_resolved.json`, `model/best_model_hpo.pt`

**Порівняння TPE vs Random**: запустити обидва sampler-и з однаковим `n_trials` і порівняти `best_score` у parent runs.
