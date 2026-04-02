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

---

## 9. Відповіді на контрольні питання

### 1. Чим відрізняються гіперпараметри від параметрів моделі?

**Параметри моделі** — ваги нейронної мережі (наприклад, мільйони чисел у шарах YOLO11), які **автоматично оновлюються** під час backpropagation. Їх не задають вручну.

**Гіперпараметри** — налаштування, які ми задаємо **до** навчання і які визначають умови процесу навчання:

| Тип | Приклади в нашому проєкті |
|-----|--------------------------|
| Параметри моделі | Ваги Conv шарів, BN статистики, anchor-free голова YOLO11 |
| Гіперпараметри | `lr0`, `lrf`, `batch`, `optimizer`, `mosaic`, `degrees`, `imgsz` |

У `src/optimize.py` Optuna пропонує значення гіперпараметрів (`trial.suggest_float(...)`), YOLO тренується з ними і повертає mAP50-95. Ваги YOLO при цьому — результат тренування, не вхід для Optuna.

---

### 2. Три методи пошуку гіперпараметрів

| Метод | Принцип | В нашому проєкті |
|-------|---------|-----------------|
| **Grid Search** | Перебирає всі комбінації із заданої сітки значень | Не використовується (замалий бюджет trials) |
| **Random Search** | Випадково семплює точки з простору пошуку | `hpo.sampler=random` → `RandomSampler` |
| **TPE (Bayesian)** | Будує дві моделі розподілу: "гарні" та "погані" trial-и, вибирає з регіону де p(гарний) >> p(поганий) | За замовчуванням `hpo.sampler=tpe` |

**Чому TPE краще для YOLO**: кожен trial коштує 1-5 хвилин тренування. TPE за 20 trials знаходить кращі регіони ніж Random, бо навчається на попередніх результатах.

---

### 3. Що таке Trial в Optuna?

**Trial** — одна "спроба" з конкретним набором гіперпараметрів, запропонованих sampler-ом.

В `src/optimize.py`:
```python
def objective(trial: optuna.Trial) -> float:
    params["lr0"] = trial.suggest_float("lr0", 0.001, 0.1, log=True)
    params["batch"] = trial.suggest_categorical("batch", [8, 16, 32])
    # ... YOLO тренується ...
    return score  # mAP50-95
```

Кожен trial:
1. Отримує від sampler набір параметрів
2. Тренує YOLO на 1-5 епохах
3. Повертає mAP50-95 як score
4. Логується як **child run** у MLflow

---

### 4. Parent run vs Child run у MLflow

```
MLflow Experiment: "HPO_YOLO_Pose"
│
└── Parent Run: "hpo_study_tpe_1711234567"   ← весь HPO-процес
    ├── Tags: sampler=tpe, seed=42, trial_epochs=5
    ├── Metrics: best_score=0.4821
    ├── Artifacts: best_params.json, config_resolved.json, best_model_hpo.pt
    │
    ├── Child Run: "trial_000"   ← один trial
    │   ├── Params: lr0=0.01234, batch=16, optimizer=Adam
    │   └── Metric: trial_score=0.3201
    │
    ├── Child Run: "trial_001"
    │   └── trial_score=0.4512
    └── ...
```

**Чому nested runs зручніші:**
- Бачиш всі 20 trials в одному місці, а не розмиті по списку всіх запусків
- Легко фільтрувати: "показати всі trials де trial_score > 0.4"
- Parent run — підсумок, Child runs — деталі

---

### 5. Яку проблему вирішує Hydra?

**Без Hydra**: конфігурація захардкоджена в коді або передається через аргументи CLI вручну.

**З Hydra**: конфігурація в YAML-файлах, перевизначається через CLI без зміни коду:

```bash
# Змінити кількість trials без редагування файлу
python src/optimize.py hpo.n_trials=50

# Переключитися на random sampler
python src/optimize.py hpo.sampler=random

# Запустити мінімальний тест
python src/optimize.py --config-name config_test
```

В нашому проєкті `config/config.yaml` і `config/config_test.yaml` описують різні режими роботи. Hydra автоматично підставляє значення через `DictConfig`.

---

### 6. n_trials=20: як зменшити час / підвищити якість?

**Зменшити час виконання:**

| Спосіб | Як реалізовано |
|--------|---------------|
| Менше епох per trial | `hpo.trial_epochs=1` (замість 5) |
| Менший imgsz | `model.imgsz=320` |
| Pruning (рання зупинка слабких trials) | Додати Optuna `MedianPruner` |
| Паралельний запуск | Кілька процесів з shared Optuna DB |

**Підвищити якість пошуку:**

| Спосіб | Як |
|--------|---|
| Більше trials | `hpo.n_trials=50` |
| TPE замість Random | `hpo.sampler=tpe` (вже є за замовчуванням) |
| Звузити search space | Прибрати неважливі параметри (наприклад, `degrees`) |
| Warm Start | Передати попередні результати в новий study через `add_trial` |

---

### 7. Чому nested runs зручніші за один run?

**В одному run**: всі параметри 20 trials перемішані → неможливо порівняти trial_005 з trial_012.

**З nested runs:**
```
Фільтр: "покажи лише trials де batch=16"
→ DagsHub MLflow вибирає тільки child runs з params.batch=16
```

Також: якщо один trial впав з помилкою — він позначений у своєму child run, не виглядає як "провалений весь HPO".

---

### 8. Навіщо фіксувати seed? Що саме "сідувати"?

**Без seed**: якщо запустити HPO двічі — отримаємо різні послідовності trials, неможливо відтворити результат.

**Що треба фіксувати:**

| Що | Як в нашому проєкті |
|----|---------------------|
| Optuna sampler | `TPESampler(seed=cfg.seed)` |
| YOLO тренування | Параметр `seed` у `model.train(seed=42)` |
| Логування | `mlflow.set_tag("seed", cfg.seed)` |

В `config.yaml`: `seed: 42` — єдине місце, звідки всі seed читаються.

---

### 9. Як коректно порівнювати TPE і Random sampler?

**Умови коректного порівняння:**

1. **Однаковий seed**: `seed=42` для обох → відтворюваність
2. **Однаковий n_trials**: наприклад, 20 для кожного
3. **Однаковий датасет** (DVC-версія зафіксована в `dvc.lock`)
4. **Однакові trial_epochs**, imgsz та інші умови

**Запуск в нашому проєкті:**
```bash
python src/optimize.py hpo.sampler=tpe   hpo.n_trials=20  # TPE
python src/optimize.py hpo.sampler=random hpo.n_trials=20  # Random
```

Порівнювати: `best_score` у parent runs + графік "best-so-far" (як швидко виходить на хороший результат).

---

### 10. Кроки перед переведенням моделі зі Staging у Production

1. **Перевірити метрики**: `mAP50 >= production_threshold` (наприклад, 0.35)
2. **Валідація на held-out test set** (не val, який використовувався в HPO)
3. **Regression test**: порівняти з поточною Production моделлю — чи не гірше?
4. **Перевірити відтворюваність**: чи є в MLflow всі артефакти (params.json, dvc.lock, git commit hash)?
5. **Review**: колега перевіряє звіт (CML report з PR)
6. **MLflow Registry**: `transition_model_version_stage(stage="Production")`

В нашому проєкті CD-крок в `ci.yaml` завантажує `best.pt` як GitHub Artifact при merge в main — це аналог "публікації" моделі.

---

### 11. Objective з 5-fold CV — вплив на стабільність і час

**Звичайна оцінка (train/val split):**
```python
score = mAP50(val_set)   # 1 оцінка → висока варіація
```

**5-fold CV:**
```python
scores = [mAP50(fold_i) for i in range(5)]
score = mean(scores)     # стабільніша оцінка
```

**Вплив:**

| Аспект | Без CV | З 5-fold CV |
|--------|--------|-------------|
| Стабільність | Залежить від випадкового split | Стабільніша, менше шуму |
| Час | 1x | 5x (5 тренувань per trial) |
| Ризик overfitting до val | Вищий | Нижчий |

Для YOLO Pose CV практично неможлива при великих датасетах (час надто великий). Натомість використовуємо фіксований val split + кілька seeds.

---

### 12. Рання зупинка (Pruning) в Optuna для нейромережі

**Без pruning**: trial тренується всі 5 epoh навіть якщо вже після 1-ї епохи видно що він поганий.

**З MedianPruner**: якщо після époch_k метрика trial гірша за медіану всіх попередніх trials на тому ж кроці → trial зупиняється.

```python
pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
study = optuna.create_study(pruner=pruner)

# В objective — повідомляти Optuna про проміжні метрики:
for epoch in range(n_epochs):
    score = train_one_epoch(...)
    trial.report(score, step=epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
```

**Для YOLO**: потрібно інтегруватися з Ultralytics callback API, що ускладнює реалізацію. В нашому проєкті pruning не реалізовано.

---

### 13. HPO для кількох моделей з динамічним вибором через Hydra

**Підхід: `model.type` як categorical параметр:**

```python
model_type = trial.suggest_categorical("model_type", ["yolo11n", "yolo11s", "yolo11m"])

if model_type == "yolo11n":
    params["imgsz"] = trial.suggest_categorical("imgsz", [320, 640])
elif model_type == "yolo11s":
    params["imgsz"] = trial.suggest_categorical("imgsz", [640, 1280])
```

**Через Hydra конфіги:**
```yaml
# config/model/yolo11n.yaml
weights: "models/pretrained/yolo11n-pose.pt"
max_imgsz: 640

# config/model/yolo11s.yaml
weights: "models/pretrained/yolo11s-pose.pt"
max_imgsz: 1280
```

```bash
python src/optimize.py model=yolo11s  # перемикання через CLI
```

---

### 14. Як інтегрувати HPO в CI/CD

**Де запускати**: НЕ в кожному PR (занадто повільно). Розумна схема:

| Тригер | Що запускати |
|--------|-------------|
| Push/PR | Smoke HPO (3 trials × 1 epoch) на CPU |
| Schedule (раз/тиждень) | Повний HPO (20+ trials) на GPU |
| Merge в main | Retrain з best_params на повному датасеті |

**Бюджет для CI**: `n_trials=3`, `trial_epochs=1` (як у нашому `config_test.yaml`).

**Артефакти після HPO**: `best_params.json` і `best_model_hpo.pt` зберігаються в MLflow та як GitHub Action artifacts.

---

### 15. Стратегія тегування версій моделей у MLflow Registry

**Рекомендована схема тегів:**

```python
tags = {
    "sampler": "tpe",
    "n_trials": 20,
    "trial_epochs": 5,
    "best_trial": 12,
    "git_commit": os.getenv("GITHUB_SHA", "local"),
    "dvc_data_hash": open("dvc.lock").read()[:16],
    "dataset": "bee_pose_v1",
    "model_arch": "yolo11n-pose",
}
mlflow.set_model_version_tag(name, version, key, val)
```

**Стадії Registry:**
- `Staging` → модель пройшла HPO і smoke Quality Gate
- `Production` → пройшла повне тестування і review
- `Archived` → замінена новою версією

**Практика**: назва версії = `"{sampler}_n{n_trials}_ep{trial_epochs}_{git_commit[:7]}"` → за назвою одразу зрозуміло як створена модель.
