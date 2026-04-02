# CI/CD для ML-проєктів — теорія та реалізація для Bee Pose Estimation

## 1. Навіщо CI/CD у Machine Learning?

У звичайній розробці CI/CD перевіряє **код**: компіляція, тести, деплой.  
В ML-проєктах додаються ще три компоненти, які впливають на результат:

```
Звичайний софт:  Код ────────────────→ Артефакт (бінарний файл)
                   ↑
                   CI: lint + tests

ML-проєкт:       Код + Дані + Конфіг → Артефакт (модель + метрики)
                   ↑          ↑
                   CI:        CI:
                   lint       data validation
                   tests      quality gate
                              baseline comparison
```

**Що може зламатися в ML, чого немає в звичайному софті:**

| Ризик | Приклад |
|-------|---------|
| Зміна даних | Нові зображення бджіл з іншим фоном → модель деградує |
| Зміна конфігу | Хтось змінив `imgsz: 640 → 160` і забув повернути |
| Тиха деградація | Код працює, тести проходять, але mAP впав на 15% |
| Невідтворюваність | «У мене на Colab працювало» — інший seed, дані, версія пакету |

---

## 2. GitHub Actions: як працює

**Workflow** — YAML-файл у `.github/workflows/`, описує що і коли запускати.

```
GitHub Events                    Runner (VM)
──────────────                   ──────────
push до main   ──────┐          ┌─────────────────┐
                     ├────────→ │ Job: lint        │
pull_request   ──────┤          │  Step 1: checkout│
                     │          │  Step 2: install │
workflow_dispatch ───┘          │  Step 3: ruff    │
                                └─────────────────┘
                                        │
                                        ▼
                                ┌─────────────────┐
                                │ Job: test + train│
                                │  Step 1: dvc pull│
                                │  Step 2: train   │
                                │  Step 3: pytest  │
                                │  Step 4: CML     │
                                └─────────────────┘
```

### Ключові терміни

| Термін | Пояснення |
|--------|-----------|
| **Workflow** | Повний YAML-файл з описом пайплайну |
| **Event** | Тригер: `push`, `pull_request`, `schedule`, `workflow_dispatch` |
| **Job** | Ізольований набір кроків на одному runner |
| **Step** | Окремий крок (bash-команда або action) |
| **Runner** | Віртуальна машина від GitHub (ubuntu-latest) |
| **Secret** | Захищена змінна (токени, паролі) — не у коді |

---

## 3. Наш CI/CD пайплайн

### Структура workflow в цьому проєкті

```
.github/workflows/ci.yaml
│
├── Job: lint                     (паралельно)
│   └── ruff check src/ scripts/ tests/
│
├── Job: pre-train-tests          (паралельно)
│   ├── DVC pull (завантажити дані)
│   ├── Prepare data split
│   └── pytest tests/test_data.py tests/test_config.py
│
└── Job: train-and-test           (після lint + pre-train)
    ├── DVC pull + prepare
    ├── CI Train (1 epoch, CPU, imgsz=160)
    ├── pytest tests/test_artifacts.py (Quality Gate)
    ├── CML Report (тільки на PR)
    └── Upload artifacts (тільки на main)
```

### Чому три окремі job-и?

- **Швидкий feedback**: lint та pre-train тести йдуть паралельно (~1 хв)
- **Ізоляція**: якщо дані некоректні — тренування навіть не стартує
- **Оптимізація часу**: повне тренування запускається тільки якщо код та дані ОК

---

## 4. Pre-train vs Post-train тести

### Pre-train (до тренування)

Мета: перевірити що **вхідні дані** та **конфігурація** валідні.

| Тест | Що перевіряє | Файл |
|------|-------------|------|
| `test_split_dir_exists` | `data/split/` існує | `test_data.py` |
| `test_dataset_yaml_keys` | `kpt_shape=[2,2]`, наявність ключів | `test_data.py` |
| `test_images_have_labels` | Кожне .jpg має .txt label | `test_data.py` |
| `test_label_format` | 9 значень на рядок (YOLO Pose) | `test_data.py` |
| `test_train_params_valid` | `epochs>0`, `lr0∈(0,1)` | `test_config.py` |
| `test_model_weights_available` | Базова модель доступна | `test_config.py` |

**Чому саме ці тести:**
- YOLO Pose формат має строгу структуру: `class cx cy w h kp1x kp1y kp2x kp2y`
- Одна помилка в label (9 → 10 значень) — тренування може мовчки провалитися
- Бджолиний датасет: всі об'єкти — class 0, координати нормалізовані [0,1]

### Post-train (після тренування)

Мета: перевірити що **модель та метрики** відповідають вимогам.

| Тест | Що перевіряє | Файл |
|------|-------------|------|
| `test_metrics_json_exists` | `metrics.json` створений | `test_artifacts.py` |
| `test_metrics_json_valid` | Містить `mAP50`, `mAP50-95` | `test_artifacts.py` |
| `test_quality_gate_map50` | `mAP50 ≥ threshold` | `test_artifacts.py` |
| `test_metrics_values_reasonable` | Значення в [0, 1] | `test_artifacts.py` |

---

## 5. Quality Gate

**Quality Gate** — формалізоване правило, яке автоматично визначає, чи приймається зміна.

### Як це працює в нашому проєкті

```
metrics.json:
{
  "mAP50": 0.45,        ← поточне значення
  "mAP50-95": 0.22
}

Quality Gate:
  MAP50_THRESHOLD=0.01   ← пороги (env vars)
  MAP50_95_THRESHOLD=0.005

Результат:
  mAP50: 0.45 ≥ 0.01  → ✅ PASSED
```

### Два режими порогів

| Режим | MAP50_THRESHOLD | Мета |
|-------|----------------|------|
| CI (1 epoch, CPU) | 0.01 (або 0.0) | Перевірити що pipeline працює |
| Production | 0.30+ | Гарантувати якість моделі |

### Ризики Quality Gate

| Ризик | Проблема | Як мінімізувати |
|-------|----------|----------------|
| False Positive | Поріг зависокий → корисні зміни відхиляються | Обґрунтувати поріг на baseline експериментах |
| False Negative | Поріг занизький → пропускаємо деградацію | Додати baseline comparison (Δ) |
| Хибне порівняння | CI = 1 epoch ≠ реальна якість | Розділити CI-smoke та production threshold |

---

## 6. CML (Continuous Machine Learning)

**CML** — інструмент від Iterative.ai (творці DVC) для ML-звітів у Git.

### Що робить CML у нашому PR

```
Pull Request ←── CML коментар:
┌──────────────────────────────────────┐
│ ## 🤖 Model CI Report               │
│                                      │
│ ### 📊 Metrics                       │
│ | Metric    | Value  |               │
│ |-----------|--------|               │
│ | mAP50     | 0.4500 |               │
│ | mAP50-95  | 0.2200 |               │
│                                      │
│ ### 📈 Comparison vs. Baseline       │
│ | Metric  | Baseline | Current | Δ   │
│ |---------|----------|---------|---- │
│ | mAP50   | 0.4200   | 0.4500  | 🟢 │
│                                      │
│ ### 🚦 Quality Gate                  │
│ mAP50: 0.4500 (threshold: 0.01) → ✅│
│                                      │
│ ### 📸 Confusion Matrix              │
│ [зображення]                         │
└──────────────────────────────────────┘
```

### Чому cml publish?

Зображення в PR-коментарях не можуть бути локальними файлами.  
`cml publish confusion_matrix.png --md` завантажує зображення на сервер і повертає markdown-посилання.

---

## 7. Baseline Comparison

**Baseline** — збережені метрики «еталонного» запуску моделі.

```
baseline/metrics.json          ←  зберігається в git
metrics.json                   ←  генерується під час CI

scripts/compare_metrics.py:
  Читає обидва файли → таблиця (було/стало/Δ)
```

### Навіщо

- Code review: рецензент бачить **конкретний вплив** змін на якість
- Regression testing: автоматично виявляємо деградацію
- Історичність: baseline оновлюється при merge в main

---

## 8. Особливості CI для YOLO/Object Detection

### Проблема: YOLO потребує GPU

GitHub Actions runners — CPU only. Повне тренування YOLO:
- 50 épох, imgsz=640 → **~2-4 години** на T4 GPU
- На CPU → **неможливо** за розумний час

### Рішення: CI smoke test

```yaml
CI_EPOCHS: 1        # 1 епоха замість 50
CI_IMGSZ: 160       # 160px замість 640px
CI_BATCH: 4         # 4 замість 16
device: cpu          # CPU only
```

Це не дає реальних метрик якості, але:
- ✅ Перевіряє що код не зламаний
- ✅ Перевіряє що pipeline цілий (дані → модель → метрики)
- ✅ Генерує артефакти для Quality Gate та CML
- ✅ Займає ~3-5 хвилин на CPU

### Повне тренування — окремо

```
CI (push/PR):     1 époch, CPU, imgsz=160  → smoke test
Colab/Local:      50 époch, GPU, imgsz=640 → реальне тренування
Schedule (раз/тиждень): повний retraining  → baseline update
```

---

## 9. Секрети та безпека

### GitHub Secrets для цього проєкту

| Secret | Призначення |
|--------|------------|
| `DAGSHUB_USER` | Username для DVC remote (DagsHub) |
| `DAGSHUB_TOKEN` | Token для DVC remote (DagsHub) |
| `GITHUB_TOKEN` | Автоматичний, для CML коментарів у PR |

### Як додати

Settings → Secrets and variables → Actions → New repository secret

### Чому не зберігаємо в коді

- `.env` файл у `.gitignore` — ніколи не потрапляє в Git
- Секрети в GitHub Actions зашифровані та не показуються в логах
- DVC remote credentials — доступ до 360 MB датасету бджіл

---

## 10. Повна структура CI/CD файлів

```
mlops_project/
├── .github/workflows/
│   └── ci.yaml              # GitHub Actions workflow
│
├── tests/
│   ├── conftest.py          # Shared fixtures (шляхи, пороги)
│   ├── test_data.py         # Pre-train: валідація даних
│   ├── test_config.py       # Pre-train: валідація конфігурації
│   └── test_artifacts.py    # Post-train: Quality Gate
│
├── scripts/
│   ├── ci_train.py          # Lightweight CI тренування
│   ├── compare_metrics.py   # Порівняння з baseline
│   └── generate_report.py   # CML звіт для PR
│
├── baseline/
│   └── metrics.json         # Еталонні метрики
│
├── pyproject.toml           # pytest + ruff конфіг
└── .gitignore               # CI-артефакти виключені
```

---

## 11. Команди для запуску

```bash
# Локальна перевірка
uv run ruff check src/ scripts/ tests/          # лінтинг
uv run pytest tests/test_data.py tests/test_config.py -v  # pre-train тести
uv run python scripts/ci_train.py               # CI тренування
uv run pytest tests/test_artifacts.py -v         # Quality Gate
uv run python scripts/compare_metrics.py         # порівняння з baseline

# GitHub Actions
# Автоматично при push/PR — перевірити на GitHub → Actions tab
```

---

## 12. Відповіді на контрольні питання

### 1. Що таке CI і яку проблему воно вирішує у ML-проєктах?

**CI (Continuous Integration)** — практика автоматичної перевірки якості кожної зміни одразу при її надходженні (commit/PR).

У звичайному ПЗ CI перевіряє код. В ML-проєктах додаються специфічні ризики:
- **Тиха деградація**: код компілюється, тести проходять, але mAP впав на 20% через зміну препроцесингу
- **Зміна даних**: нова партія зображень бджіл з іншим освітленням → модель деградує без жодної помилки коду
- **Невідтворюваність**: «у мене на GPU працювало» — але конфігурація, seed або версія датасету різна

CI для YOLO Pose вирішує: автоматично перевіряє що зміна коду/конфігу/даних не ламає пайплайн і не погіршує метрики нижче визначеного порогу.

---

### 2. У чому основна різниця між CI та CD у контексті цієї лабораторної?

| | CI | CD |
|---|---|---|
| **Мета** | Перевірити зміну | Доставити артефакт |
| **Тригер** | push / PR | merge в main |
| **Результат** | ✅/❌ pipeline | Опублікований артефакт |

**В цьому проєкті:**
- **CI**: lint → pre-train тести → CI train (1 epoch) → Quality Gate → CML report у PR
- **CD**: після merge в `main` — `upload-artifact` публікує `best.pt` як GitHub Artifact (або реєстрація в MLflow Registry)

---

### 3. Які основні складові GitHub Actions: workflow, job, step, runner?

```
Workflow (.github/workflows/ci.yaml)
│  ← запускається при push/PR
│
├── Job: lint           (runner: ubuntu-latest)
│   ├── Step: checkout
│   ├── Step: install uv
│   └── Step: ruff check
│
├── Job: pre-train-tests  (runner: ubuntu-latest, паралельно з lint)
│   ├── Step: dvc pull
│   ├── Step: prepare.py
│   └── Step: pytest test_data.py test_config.py
│
└── Job: train-and-test  (runner: ubuntu-latest, після lint + pre-train)
    ├── Step: CI train (1 epoch, CPU)
    ├── Step: pytest test_artifacts.py  ← Quality Gate
    ├── Step: generate_report.py + cml comment
    └── Step: upload-artifact
```

| Термін | В нашому проєкті |
|--------|-----------------|
| **Workflow** | `ci.yaml` |
| **Event** | `push` до main, `pull_request`, `workflow_dispatch` |
| **Job** | `lint`, `pre-train-tests`, `train-and-test` |
| **Step** | Кожен `- name: ...` блок |
| **Runner** | `ubuntu-latest` (GitHub-hosted VM) |

---

### 4. Чому для ML-проєктів важливо зберігати метрики у машинно-читаному форматі?

Якби метрики були лише у консолі:
- CI не може **автоматично перевірити** Quality Gate (треба парсити текст — крихко)
- CML не може **побудувати таблицю** порівняння з baseline
- MLflow/DVC не можуть **версіонувати** метрики для наступних порівнянь

У нашому проєкті `ci_train.py` зберігає `metrics.json`:
```json
{
  "mAP50": 0.4500,
  "mAP50-95": 0.2200,
  "precision": 0.8100,
  "recall": 0.6700
}
```
Це дозволяє `test_artifacts.py` читати `metrics["mAP50"]` і порівнювати з порогом, а `generate_report.py` — будувати таблицю Baseline vs Current з Δ.

---

### 5. Що таке Quality Gate і навіщо він потрібен у Pull Request?

**Quality Gate** — формалізоване правило, яке автоматично визначає, чи приймається зміна.

**Без Quality Gate**: рецензент змушений мануально перевіряти чи не погіршились метрики → пропускає деградацію.

**З Quality Gate** у нашому PR:
```
pytest tests/test_artifacts.py -v
  test_quality_gate_map50: mAP50=0.45 ≥ 0.01 → ✅ PASSED
  test_quality_gate_map50_95: mAP50-95=0.22 ≥ 0.005 → ✅ PASSED
```

Якщо тест фейлиться — PR блокується автоматично, без участі людини.

**Два рівні порогів** в проєкті:
- CI (1 epoch): `MAP50_THRESHOLD=0.0` — smoke test (перевіряємо що пайплайн не зламаний)
- Production: `0.30+` — реальна якісна гарантія

---

### 6. Різниця між pre-train та post-train тестами (по 2 приклади)

**Pre-train** — до тренування, перевіряємо вхідні умови:

| Тест | Файл | Що перевіряє |
|------|------|-------------|
| `test_label_format` | `test_data.py` | Кожен label має рівно 9 значень (YOLO Pose формат) |
| `test_train_params_valid` | `test_config.py` | `epochs > 0`, `lr0 ∈ (0,1)`, коректний `imgsz` |

**Post-train** — після тренування, перевіряємо результати:

| Тест | Файл | Що перевіряє |
|------|------|-------------|
| `test_metrics_json_exists` | `test_artifacts.py` | `metrics.json` створений скриптом `ci_train.py` |
| `test_quality_gate_map50` | `test_artifacts.py` | `mAP50 ≥ MAP50_THRESHOLD` (Quality Gate) |

---

### 7. Типові перевірки data validation для YOLO Pose

| Перевірка | Реалізація | Чому важлива |
|-----------|-----------|-------------|
| Структура директорій | `test_train_val_dirs_exist` | Без `train/images/` YOLO не стартує |
| Формат міток (9 значень) | `test_label_format` | YOLO Pose строго вимагає: `class cx cy w h kp1x kp1y kp2x kp2y` |
| Кожне зображення має label | `test_images_have_labels` | Зображення без label → YOLO пропускає або падає |
| `kpt_shape=[2,2]` у dataset.yaml | `test_dataset_yaml_keys` | 2 keypoints, 2D координати — специфіка нашого bee pose датасету |
| Мінімальна кількість зразків | `test_min_images` | ≥20 train, ≥5 val — інакше навчання безглузде |
| Координати bbox ∈ [0,1] | `test_label_format` | Нормалізовані координати — YOLO вимога |

**Чому саме ці:** YOLO Pose має суворий формат вхідних даних, одна помилка (наприклад, 10 значень замість 9) спричиняє тихий збій під час тренування.

---

### 8. Які permissions потрібні для коментарів у PR?

У `ci.yaml`:
```yaml
permissions:
  contents: read
  pull-requests: write   # ← потрібен для CML comment create
```

**Що станеться без `pull-requests: write`:**
CML спробує створити коментар через GitHub API і отримає `403 Forbidden`. Workflow завершиться з помилкою на кроці `cml comment create`.

**Для PR з форків** — `GITHUB_TOKEN` не має прав на PR від сторонніх форків. Тоді використовується PAT (Personal Access Token) як секрет `CML_TOKEN`, передається як `REPO_TOKEN`.

---

### 9. Чому зображення в CML-звіті краще додавати через `cml publish`?

PR-коментарі в GitHub — це markdown-текст. Markdown підтримує лише **URL-посилання**, а не локальні шляхи файлів.

Без `cml publish`:
```markdown
![confusion matrix](confusion_matrix.png)  ← не відображається, це локальний шлях
```

З `cml publish confusion_matrix.png --md`:
```markdown
![confusion_matrix](https://asset.cml.dev/abc123.png)  ← відображається
```

`cml publish` завантажує файл на сервер (GitHub Assets або DVC Studio) і повертає готовий markdown із публічним URL.

---

### 10. Як забезпечити доступність даних у CI-раннері (DVC remote)?

**Проблема:** GitHub Actions runner — «чиста» VM, без ваших даних. Датасет (360 MB зображень бджіл) знаходиться на DagsHub.

**Рішення у `ci.yaml`:**
```yaml
- name: Setup DVC credentials
  env:
    DAGSHUB_USER: ${{ secrets.DAGSHUB_USER }}
    DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
  run: |
    uv run dvc remote modify origin --local auth basic
    uv run dvc remote modify origin --local user "$DAGSHUB_USER"
    uv run dvc remote modify origin --local password "$DAGSHUB_TOKEN"

- name: Pull data (DVC)
  run: uv run dvc pull
```

**Категорії секретів:**
| Secret | Тип | Де використовується |
|--------|-----|---------------------|
| `DAGSHUB_USER` | DVC remote credentials | `dvc remote modify ... user` |
| `DAGSHUB_TOKEN` | DVC remote credentials | `dvc remote modify ... password` |
| `GITHUB_TOKEN` | Автоматичний | CML comment create |

Секрети додаються: `Settings → Secrets and variables → Actions → New repository secret`.

---

### 11. Ризики «хибно зеленого» та «хибно червоного» Quality Gate

| Ризик | Опис | Приклад | Як мінімізувати |
|-------|------|---------|----------------|
| **False Positive** (хибно зелений) | CI проходить, але модель погана | CI поріг = 0.0 → будь-який результат OK | Розділити CI-smoke та production пороги |
| **False Negative** (хибно червоний) | CI фейлиться на корисній зміні | 1 epoch не дає достатній mAP | Обґрунтувати поріг на базі baseline exp. |
| **Хибне порівняння** | CI (1 epoch) ≠ реальна якість | mAP при 1 epoch і 50 epoch несопоставні | Мати окремі пороги для CI та prod |

**В нашому проєкті:**
- CI Quality Gate: `MAP50_THRESHOLD=0.0` (smoke test — перевіряємо лише що pipeline не зламаний)
- Production: baseline comparison показує реальне Δ від повного тренування

---

### 12. Обґрунтування порогу Quality Gate

**Підхід:**

1. **Провести серію baseline-експериментів** (5-10 запусків з різними seed) → зафіксувати середнє mAP50 та стандартне відхилення σ
2. **Встановити поріг = середнє − 2σ** → відсікає аномальну деградацію, але не блокує нормальну варіацію
3. **Обновити baseline після merge** у main → baseline відображає поточний «кращий» стан

**Для нашого YOLO11n-pose на bee датасеті:**
- Повне тренування (50 epoch, imgsz=640): mAP50 ≈ 0.40–0.55
- Production поріг: ~0.35 (залишаємо ~15% запас)
- CI поріг: 0.0 (smoke test, 1 epoch на CPUдає ~0.01–0.05)

---

### 13. Дизайн baseline comparison у PR

**Структура в проєкті:**
```
baseline/metrics.json    ← зберігається в git, "еталон" на main
metrics.json             ← генерується при CI тренуванні
```

**`scripts/compare_metrics.py`** читає обидва файли та будує таблицю:

| Metric | Baseline | Current | Δ |
|--------|----------|---------|---|
| mAP50 | 0.4200 | 0.4500 | 🟢 +0.0300 |
| mAP50-95 | 0.2100 | 0.2200 | 🟢 +0.0100 |

**Для рецензента цінне:**
- Конкретне числове Δ замість «нічого не змінилось»
- 🟢/🔴 візуальний індикатор
- Блокування PR якщо Δ < −threshold (регресивне тестування)

**Оновлення baseline:** після merge в main — вручну або через scheduled job оновити `baseline/metrics.json`.

---

### 14. Як зробити CI швидким для YOLO?

**Проблема:** YOLO11 на повному датасеті (50 epoch, imgsz=640) → 2-4 год на T4 GPU → неможливо на CPU-runner.

**Рішення в проєкті:**

```yaml
CI_EPOCHS: 1       # 1 замість 50
CI_IMGSZ: 160      # 160px замість 640px
CI_BATCH: 4        # 4 замість 16
device: cpu        # CPU, не GPU
```

| Техніка | Виграш | Компроміс |
|---------|--------|----------|
| 1 epoch замість 50 | ~50x | Метрики нереалістичні |
| imgsz=160 замість 640 | ~16x | Дрібні об'єкти можуть не детектуватися |
| Паралельні job-и (lint + pre-train) | ~2x | Потребує більше runner-квоти |
| pip/uv кеш | ~30-60с | Потрібна правильна настройка |
| schedule для повного retrain | CI залишається швидким | Затримка у виявленні проблем |

**Архітектура:**
- `push/PR` → smoke test (1 epoch, CPU, ~5 хв)
- `schedule` (раз/тиждень) → повне тренування (50 epoch, GPU через self-hosted runner або Colab)

---

### 15. Policy-as-code для ML-якості

**Ідея:** правила якості (пороги, вимоги до даних, вимоги до артефактів) мають бути у версіонованих файлах, а не «у голові» розробника.

**Реалізація в нашому проєкті:**

```
pyproject.toml          ← конфіг pytest, ruff
params.yaml             ← гіперпараметри (версіоновані DVC)
tests/conftest.py       ← пороги Quality Gate (читаються з env vars)
.github/workflows/ci.yaml ← env vars: MAP50_THRESHOLD=0.0
baseline/metrics.json   ← "еталонні" метрики (в git)
```

**Переваги:**
- Зміна порогу = PR з peer review → прозорість
- Пороги прив'язані до конкретного git commit
- `git blame` показує хто і чому змінив поріг

**Розширення:** можна використати окремий `quality_policy.yaml`:
```yaml
quality_gate:
  mAP50_min: 0.35
  mAP50_95_min: 0.15
  min_train_images: 100
  required_artifacts:
    - metrics.json
    - confusion_matrix.png
```
Який читається і `conftest.py`, і `generate_report.py` — єдине джерело правди.
