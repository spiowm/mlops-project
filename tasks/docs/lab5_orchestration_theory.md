# Лабораторна №5 — Оркестрація ML-пайплайнів: від CI/CD до Continuous Training

> **Реалізація в проєкті**: YOLO11 Pose Estimation (бджоли) з Prefect Flows

---

## Зміст

1. [Що таке оркестрація і навіщо вона потрібна](#1-що-таке-оркестрація-і-навіщо-вона-потрібна)
2. [Рівні зрілості MLOps — де ми знаходимось](#2-рівні-зрілості-mlops)
3. [Чим відрізняється CI/CD від Continuous Training](#3-різниця-між-cicd-та-continuous-training)
4. [Що таке DAG і як він виглядає в коді](#4-що-таке-dag)
5. [Чому Prefect, а не Airflow — детальне пояснення](#5-prefect-vs-airflow)
6. [Контейнеризація з Docker — навіщо і як](#6-контейнеризація-docker)
7. [Як це реалізовано в проєкті — покроково](#7-реалізація-в-проєкті)
8. [Як запустити — повна інструкція](#8-як-запустити)
9. [Відповіді на контрольні питання](#9-відповіді-на-контрольні-питання)

---

## 1. Що таке оркестрація і навіщо вона потрібна

### Спочатку — проблема без оркестратора

Уяви, що у тебе є ML-пайплайн. Наприклад, такий скрипт `run_all.sh`:

```bash
python src/prepare.py         # підготувати дані
python src/train.py           # навчити модель
python tests/test_artifacts.py # перевірити якість
python scripts/register_model.py  # зареєструвати модель
```

Здається розумно. Але що відбувається на практиці?

| Проблема | Що станеться |
|----------|-------------|
| `train.py` впав через timeout | Скрипт зупинився. `register_model.py` ніколи не запустився. Ти не знаєш про це. |
| Сервер виключили на кроці 2 | Всі попередні результати втрачені, треба починати з нуля |
| Хочеш запускати кожного тижня | Вручну? Через `cron`? А якщо сервер в той момент перезавантажується? |
| Хочеш знати що зараз виконується | Дивишся в термінал і молишся |
| Крок 2 залежить від крок 1 | Логіка захардкоджена в скрипті — крихка і незрозуміла |

**Оркестратор** — це система, яка управляє виконанням складного пайплайну:
- знає про залежності між кроками ("train після prepare")
- автоматично перезапускає кроки при збоях
- дає UI для спостереження
- може запускати за розкладом або по тригеру
- дозволяє перезапустити лише невдалі кроки, не весь пайплайн

### Що таке "оркестрація" простими словами

Уяви диригента оркестру. Він не грає сам — він управляє музикантами, каже кому вступати і коли, слідкує за загальною картиною. **Оркестратор ML-пайплайну** — це той самий диригент, але для ML-задач.

---

## 2. Рівні зрілості MLOps

Google описує еволюцію MLOps як 3 рівні. Ось де знаходиться **наш проєкт**:

```
ЛР1-2: Рівень 0                ЛР3: Рівень 1              ЛР4-5: Рівень 2
────────────────────────       ───────────────────────    ─────────────────────────
Jupyter Notebook               DVC pipeline               GitHub Actions (CI код)
Ручне тренування               Optuna HPO                 Prefect Flow (CT модель)
Немає версіонування            MLflow Tracking            Автоматична реєстрація
Файл send через email          DVC Data versioning        Docker контейнери
```

### Рівень 0 — "Все вручну"

Дата-сайентист запускає ноутбук, отримує модель, вручну кладе файл `best.pt` на сервер. Якщо щось пішло не так — він і сам не знає. Відтворити той самий результат через тиждень — квест.

### Рівень 1 — "Автоматизований пайплайн"

З'являється **DVC** (версіонування даних), **MLflow** (відстеження метрик), **Optuna** (автоматичний підбір параметрів). Тренування запускається командою, а не вручну через ноутбук. Але оновлення коду все ще вручну, без перевірок.

### Рівень 2 — "Повна автоматизація" ← **де ми після ЛР5**

Два незалежних автоматизованих потоки:

**Потік 1 — CI/CD для коду** (GitHub Actions, ЛР4):
> "Захостив PR → код перевірили → тести пройшли → можна мерджити"

**Потік 2 — Continuous Training для моделі** (Prefect, ЛР5):
> "Раз на тиждень: завантажити нові дані → навчити → перевірити → якщо краще — оновити модель"

---

## 3. Різниця між CI/CD та Continuous Training

Це базова концепція ЛР5. Люди часто плутають.

```
GitHub Actions (CI/CD)            Prefect Flow (Continuous Training)
──────────────────────────        ──────────────────────────────────
ЩО перевіряє: КОД                ЩО оновлює: МОДЕЛЬ
КОЛИ: при кожному push/PR         КОЛИ: за розкладом або при нових даних
ТРЕНУВАННЯ: 1 epoch smoke test    ТРЕНУВАННЯ: повне (скільки треба)
СЕРЕДОВИЩЕ: GitHub-hosted runner  СЕРЕДОВИЩЕ: твій сервер / локально
ЦІЛЬ: "чи не зламали ми код?"    ЦІЛЬ: "чи не застаріла наша модель?"
РЕЗУЛЬТАТ: CML report в PR        РЕЗУЛЬТАТ: нова версія моделі в MLflow
```

**Практичний приклад:**

Уяви що ти додав новий файл `src/utils.py` з функцією. GitHub Actions перевірить:
- чи немає синтаксичних помилок (ruff lint)
- чи проходять тести (pytest)
- чи запускається тренування хоча б на 1 epoci (smoke test)

Але **дані для бджіл** не змінились. Модель не оновилась.

А тепер уяви що в серпні прилетіли нові партії бджіл і ти додав 100 нових фото до датасету. Prefect Flow побачить нові дані, запустить повне тренування, перевірить що mAP50 покращився — і автоматично оновить baseline. **Без жодного push в GitHub**.

---

## 4. Що таке DAG

**DAG = Directed Acyclic Graph = Спрямований ациклічний граф**

Страшна назва, проста суть:

```
prepare_data ──→ train_model ──→ evaluate_model ──→ register_model
                                       ↓
                                 notify_failure  (якщо якість погана)
```

- **Вузол** (node) = одна задача (підготовка даних, тренування, тощо)
- **Стрілка** (edge) = залежність ("train" починається лише після "prepare")
- **Directed** = стрілки мають напрямок (не двосторонні)
- **Acyclic** = немає циклів (задача A не може залежати від задачі B, яка залежить від A)

### Чому не просто список кроків?

У графа є суперсила: **паралельне виконання**. Якщо два кроки не залежать один від одного — вони можуть виконуватись одночасно:

```
prepare_data ──→ check_config ──┐
                                ├──→ train_model
download_weights ───────────────┘
```

Тут `prepare_data`, `check_config`, і `download_weights` можна запустити паралельно — і час виконання скорочується.

### DAG у коді (Prefect-стиль)

```python
@task
def prepare_data():
    # підготовка даних...
    return True

@task
def train_model():
    # тренування YOLO...
    return Path("metrics.json")

@flow  # <-- це і є DAG
def my_pipeline():
    step1 = prepare_data()    # залежність визначається порядком виклику
    metrics = train_model()   # train починається після prepare
```

---

## 5. Prefect vs Airflow

Це найважливіше питання лаби — **чому ми обрали Prefect**.

### Apache Airflow — промисловий стандарт

Airflow — найпопулярніший оркестратор в індустрії. Google, Lyft, Airbnb — всі використовують. Але за що платимо?

**Щоб запустити Airflow, потрібно:**

```yaml
# docker-compose.yml для Airflow (мінімальний)
services:
  postgres:         # база даних для зберігання стану
    image: postgres:13
  redis:            # черга повідомлень для Celery
    image: redis:7
  airflow-webserver:  # UI
    image: apache/airflow:2.8.0
    depends_on: [postgres, redis]
  airflow-scheduler:  # планувальник
    image: apache/airflow:2.8.0
    depends_on: [postgres, redis]
  airflow-worker:     # виконавець задач
    image: apache/airflow:2.8.0
    depends_on: [redis]
  airflow-triggerer:  # обробник сенсорів
    image: apache/airflow:2.8.0
```

Це **6 Docker-контейнерів** тільки для запуску оркестратора. Ще до того як написати перший рядок свого коду.

Плюс:
- Спеціальний синтаксис для DAG (не звичайний Python)
- Потрібно вручну встановлювати залежності через `airflow.cfg`
- DAG-файли мають бути в певних папках
- Debugging — окремий квест

**Коли Airflow виправданий:**
- Команда 10+ людей
- Сотні DAG-ів у продакшні
- Складні залежності між різними командами
- Корпоративна інфраструктура

### Prefect — Python-first оркестратор

Prefect (версія 2/3) — повністю переосмислений підхід.

**Щоб запустити Prefect:**
```bash
pip install prefect
python my_flow.py  # просто запустити Python файл
```

Все. Ніяких додаткових сервісів для базового запуску.

**Синтаксис — звичайний Python:**

```python
# Airflow (специфічний синтаксис)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG("my_dag", start_date=datetime(2024,1,1)) as dag:
    task = PythonOperator(
        task_id="my_task",
        python_callable=my_function,
    )

# Prefect (звичайний Python)
@task
def my_function():
    return "result"

@flow
def my_flow():
    my_function()
```

**Порівняння для нашого проєкту:**

| Аспект | Apache Airflow | Prefect 3 |
|--------|---------------|-----------|
| Залежності | PostgreSQL + Redis + 6 контейнерів | pip install prefect (1 пакет) |
| Синтаксис | Специфічний API Airflow | Звичайний Python + декоратори |
| Локальний запуск | Складно (потрібен весь stack) | `python flow.py` |
| Debugging | Через airflow logs + UI | print() + стандартний Python дебагер |
| UI | Є (важкий, slow) | Є (легкий, Prefect Cloud або локальний) |
| Підходить для |  Корпоративний продакшн | Навчання, стартап, одна людина |
| "Відчуття" | як Spring Boot (потужно але важко) | як FastAPI (просто і елегантно) |

### Чи норм ідея — використовувати Prefect?

**Так, і ось чому для цього проєкту:**

1. **Ставиться задача "зрозуміти оркестрацію"**, а не "налаштувати Airflow". З Prefect ти розумієш ключові концепції (DAG, tasks, flows, branching, retry) без боротьби з інфраструктурою.

2. **Навчальна цінність однакова**: концепції DAG, Sensor/Task, Branching, Retry — все те саме, просто синтаксис простіший.

3. **Перенести знання легко**: якщо потім треба Airflow — ти вже розумієш концепції, залишається тільки вивчити специфічний синтаксис.

4. **В реальній індустрії Prefect використовується**: Airflow не єдиний вибір. Багато ML-команд переходять на Prefect/Dagster саме через простоту.

5. **Лаба каже**: *"ви може обрати довільний фреймворк або бібліотеку на власний розсуд. Основна мета — дотримання принципів та етапів MLOps"*.

---

## 6. Контейнеризація Docker

### Навіщо взагалі Docker для ML?

**Без Docker** проблема №1 в ML командах:
> "У мене на ноуті працює! А на сервері — ні!"

Причина: різні версії Python, бібліотек, система інша. Docker вирішує це — фіксує **все середовище** у файлі.

**Docker Image** — це як "знімок" всього середовища:
- Python 3.12 (конкретна версія)
- ultralytics 8.4.19 (конкретна версія)
- всі інші залежності
- твій код

Запускаєш на будь-якій машині — однаковий результат.

### Multi-stage build — навіщо два етапи?

**Проблема одного Dockerfile:**
```dockerfile
FROM python:3.12
RUN pip install ultralytics  # важкий пакет
RUN apt-get install gcc build-essential  # потрібно для компіляції
COPY . .
```

Результат: образ 4+ GB, бо `gcc`, `build-essential` залишаються в образі назавжди — а в production вони не потрібні.

**Multi-stage вирішує це:**

```dockerfile
# === Етап 1: Збірка (важкий) ===
FROM python:3.12-slim AS builder
RUN apt-get install build-essential   # потрібен для компіляції C-extensions
RUN pip install uv
COPY pyproject.toml .
RUN uv sync  # встановлюємо всі залежності в .venv

# === Етап 2: Runtime (легкий) ===
FROM python:3.12-slim AS runtime
# Беремо лише готове .venv з першого етапу
COPY --from=builder /app/.venv /app/.venv
# build-essential, gcc — НЕ копіюємо. Вони більше не потрібні.
COPY src/ scripts/ tests/ dags/ .
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "scripts/ci_train.py"]
```

**Аналогія**: будинок будують з будівельними лісами (scaffolding). Коли будинок готовий — ліси знімають. Multi-stage build — це "зняти ліси" перед тим, як здати будинок.

**Результат:**
- Builder stage: \~2 GB (gcc + всі dev deps)
- Runtime stage: \~600 MB (тільки .venv + код)
- Образ на 70% менший → завантажується швидше → CI runner запускається швидше

---

## 7. Реалізація в проєкті

### Нові файли (ЛР5)

```
mlops_project/
├── dags/
│   ├── __init__.py
│   └── yolo_training_pipeline.py   ← Prefect Flow (головний файл ЛР5)
│
├── Dockerfile                       ← Multi-stage build
│
├── tests/
│   └── test_dag.py                 ← 12 тестів цілісності Flow
│
└── .github/workflows/
    └── ci.yaml                      ← оновлено: + dag-tests + docker-build jobs
```

### Як влаштований `dags/yolo_training_pipeline.py`

Це центральний файл ЛР5. Розберемо його крок за кроком.

#### Крок 1 — `check_data_freshness` (аналог Airflow Sensor)

```python
@task(name="Check Data Freshness", retries=2, retry_delay_seconds=30)
def check_data_freshness() -> dict:
    """Перевіряє чи є дані. Якщо немає — сигналізує що треба prepare."""
    split_dir = PROJECT_ROOT / "data" / "split"
    return {
        "needs_prepare": not (split_dir / "dataset.yaml").exists(),
        "train_images": len(list(...)),
    }
```

`retries=2, retry_delay_seconds=30` — якщо задача впала (наприклад, мережева помилка при DVC pull) — Prefect автоматично спробує ще двічі через 30 секунд.

Роль **Sensor** з Airflow: у нашій реалізації це звичайна task, яка перевіряє наявність даних. В Airflow був би окремий `FileSensor`, але концепція та сама.

#### Крок 2 — `prepare_data`

```python
@task(name="Prepare Data", retries=1)
def prepare_data(data_status: dict) -> bool:
    if not data_status["needs_prepare"]:
        return True  # ідемпотентність: якщо вже готово — пропускаємо
    subprocess.run(["python", "src/prepare.py"])
```

**Ідемпотентність** вбудована: якщо дані вже підготовлені — task не перезапускає `prepare.py` зайвий раз.

#### Крок 3 — `run_pretrain_tests`

```python
@task(name="Run Pre-train Tests")
def run_pretrain_tests() -> bool:
    result = subprocess.run([
        "pytest", "tests/test_data.py", "tests/test_config.py", "-v"
    ])
    if result.returncode != 0:
        raise RuntimeError("Pre-train tests failed — pipeline зупинено")
```

Якщо тести падуть — пайплайн зупиняється. Тренування **не запуститься** з поганими даними.

#### Крок 4 — `train_model`

```python
@task(name="Train Model", retries=1, retry_delay_seconds=60)
def train_model() -> Path:
    subprocess.run(["python", "scripts/ci_train.py"])
    if not METRICS_PATH.exists():
        raise RuntimeError("metrics.json не створено")
    return METRICS_PATH
```

Повертає **шлях** до metrics.json — наступна task отримає його як вхідний параметр.

#### Крок 5 — `evaluate_model`

```python
@task(name="Evaluate Model (Quality Gate)")
def evaluate_model(metrics_path: Path) -> dict:
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {
        "mAP50": metrics.get("mAP50", 0.0),
        "mAP50-95": metrics.get("mAP50-95", 0.0),
    }
```

#### Крок 6 — `check_quality_gate` (аналог BranchPythonOperator)

```python
@task(name="Check Quality Gate")
def check_quality_gate(evaluation: dict) -> bool:
    map50 = evaluation["mAP50"]
    return map50 >= MAP50_THRESHOLD  # True або False
```

Повертає булеве значення. Головний Flow на основі нього обирає гілку.

#### Крок 7 — `register_model` або `notify_failure`

```python
@task(name="Register Model")
def register_model(evaluation: dict):
    # Оновлює baseline/metrics.json
    # Публікує Markdown звіт у Prefect Artifacts
    with open(BASELINE_PATH, "w") as f:
        json.dump(evaluation, f)

@task(name="Notify Failure")
def notify_failure(evaluation: dict):
    # В production: email/Slack
    # У нас: Prefect Artifact зі звітом про провал
```

#### Головний Flow — все разом

```python
@flow(name="YOLO Pose CT Pipeline")
def yolo_pose_training_pipeline(ct_epochs=1, ct_imgsz=320, map50_threshold=0.0):
    # Послідовне виконання (кожен крок залежить від попереднього)
    data_status = check_data_freshness()          # Крок 1
    prepare_data(data_status)                      # Крок 2
    run_pretrain_tests()                           # Крок 3
    metrics_path = train_model()                   # Крок 4
    evaluation = evaluate_model(metrics_path)      # Крок 5
    gate_passed = check_quality_gate(evaluation)   # Крок 6

    # Branching — аналог BranchPythonOperator
    if gate_passed:
        register_model(evaluation)      # Гілка "успіх"
    else:
        notify_failure(evaluation)      # Гілка "провал"
```

### Як CI/CD та Continuous Training взаємодіють

```
Розробник:                         Оркестратор (розклад):
──────────────────────             ──────────────────────────────
git push → GitHub Actions          Кожного тижня або вручну
    │                                  │
    ▼                                  ▼
lint (ruff)            check_data → prepare → pre-train tests
test_dag.py            → train (повна, не smoke) → evaluate
docker build check     → Quality Gate → register/notify
smoke train (1 epoch)      │
CML report in PR           ▼
    │                  MLflow: нова версія моделі
    ▼                  baseline/metrics.json оновлено
GitHub Artifacts
```

**Ключовий момент**: обидва пайплайни ведуть до MLflow як єдиного джерела правди про моделі. CI/CD перевіряє що **код** не зламаний. CT оновлює **модель**.

---

## 8. Як запустити

### Передумови

```bash
# 1. Клонувати та встановити залежності
git clone https://github.com/spiowm/mlops-project.git
cd mlops-project
uv sync

# 2. Створити .env файл
cp .env.example .env  # заповнити DAGSHUB_TOKEN, DAGSHUB_USER, DAGSHUB_REPO

# 3. Завантажити дані через DVC
uv run dvc pull
```

### Варіант 1 — Запустити CT Pipeline один раз (без UI)

Найпростіший варіант. Prefect запускається локально, без сервера.

```bash
# Smoke запуск (1 epoh, imgsz=320, threshold=0.0 — завжди проходить)
uv run python dags/yolo_training_pipeline.py

# Або з параметрами через env-змінні
CT_EPOCHS=1 CT_IMGSZ=320 CT_MAP50_THRESHOLD=0.0 \
  uv run python dags/yolo_training_pipeline.py
```

**Що відбудеться:**
1. Перевірить чи є `data/split/` → виведе лог
2. Запустить `src/prepare.py` якщо потрібно → дані підготовляться
3. Запустить `pytest tests/test_data.py tests/test_config.py` → якщо впадуть — зупиниться
4. Запустить `scripts/ci_train.py` → модель навчиться (1 epoh, CPU)
5. Прочитає `metrics.json` → виведе mAP50
6. Порівняє з threshold (0.0) → PASSED
7. Оновить `baseline/metrics.json` → виведе Prefect Artifact звіт

Вивід виглядатиме так:
```
============================================================
  YOLO Pose CT Pipeline
  epochs=1, imgsz=320, threshold=0.0
============================================================
[check_data_freshness] Дані знайдені: 245 train, 81 val
[prepare_data] Дані вже підготовлені, пропускаємо
[run_pretrain_tests] Pre-train тести ✓
[train_model] Тренування ✓ → metrics.json
[evaluate_model] Метрики: mAP50=0.1234, mAP50-95=0.0567
[check_quality_gate] ✅ Quality Gate PASSED: mAP50=0.1234 ≥ 0.0
[register_model] ✅ Модель зареєстрована (baseline оновлено)

Результат: {'status': 'success', 'mAP50': 0.1234, ...}
```

### Варіант 2 — Запуск з Prefect UI (для спостереження)

Якщо хочеш бачити граф виконання задач у красивому інтерфейсі:

```bash
# Термінал 1: запустити Prefect сервер
uv run prefect server start
# Відкриє UI на http://localhost:4200

# Термінал 2: запустити pipeline
uv run python dags/yolo_training_pipeline.py
```

Відкрий `http://localhost:4200` у браузері — побачиш:
- Граф виконання задач
- Статус кожної задачі (Running/Completed/Failed)
- Логи кожного кроку
- Artifacts (Markdown звіти про результати)

### Варіант 3 — Запуск тільки тестів Flow (для CI)

```bash
# Перевірити що Flow коректно визначений (без тренування)
uv run pytest tests/test_dag.py -v

# Очікуваний результат:
# tests/test_dag.py::test_pipeline_module_importable PASSED
# tests/test_dag.py::test_flow_function_exists PASSED
# tests/test_dag.py::test_flow_is_prefect_flow PASSED
# ... (12/12 тестів)
```

Це те, що робить GitHub Actions у job `dag-tests`.

### Варіант 4 — Docker запуск

```bash
# Збудувати образ
docker build --target runtime -t yolo-pose:local .

# Запустити smoke training в контейнері
docker run --rm \
  -v $(pwd)/data:/app/data \        # монтуємо дані
  -v $(pwd)/runs:/app/runs \        # щоб збереглись результати
  -v $(pwd)/metrics.json:/app/metrics.json \
  yolo-pose:local \
  python scripts/ci_train.py

# Перевірити що ultralytics встановлено коректно
docker run --rm yolo-pose:local \
  python -c "from ultralytics import YOLO; print('OK')"
```

### Варіант 5 — Через GitHub Actions (автоматично)

При кожному `push` або `pull_request` CI виконує:

```yaml
# .github/workflows/ci.yaml — нові job-и з ЛР5:

dag-tests:
  # uv run pytest tests/test_dag.py -v
  # Перевіряє що Flow завантажується без помилок

docker-build:
  # docker build --target runtime .
  # Перевіряє що Dockerfile будується
  # docker run --rm yolo-pose python -c "from ultralytics import YOLO; print('OK')"
```

### Параметри запуску CT Pipeline

| Параметр | Env змінна | Default | Опис |
|----------|-----------|---------|------|
| `ct_epochs` | `CT_EPOCHS` | `1` | Кількість epochs тренування |
| `ct_imgsz` | `CT_IMGSZ` | `320` | Розмір зображення |
| `map50_threshold` | `CT_MAP50_THRESHOLD` | `0.0` | Мінімальний mAP50 для реєстрації |

```bash
# Приклади різних режимів:

# Мінімальний (smoke, ~3 хвилини)
CT_EPOCHS=1 CT_IMGSZ=160 CT_MAP50_THRESHOLD=0.0 \
  uv run python dags/yolo_training_pipeline.py

# Реальний CT (потрібен GPU, ~30+ хвилин)
CT_EPOCHS=50 CT_IMGSZ=640 CT_MAP50_THRESHOLD=0.30 \
  uv run python dags/yolo_training_pipeline.py

# Суворіший Quality Gate
CT_MAP50_THRESHOLD=0.35 uv run python dags/yolo_training_pipeline.py
```

### Що перевірити після запуску

```bash
# 1. Чи оновились метрики?
cat metrics.json

# 2. Чи оновився baseline (якщо Quality Gate пройшов)?
cat baseline/metrics.json

# 3. Чи є результати тренування?
ls runs/ci/ci_train/weights/

# 4. Якщо є Prefect UI — відкрити http://localhost:4200
```

### Можливі помилки та рішення

| Помилка | Причина | Рішення |
|---------|---------|---------|
| `FileNotFoundError: data/split/dataset.yaml` | Дані не підготовлені | `uv run dvc pull && uv run python src/prepare.py` |
| `ModuleNotFoundError: prefect` | Prefect не встановлено | `uv sync` (він тепер у pyproject.toml) |
| `Pre-train tests FAILED` | Проблема з даними | Переглянути `pytest tests/test_data.py -v` детально |
| `Quality Gate FAILED` | mAP50 < threshold | Зменши `CT_MAP50_THRESHOLD` або збільш `CT_EPOCHS` |

---

## 9. Відповіді на контрольні питання

### 1. Як взаємодіють GitHub Actions та Prefect у системі?

**Коротко**: вони **незалежні**, але взаємодоповнюють. CI/CD охороняє **код**, CT оновлює **модель**.

```
Push коду       →  GitHub Actions  →  Перевірка коду (lint, tests, smoke train)
                                              ↓ (якщо OK)
                                        Дозвіл на merge у main

Нові дані /     →  Prefect Flow   →  Повне тренування → Quality Gate
Розклад                                       ↓ (якщо PASSED)
                                        Оновлення baseline/metrics.json
```

Обидва ведуть до **MLflow** як єдиного сховища метрик і артефактів моделей.

**Що запускає що:**
- Розробник робить PR → GitHub triggers Actions workflow
- Розклад (`schedule`) або ручний виклик → Prefect запускає Flow

---

### 2. Переваги DAG перед лінійним `run_all.sh`

| | `run_all.sh` | DAG (Prefect) |
|--|------|------|
| Збій на кроці 2 | Все зупиняється, нічого не виконується далі | Prefect зафіксував збій, можна перезапустити лише крок 2 |
| Retry | Немає | `@task(retries=2, retry_delay_seconds=30)` |
| Паралельність | Ні (всі кроки послідовно) | Незалежні кроки виконуються паралельно |
| Моніторинг | Дивишся в термінал, сподіваєшся | UI з графом, статусами, логами кожного кроку |
| Параметризація | Хардкодити або аргументи | `@flow` приймає параметри Python-функцій |
| Розклад | `cron` окремо (і хтось має за ним слідкувати) | Вбудований у Flow |

**Ключовий приклад**: `train_model` впав через timeout після 3х годин на кроці `evaluate`. З `run_all.sh` — запускаєш весь пайплайн знову, витрачаєш ще 3 години на тренування. З DAG — Prefect перезапускає лише `evaluate`.

---

### 3. BranchPythonOperator (Airflow) / Branching в Prefect

**BranchPythonOperator** — оператор Airflow, що після виконання Python-функції вирішує яку гілку DAG запускати далі (а решту пропускає).

**В Airflow:**
```python
def decide_branch(**kwargs):
    metrics = kwargs['ti'].xcom_pull(task_ids='evaluate')
    return 'register_model' if metrics['mAP50'] >= 0.35 else 'notify_failure'

branch_task = BranchPythonOperator(
    task_id='check_quality_gate',
    python_callable=decide_branch,
)
# Після цього — тільки одна гілка виконається
```

**В Prefect** — просто Python `if/else` у Flow:
```python
@flow
def pipeline():
    evaluation = evaluate_model(...)
    passed = check_quality_gate(evaluation)
    if passed:
        register_model(evaluation)   # ←виконається лише одна гілка
    else:
        notify_failure(evaluation)
```

**Сценарії MLOps де це потрібно:**
1. **Quality Gate**: якщо mAP50 ≥ поріг → реєструємо, інакше → сповіщення
2. **Data freshness**: якщо нові дані є → тренуємо, якщо старі → skip
3. **A/B порівняння**: якщо нова модель краща → деплоїмо, якщо ні → rollback

---

### 4. Multi-stage Docker build

**Безпека:**
- Перший (builder) stage містить `gcc`, `build-essential`, `curl` ← потенційні вразливості
- Фінальний (runtime) stage їх **не містить** → менша attack surface
- `.dockerignore` виключає `.env`, `.git`, `mlruns/` → секрети не потрапляють в образ

**Швидкість:**
- Docker кешує шари: якщо `pyproject.toml` не змінився → `uv sync` не перезапускається
- Фінальний образ ~600MB замість ~4GB → CI runner стартує швидше
- Docker Hub pull швидший → deployment швидший

**Аналогія для розуміння:** уяви що ти готуєш їжу. Тобі потрібні ножі, дошки, плита — але в тарілку ти кладеш лише готову страву, не весь кухонний інвентар. Multi-stage build — це "покласти в тарілку лише страву".

---

### 5. Чому важливо тестувати DAG у CI?

**Звичайні помилки в DAG/Flow файлах які CI ловить:**

| Тип помилки | Без тесту в CI | З `test_dag.py` в CI |
|-------------|---------------|---------------------|
| `ImportError: No module named 'prefect'` | Падає при запуску Flow в production | Виявляється одразу при PR |
| Синтаксична помилка у flow функції | Тихо ламає весь пайплайн | pytest повідомляє при запуску |
| Забули додати нову task до Flow | Логіка неповна, непомітно | Тест `test_all_tasks_importable` провалиться |
| PROJECT_ROOT вказує не туди | В CI env шлях інший | `test_project_root_defined` вловить |

**Наш `tests/test_dag.py`** перевіряє:
- Flow завантажується без помилок
- Всі 8 tasks (check_data, prepare, pretrain_tests, train, evaluate, gate, register, notify) присутні
- Flow приймає правильні параметри
- PROJECT_ROOT вказує на папку з `dvc.yaml`
- MAP50_THRESHOLD є float у [0, 1]

---

### 6. Архітектура Apache Airflow

```
┌─────────────────────────────────────────────────────┐
│              Apache Airflow (повний стек)            │
│                                                     │
│  ┌─────────────┐  Читає DAG файли  ┌─────────────┐ │
│  │  Scheduler  │ ────────────────→ │ Metadata DB │ │
│  │(планувальник│  Записує стан задач│ (PostgreSQL) │ │
│  │     )       │ ←──────────────── │             │ │
│  └─────────────┘                   └─────────────┘ │
│         │                                │          │
│         │ (відправляє задачі)            │          │
│         ▼                               │          │
│  ┌─────────────┐                        │          │
│  │  Executor   │    ┌─────────────┐     │          │
│  │  (CeleryEx) │───→│    Redis    │     │          │
│  │             │    │  (черга)    │     │          │
│  └─────────────┘    └─────────────┘     │          │
│                           │             │          │
│                    Worker1 Worker2       │          │
│                       (виконують tasks) │          │
│                                         │          │
│  ┌─────────────┐                        │          │
│  │  Webserver  │ ← читає статуси з ─────┘          │
│  │  (UI :8080) │   Metadata DB                     │
│  └─────────────┘                                   │
└─────────────────────────────────────────────────────┘
```

| Компонент | Роль простими словами |
|-----------|----------------------|
| **Scheduler** | "Годинникар" — дивиться на час, вирішує що і коли запускати |
| **Executor** | "Менеджер" — підбирає worker для виконання задачі |
| **Worker** | "Виконавець" — власне запускає твій Python-код |
| **Webserver** | "Дашборд" — красивий UI де видно що відбувається |
| **Metadata DB** | "Пам'ять" — зберігає стан всіх runs, логи, конфіги |
| **Redis** | "Диспетчер автобусів" — черга задач між Scheduler і Workers |

---

### 7. XComs в Airflow — передача даних між задачами

**XCom = Cross-Communication** — механізм передачі **маленьких** даних між Tasks в Airflow.

```python
# Task "evaluate" публікує результат
def evaluate_model(ti, **kwargs):
    result = {"mAP50": 0.45, "mAP50-95": 0.22}
    ti.xcom_push(key='metrics', value=result)  # зберегти в DB

# Task "check_gate" читає результат
def check_quality_gate(ti, **kwargs):
    metrics = ti.xcom_pull(task_ids='evaluate_model', key='metrics')
    return metrics['mAP50'] >= 0.35
```

**ВАЖНЕ**: XCom зберігається в **Metadata DB** (SQL база даних). Тому:

| Що передавати | Можна? |
|---------------|--------|
| `{"mAP50": 0.45}` (dict) | ✅ |
| `"path/to/model.pt"` (рядок) | ✅ |
| `model.pt` сам файл (сотні MB) | ❌ Переповнить DB |
| `numpy array (1000, 640, 640, 3)` | ❌ |

**Правило**: через XCom — лише **шляхи та метрики**. Самі файли — через DVC/MLflow/S3.

**В нашому Prefect**: взагалі не потрібно — функції просто повертають значення Python. `evaluate_model()` повертає `dict`, `check_quality_gate()` приймає цей dict як аргумент.

---

### 8. Sensor vs Task в Airflow

**Task** — "зроби щось і завершись" (успіхом або помилкою).

**Sensor** — "чекай поки щось станеться, опитуючи кожні N секунд".

```python
# Task: запустити і чекати результату
train_task = PythonOperator(task_id='train', python_callable=train)

# Sensor: чекати поки з'явиться файл
wait_sensor = FileSensor(
    task_id='wait_for_new_data',
    filepath='/data/new_batch.tar.gz',
    poke_interval=3600,  # перевіряти кожну годину
    timeout=86400,       # максимум 24 год чекання
)
```

**Коли Sensor потрібен у MLOps для YOLO Pose:**
- Чекати поки DVC-remote оновиться новими зображеннями бджіл
- Чекати поки гарантований час кожний понеділок після 8:00
- Чекати поки попередній пайплайн зафіксував всі артефакти в MLflow

**В нашому Prefect**: `check_data_freshness` виконує роль Sensor — перевіряє наявність даних. Але замість нескінченного polling — просто одноразова перевірка (спрощення для навчального проєкту).

---

### 9. Ідемпотентність задач

**Ідемпотентна задача** = результат однаковий незалежно від того, скільки разів запустити.

**Чому це питання для оркестраторів:**
- Scheduler може задати задачу двічі (race condition, мережевий збій)
- `retries=3` → задача виконається 4 рази (1 спроба + 3 retry)
- Backfill запускає всі минулі задачі заново

**НЕ ідемпотентно (проблема):**
```python
def prepare_data():
    shutil.copy("raw/data", "processed/")  # якщо папка вже є — Exception!
    # або
    db.insert(record)  # дублікати в БД при retry
```

**Ідемпотентно (правильно):**
```python
def prepare_data():
    shutil.copytree("raw/data", "processed/", dirs_exist_ok=True)  # OK якщо є
    # або
    db.upsert(record)  # INSERT OR UPDATE — без дублікатів
```

**В нашому проєкті:**
- `prepare_data()` перевіряє `needs_prepare` → якщо дані є, не запускає повторно
- `model.train(exist_ok=True)` → перезаписує, не падає якщо папка є
- `baseline/metrics.json` — просто перезаписується (`mode='w'`)

---

### 10. Infrastructure as Code в MLOps

**IaC** = вся інфраструктура описана у файлах у git, а не налаштована вручну.

**Чому важливо**: якщо твій сервер помре — ти можеш відновити всю систему командою, а не витрачати тиждень на ручне налаштування.

**Рівні IaC в нашому проєкті:**

| Що | Файл | Команда відновлення |
|----|------|---------------------|
| Python deps | `pyproject.toml` / `uv.lock` | `uv sync` |
| Дані | `dvc.lock` / `.dvc/config` | `dvc pull` |
| CI пайплайн | `.github/workflows/ci.yaml` | автоматично у GitHub |
| CT пайплайн | `dags/yolo_training_pipeline.py` | `python dags/...` |
| Контейнер | `Dockerfile` | `docker build .` |
| Конфіги | `config/*.yaml`, `params.yaml` | `git clone` |

**Практичний тест IaC**: чи може новий член команди відтворити всю систему? Команди:
```bash
git clone ... && cd mlops-project && uv sync && dvc pull
python dags/yolo_training_pipeline.py  # і все запрацювало
```

---

### 11. Build Artifact vs Model Artifact

| | Build Artifact | Model Artifact |
|---|---|---|
| **Що це** | Docker image, compiled binary | Навчена модель + метрики |
| **Коли створюється** | CI/CD при збірці коду | Після тренування (CT pipeline) |
| **Де зберігається** | Docker Hub, GitHub Packages | MLflow Registry, DVC remote |
| **Приклад для нас** | `yolo-pose:ci-abc123` | `best.pt`, `metrics.json` |
| **Версіонування** | Git SHA tag | MLflow run_id, Registry stage |
| **Термін зберігання** | Кілька версій (для rollback) | Довго (аудит) |

**Зв'язок між ними**: Build Artifact (Docker image) служить **середовищем** для CT Pipeline, який виробляє Model Artifact.

```
Docker image (Build Artifact)
         ↓
  docker run ...
         ↓
  train_model()  →  best.pt (Model Artifact)
                 →  metrics.json (Model Artifact)
```

---

### 12. Обробка помилок в Airflow

```python
dag = DAG(
    dag_id='yolo_training',
    default_args={
        'retries': 3,               # спробувати 3 рази перед провалом
        'retry_delay': timedelta(minutes=5),
        'email_on_failure': True,
        'email': ['team@lab.com'],
    },
    sla_miss_callback=send_slack_alert,  # якщо зайняло більше ніж SLA
)

task = PythonOperator(
    task_id='train_model',
    python_callable=train,
    execution_timeout=timedelta(hours=4),   # вбити процес через 4 год
    trigger_rule='all_success',             # запускати лише якщо всі попередні OK
)
```

**В нашому Prefect:**
```python
@task(retries=2, retry_delay_seconds=60)  # аналог retries + retry_delay
def train_model():
    ...

@flow(timeout_seconds=3600)  # аналог execution_timeout (1 год)
def yolo_pipeline():
    ...
```

---

### 13. Backfill в Airflow

**Backfill** = "наздогін" — повторне виконання DAG за минулі дати.

**Сценарій**: твій DAG запускається щотижня. Але цього тижня сервер не працював 3 дні. Тепер є Gap у 3 тижні. Backfill "заповнює" пропущені запуски.

```bash
airflow dags backfill \
    --start-date 2026-03-01 \
    --end-date 2026-03-21 \
    yolo_training_pipeline
# Запустить DAG для кожного тижня між цими датами
```

**Коли потрібен у ML:**
- Виправили баг у `prepare_data` → потрібно перепроцесити всі старі батчі
- Додали нову метрику → порахувати її для всіх минулих runs
- Сервер не працював → запустити пропущені тренування

**КРИТИЧНО**: задачі мають бути ідемпотентними, інакше backfill додасть дублікати у БД/MLflow.

---

### 14. Безпека секретів у CI/CD та оркестраторах

**Два основних правила:**

**1. Секрети не в git-репозиторії**
```yaml
# ❌ ПОГАНО (буде у git history назавжди)
env:
  DAGSHUB_TOKEN: "abc123secrettoken"

# ✅ ДОБРЕ (зашифровано в GitHub Secrets)
env:
  DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
```

**2. Секрети не в Docker образі**
```dockerfile
# ❌ ПОГАНО
COPY .env .  # .env з токенами потрапить в образ

# ✅ ДОБРЕ
# .env в .dockerignore
# токени — через docker run -e DAGSHUB_TOKEN=$TOKEN
```

**Практика в нашому проєкті:**
| Де | Як зберігаємо секрет |
|-----|---------------------|
| GitHub Actions | `${{ secrets.DAGSHUB_USER }}` / `${{ secrets.DAGSHUB_TOKEN }}` |
| Локально | `.env` файл (у `.gitignore`) |
| Prefect | `os.getenv("DAGSHUB_TOKEN")` з `.env` |
| Docker | `docker run -e DAGSHUB_TOKEN=...` або через .env при запуску |
