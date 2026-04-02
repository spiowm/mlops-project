# ── Stage 1: Builder ──────────────────────────────────────────────────────
# Встановлює всі важкі Python-залежності.
# Цей шар кешується Docker — якщо pyproject.toml не змінився,
# pip install не перезапускається.
FROM python:3.12-slim AS builder

# Системні залежності для компіляції C-extensions (потрібні деяким пакетам)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо uv — швидкий менеджер пакетів (замість pip)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Копіюємо файли залежностей ПЕРЕД кодом (кеш-оптимізація)
COPY pyproject.toml uv.lock* ./

# Встановлюємо залежності в ізольоване venv
RUN uv sync --frozen --no-cache

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
# Фінальний легкий образ — тільки те що потрібно для запуску.
# НЕ містить: build-essential, git, curl, кеш pip
FROM python:3.12-slim AS runtime

# Мінімальні системні залежності для runtime
# libgl1 потрібен для OpenCV (ultralytics використовує cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо встановлене venv з builder stage
COPY --from=builder /app/.venv /app/.venv

# Копіюємо код проєкту
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY dags/ ./dags/
COPY config/ ./config/
COPY params.yaml ./
COPY pyproject.toml ./
COPY baseline/ ./baseline/

# Активуємо venv через PATH
ENV PATH="/app/.venv/bin:$PATH"

# Налаштування для ultralytics (вимикаємо MLflow/wandb інтеграцію)
ENV YOLO_CONFIG_DIR="/tmp/yolo_config"

# Типова команда: запуск CI training
# Перевизначається при docker run:
#   docker run yolo-pose python dags/yolo_training_pipeline.py
CMD ["python", "scripts/ci_train.py"]

# ── Метадані ────────────────────────────────────────────────────────────
LABEL maintainer="mlops-course"
LABEL description="YOLO11 Pose Estimation — CI/CD + CT Pipeline"
LABEL version="1.0"
