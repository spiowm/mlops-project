"""
Continuous Training Pipeline для YOLO11 Pose з Prefect.

Реалізує ідею ЛР5: оркестрація ML-пайплайну як DAG, де кожен крок —
окрема задача з визначеними залежностями, retry та моніторингом.

Структура Flow:
    check_data_freshness → prepare_data → run_pretrain_tests
        → train_model → evaluate_model → check_quality_gate
        → register_model (якщо пройшов) | notify_failure (якщо ні)

Запуск:
    # Один раз
    uv run python dags/yolo_training_pipeline.py

    # Із розкладом (щотижня) — запустіть prefect server спочатку:
    uv run prefect server start
    uv run python dags/yolo_training_pipeline.py  # задеплоїть schedule
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Prefect
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

# ── Шляхи ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Конфігурація ────────────────────────────────────────────────────────────

METRICS_PATH = PROJECT_ROOT / "metrics.json"
BASELINE_PATH = PROJECT_ROOT / "baseline" / "metrics.json"
MAP50_THRESHOLD = float(os.getenv("CT_MAP50_THRESHOLD", "0.20"))
CI_EPOCHS = int(os.getenv("CT_EPOCHS", "1"))
CI_IMGSZ = int(os.getenv("CT_IMGSZ", "320"))


# ── Tasks ────────────────────────────────────────────────────────────────────


@task(name="Check Data Freshness", retries=2, retry_delay_seconds=30)
def check_data_freshness() -> dict:
    """
    Перевіряє наявність та свіжість даних.
    Додатково перевіряє, чи шляхи в dataset.yaml актуальні
    для поточного середовища (хост vs Docker).
    """
    logger = get_run_logger()

    split_dir = PROJECT_ROOT / "data" / "split"
    dataset_yaml = split_dir / "dataset.yaml"

    result = {
        "split_dir_exists": split_dir.exists(),
        "dataset_yaml_exists": dataset_yaml.exists(),
        "needs_prepare": not dataset_yaml.exists(),
    }

    if split_dir.exists():
        train_images = list((split_dir / "train" / "images").glob("*.jpg"))
        val_images = list((split_dir / "val" / "images").glob("*.jpg"))
        result["train_images"] = len(train_images)
        result["val_images"] = len(val_images)
        logger.info(
            f"Дані знайдені: {len(train_images)} train, {len(val_images)} val"
        )
    else:
        logger.warning(f"data/split/ не знайдено: {split_dir}")
        result["train_images"] = 0
        result["val_images"] = 0

    # Перевіряємо чи paths у dataset.yaml актуальні для цього середовища
    # (критично для Docker: yaml може містити absolute paths від хост-машини)
    if dataset_yaml.exists() and not result["needs_prepare"]:
        try:
            import yaml as _yaml
            with open(dataset_yaml) as f:
                ds = _yaml.safe_load(f)
            yaml_root = ds.get("path", "")
            if yaml_root and not Path(yaml_root).exists():
                logger.warning(
                    f"dataset.yaml містить недоступний шлях: '{yaml_root}'. "
                    "Це absolute path від іншої машини (наприклад хоста). "
                    "Запустимо prepare.py щоб перегенерувати з правильними шляхами."
                )
                result["needs_prepare"] = True
        except Exception as e:
            logger.warning(f"Не вдалось перевірити dataset.yaml paths: {e}")

    return result


@task(name="Prepare Data", retries=1)
def prepare_data(data_status: dict) -> bool:
    """Запускає src/prepare.py якщо потрібно."""
    logger = get_run_logger()

    if not data_status["needs_prepare"] and data_status["train_images"] >= 20:
        logger.info("Дані вже підготовлені, пропускаємо prepare.py")
        return True

    logger.info("Запуск src/prepare.py...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "prepare.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"prepare.py завершився з помилкою:\n{result.stderr}")
        raise RuntimeError(f"Prepare failed: {result.stderr[:500]}")

    logger.info("prepare.py ✓")
    return True


@task(name="Run Pre-train Tests", retries=0)
def run_pretrain_tests() -> bool:
    """
    Запускає pytest для валідації даних і конфігурації.
    Аналог 'pre-train-tests' job з ci.yaml.
    """
    logger = get_run_logger()
    logger.info("Запуск pre-train тестів...")

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_data.py", "tests/test_config.py",
            "-v", "--tb=short", "-q",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    if result.returncode != 0:
        logger.error(f"Pre-train тести FAILED:\n{result.stderr}")
        raise RuntimeError("Pre-train tests failed — pipeline зупинено")

    logger.info("Pre-train тести ✓")
    return True


@task(name="Train Model", retries=1, retry_delay_seconds=60)
def train_model() -> Path:
    """
    Запускає CI-тренування YOLO11 Pose.
    В реальному CT pipeline — повне тренування з src/train.py.
    """
    logger = get_run_logger()
    logger.info(f"Запуск тренування: epochs={CI_EPOCHS}, imgsz={CI_IMGSZ}")

    env = {
        **os.environ,
        "CI_EPOCHS": str(CI_EPOCHS),
        "CI_IMGSZ": str(CI_IMGSZ),
        "CI_BATCH": "4",
    }

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "ci_train.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    logger.info(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)

    if result.returncode != 0:
        logger.error(f"Тренування FAILED:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"Training failed: {result.stderr[-500:]}")

    if not METRICS_PATH.exists():
        raise RuntimeError(f"metrics.json не створено тренуванням: {METRICS_PATH}")

    logger.info(f"Тренування ✓ → {METRICS_PATH}")
    return METRICS_PATH


@task(name="Evaluate Model (Quality Gate)")
def evaluate_model(metrics_path: Path) -> dict:
    """
    Читає metrics.json і запускає pytest Quality Gate тести.
    Повертає словник з метриками.
    """
    logger = get_run_logger()

    # Читаємо метрики
    with open(metrics_path) as f:
        metrics = json.load(f)

    map50 = metrics.get("mAP50", 0.0)
    map50_95 = metrics.get("mAP50-95", 0.0)

    logger.info(f"Метрики: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")

    # Запускаємо pytest Quality Gate
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_artifacts.py",
            "-v", "--tb=short", "-q",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "MAP50_THRESHOLD": "0.0", "MAP50_95_THRESHOLD": "0.0"},
    )

    logger.info(result.stdout[-1000:])
    tests_passed = result.returncode == 0

    return {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "tests_passed": tests_passed,
    }


@task(name="Check Quality Gate")
def check_quality_gate(evaluation: dict) -> bool:
    """
    Реалізує логіку BranchPythonOperator з Airflow.
    Повертає True якщо модель пройшла, False — якщо ні.
    """
    logger = get_run_logger()

    map50 = evaluation["mAP50"]
    tests_passed = evaluation["tests_passed"]
    passed = map50 >= MAP50_THRESHOLD and tests_passed

    if passed:
        logger.info(
            f"✅ Quality Gate PASSED: mAP50={map50:.4f} ≥ {MAP50_THRESHOLD}"
        )
    else:
        logger.warning(
            f"❌ Quality Gate FAILED: mAP50={map50:.4f} < {MAP50_THRESHOLD} "
            f"або tests_passed={tests_passed}"
        )

    return passed


@task(name="Register Model")
def register_model(evaluation: dict) -> str:
    """
    Реєструє модель: оновлює baseline та публікує звіт.
    В production — реєстрація в MLflow Model Registry.
    """
    logger = get_run_logger()

    # Оновлюємо baseline
    baseline_data = {
        "mAP50": evaluation["mAP50"],
        "mAP50-95": evaluation["mAP50-95"],
        "precision": evaluation.get("precision", 0.0),
        "recall": evaluation.get("recall", 0.0),
    }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Читаємо старий baseline для звіту
    old_baseline = {}
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            old_baseline = json.load(f)

    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline_data, f, indent=2)

    logger.info(f"Baseline оновлено: {BASELINE_PATH}")

    # Формуємо Markdown звіт для Prefect Artifacts
    delta_str = ""
    if old_baseline:
        old_map50 = old_baseline.get("mAP50", 0)
        delta = evaluation["mAP50"] - old_map50
        emoji = "🟢" if delta >= 0 else "🔴"
        delta_str = f"\n\n**Δ vs попередній baseline**: {emoji} {delta:+.4f}"

    report_md = f"""## ✅ Model Registered

| Metric | Value |
|--------|-------|
| mAP50 | {evaluation['mAP50']:.4f} |
| mAP50-95 | {evaluation['mAP50-95']:.4f} |
| precision | {evaluation.get('precision', 0):.4f} |
| recall | {evaluation.get('recall', 0):.4f} |

**Quality Gate**: mAP50 ≥ {MAP50_THRESHOLD} → ✅ PASSED{delta_str}

Baseline оновлено: `{BASELINE_PATH.relative_to(PROJECT_ROOT)}`
"""

    create_markdown_artifact(
        markdown=report_md,
        key="model-registration-report",
        description="CT Pipeline — Model registered",
    )

    logger.info("✅ Модель зареєстрована (baseline оновлено)")
    return f"Registered: mAP50={evaluation['mAP50']:.4f}"


@task(name="Notify Failure")
def notify_failure(evaluation: dict) -> str:
    """
    Дія при невдалому Quality Gate.
    В production: надіслати email/Slack/Teams повідомлення.
    """
    logger = get_run_logger()

    report_md = f"""## ❌ Quality Gate FAILED

| Metric | Value | Threshold |
|--------|-------|-----------|
| mAP50 | {evaluation['mAP50']:.4f} | {MAP50_THRESHOLD:.4f} |
| tests_passed | {evaluation['tests_passed']} | True |

**Дія**: Baseline НЕ оновлено. Попередня модель залишається активною.

Перегляньте логи тренування для виявлення причини деградації.
"""

    create_markdown_artifact(
        markdown=report_md,
        key="quality-gate-failure-report",
        description="CT Pipeline — Quality Gate Failed",
    )

    logger.warning(
        f"❌ Notify: mAP50={evaluation['mAP50']:.4f} < threshold={MAP50_THRESHOLD}"
    )
    return f"Failed: mAP50={evaluation['mAP50']:.4f}"


# ── Flow ──────────────────────────────────────────────────────────────────────


@flow(
    name="YOLO Pose CT Pipeline",
    description=(
        "Continuous Training pipeline для YOLO11 Pose. "
        "Виконує: валідацію даних → тренування → Quality Gate → реєстрацію."
    ),
    timeout_seconds=3600,  # максимум 1 година
)
def yolo_pose_training_pipeline(
    ct_epochs: int = CI_EPOCHS,
    ct_imgsz: int = CI_IMGSZ,
    map50_threshold: float = MAP50_THRESHOLD,
) -> dict:
    """
    Головний Flow — оркеструє весь ML-пайплайн.

    Args:
        ct_epochs: Кількість епох тренування
        ct_imgsz: Розмір зображення для тренування
        map50_threshold: Мінімальний mAP50 для реєстрації моделі

    Returns:
        dict з результатами пайплайну
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("  YOLO Pose CT Pipeline")
    logger.info(f"  epochs={ct_epochs}, imgsz={ct_imgsz}, threshold={map50_threshold}")
    logger.info("=" * 60)

    # Оновлюємо env vars з параметрів flow (для subprocess)
    os.environ["CT_EPOCHS"] = str(ct_epochs)
    os.environ["CT_IMGSZ"] = str(ct_imgsz)
    os.environ["CT_MAP50_THRESHOLD"] = str(map50_threshold)

    # ── Крок 1: Перевірка даних ──────────────────────────────────────────
    data_status = check_data_freshness()

    # ── Крок 2: Підготовка даних ─────────────────────────────────────────
    prepare_data(data_status)

    # ── Крок 3: Pre-train тести ──────────────────────────────────────────
    run_pretrain_tests()

    # ── Крок 4: Тренування ───────────────────────────────────────────────
    metrics_path = train_model()

    # ── Крок 5: Оцінка (Quality Gate тести) ─────────────────────────────
    evaluation = evaluate_model(metrics_path)

    # ── Крок 6: Розгалуження — аналог BranchPythonOperator ───────────────
    gate_passed = check_quality_gate(evaluation)

    # ── Крок 7: Реєстрація або сповіщення про збій ───────────────────────
    if gate_passed:
        outcome = register_model(evaluation)
        status = "success"
    else:
        outcome = notify_failure(evaluation)
        status = "quality_gate_failed"

    result = {
        "status": status,
        "mAP50": evaluation["mAP50"],
        "mAP50-95": evaluation["mAP50-95"],
        "gate_passed": gate_passed,
        "outcome": outcome,
    }

    logger.info(f"\nPipeline завершено: {result}")
    return result


# ── Entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    # Завантажуємо дефолтні параметри з ENV
    default_params = {
        "ct_epochs": int(os.getenv("CT_EPOCHS", "1")),
        "ct_imgsz": int(os.getenv("CT_IMGSZ", "320")),
        "map50_threshold": float(os.getenv("CT_MAP50_THRESHOLD", "0.0")),
    }

    if "--serve" in sys.argv:
        # ЗАПУСК ЧЕРЕЗ UI (Джей Деплоймент)
        # Команда: uv run python dags/yolo_training_pipeline.py --serve
        print("🚀 Запуск у режимі Deployment. Перейди на http://localhost:4200 (Deployments -> Quick Run)")
        yolo_pose_training_pipeline.serve(
            name="ct-pipeline-ui-deployment",
            tags=["yolo", "ct"],
            parameters=default_params
        )
    else:
        # ПРЯМИЙ ЗАПУСК (CI/CD, Docker, локальні тести)
        # Команда: uv run python dags/yolo_training_pipeline.py
        result = yolo_pose_training_pipeline(**default_params)
        print(f"\nРезультат: {result}")
