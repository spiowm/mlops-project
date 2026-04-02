"""
Тести для Prefect Flow (аналог DagBag тестів для Airflow).

Перевіряє:
- Коректне завантаження Flow без помилок імпорту
- Наявність усіх ключових tasks
- Базову структуру Flow

Запуск:
    uv run pytest tests/test_dag.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAGS_DIR = PROJECT_ROOT / "dags"

# Додаємо dags/ до sys.path для імпорту
sys.path.insert(0, str(DAGS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Тести завантаження Flow ──────────────────────────────────────────────────


def test_pipeline_module_importable():
    """DAG-файл має завантажуватися без помилок імпорту."""
    try:
        import yolo_training_pipeline  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Не вдалося завантажити yolo_training_pipeline: {e}")


def test_flow_function_exists():
    """Головна flow-функція має існувати та бути callable."""
    from yolo_training_pipeline import yolo_pose_training_pipeline

    assert callable(yolo_pose_training_pipeline), (
        "yolo_pose_training_pipeline має бути функцією"
    )


def test_flow_is_prefect_flow():
    """Перевіряє що функція декорована як Prefect @flow."""
    from yolo_training_pipeline import yolo_pose_training_pipeline

    # Prefect flow додає атрибут .fn або .name
    assert hasattr(yolo_pose_training_pipeline, "name"), (
        "yolo_pose_training_pipeline не є Prefect flow (немає .name)"
    )
    assert "YOLO" in yolo_pose_training_pipeline.name, (
        f"Назва flow '{yolo_pose_training_pipeline.name}' не містить 'YOLO'"
    )


def test_all_tasks_importable():
    """Всі ключові task-функції мають бути доступні."""
    required_tasks = [
        "check_data_freshness",
        "prepare_data",
        "run_pretrain_tests",
        "train_model",
        "evaluate_model",
        "check_quality_gate",
        "register_model",
        "notify_failure",
    ]

    import yolo_training_pipeline as pipeline

    missing = [t for t in required_tasks if not hasattr(pipeline, t)]
    assert not missing, f"Відсутні task-функції: {missing}"


def test_tasks_are_callable():
    """Кожна task-функція має бути callable."""
    import yolo_training_pipeline as pipeline

    tasks = [
        "check_data_freshness",
        "prepare_data",
        "run_pretrain_tests",
        "train_model",
        "evaluate_model",
        "check_quality_gate",
        "register_model",
        "notify_failure",
    ]

    for task_name in tasks:
        fn = getattr(pipeline, task_name)
        assert callable(fn), f"{task_name} не є callable"


def test_tasks_are_prefect_tasks():
    """Ключові tasks мають бути декоровані @task."""
    import yolo_training_pipeline as pipeline

    tasks_to_check = [
        "check_data_freshness",
        "train_model",
        "check_quality_gate",
        "register_model",
    ]

    for task_name in tasks_to_check:
        fn = getattr(pipeline, task_name)
        # Prefect task додає атрибут .name або .fn
        assert hasattr(fn, "name") or hasattr(fn, "fn"), (
            f"{task_name} не є Prefect @task"
        )


# ── Тести конфігурації ────────────────────────────────────────────────────────


def test_project_root_defined():
    """PROJECT_ROOT має вказувати на коректний корінь проєкту."""
    import yolo_training_pipeline as pipeline

    assert hasattr(pipeline, "PROJECT_ROOT"), "PROJECT_ROOT не визначено"
    root = pipeline.PROJECT_ROOT
    assert root.exists(), f"PROJECT_ROOT не існує: {root}"
    assert (root / "dvc.yaml").exists(), (
        f"dvc.yaml не знайдено в PROJECT_ROOT: {root}"
    )


def test_threshold_is_valid():
    """MAP50_THRESHOLD має бути числом у [0, 1]."""
    import yolo_training_pipeline as pipeline

    threshold = pipeline.MAP50_THRESHOLD
    assert isinstance(threshold, float), f"MAP50_THRESHOLD має бути float, не {type(threshold)}"
    assert 0.0 <= threshold <= 1.0, (
        f"MAP50_THRESHOLD={threshold} поза діапазоном [0, 1]"
    )


def test_paths_referenced_correctly():
    """METRICS_PATH та BASELINE_PATH мають вказувати в межах проєкту."""
    import yolo_training_pipeline as pipeline

    assert pipeline.METRICS_PATH.parent == pipeline.PROJECT_ROOT, (
        "METRICS_PATH має бути в PROJECT_ROOT"
    )
    assert pipeline.BASELINE_PATH.parent.parent == pipeline.PROJECT_ROOT, (
        "BASELINE_PATH має бути в PROJECT_ROOT/baseline/"
    )


# ── Тест структури Flow параметрів ───────────────────────────────────────────


def test_flow_accepts_parameters():
    """Flow має приймати параметри ct_epochs, ct_imgsz, map50_threshold."""
    import inspect

    from yolo_training_pipeline import yolo_pose_training_pipeline

    # Для Prefect 2.x/3.x перевіряємо .fn або сам об'єкт
    fn = getattr(yolo_pose_training_pipeline, "fn", yolo_pose_training_pipeline)
    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())

    expected_params = {"ct_epochs", "ct_imgsz", "map50_threshold"}
    missing = expected_params - params
    assert not missing, f"Flow не має параметрів: {missing}"


def test_dags_dir_exists():
    """dags/ директорія має існувати."""
    assert DAGS_DIR.exists(), f"dags/ директорія не знайдена: {DAGS_DIR}"


def test_pipeline_file_exists():
    """Файл pipeline має існувати."""
    pipeline_file = DAGS_DIR / "yolo_training_pipeline.py"
    assert pipeline_file.exists(), f"Pipeline файл не знайдено: {pipeline_file}"
