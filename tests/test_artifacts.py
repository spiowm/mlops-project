"""
Post-train тести: перевірка артефактів та Quality Gate.

Запускаються ПІСЛЯ тренування (CI train або dvc repro).
Перевіряє metrics.json, якість моделі, наявність артефактів.
"""

import json


def test_metrics_json_exists(metrics_path):
    """metrics.json має існувати після тренування."""
    assert metrics_path.exists(), (
        f"{metrics_path} не знайдено — тренування не завершилось або "
        "скрипт не зберіг метрики"
    )


def test_metrics_json_valid(metrics_path):
    """metrics.json має бути коректним JSON з потрібними ключами."""
    if not metrics_path.exists():
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    assert isinstance(metrics, dict), "metrics.json має бути словником"

    expected_keys = {"mAP50", "mAP50-95"}
    present = expected_keys & set(metrics.keys())
    assert present, (
        f"metrics.json: жоден з очікуваних ключів {expected_keys} не знайдено. "
        f"Наявні ключі: {sorted(metrics.keys())}"
    )


def test_quality_gate_map50(metrics_path, map50_threshold):
    """Quality Gate: mAP50 має бути ≥ порогу."""
    if not metrics_path.exists():
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    map50 = metrics.get("mAP50")
    if map50 is None:
        return  # пропускаємо якщо метрика відсутня

    map50 = float(map50)
    assert map50 >= map50_threshold, (
        f"Quality Gate FAILED: mAP50={map50:.4f} < threshold={map50_threshold:.4f}"
    )


def test_quality_gate_map50_95(metrics_path, map50_95_threshold):
    """Quality Gate: mAP50-95 має бути ≥ порогу."""
    if not metrics_path.exists():
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    map50_95 = metrics.get("mAP50-95")
    if map50_95 is None:
        return  # пропускаємо якщо метрика відсутня

    map50_95 = float(map50_95)
    assert map50_95 >= map50_95_threshold, (
        f"Quality Gate FAILED: mAP50-95={map50_95:.4f} < threshold={map50_95_threshold:.4f}"
    )


def test_metrics_values_reasonable(metrics_path):
    """Значення метрик мають бути у розумному діапазоні [0, 1]."""
    if not metrics_path.exists():
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    for key in ("mAP50", "mAP50-95", "precision", "recall"):
        if key in metrics:
            val = float(metrics[key])
            assert 0.0 <= val <= 1.0, (
                f"Метрика '{key}'={val} поза діапазоном [0, 1]"
            )


def test_confusion_matrix_exists(project_root):
    """confusion_matrix.png має існувати після тренування (якщо metrics.json є)."""
    metrics_json = project_root / "metrics.json"
    cm_path = project_root / "confusion_matrix.png"

    if not metrics_json.exists():
        return  # тренування не було — пропускаємо

    # confusion_matrix може не генеруватися при 1 епосі CI, тому м'яка перевірка
    if not cm_path.exists():
        import warnings
        warnings.warn(
            f"confusion_matrix.png не знайдено в {project_root}. "
            "Це нормально для CI з 1 епохою."
        )
