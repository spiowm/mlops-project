"""
Легковагове CI-тренування YOLO Pose для GitHub Actions.

Мінімальні параметри (1 епоха, imgsz=160, CPU) —
мета: перевірити що pipeline запускається без помилок.

Зберігає:
  - metrics.json (для Quality Gate тестів)
  - confusion_matrix.png (для CML звіту, якщо є)
"""

import json
import os
import sys
from pathlib import Path

# Додаємо src/ до sys.path для імпорту config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ultralytics import YOLO, settings  # noqa: E402


def main():
    settings.update({"mlflow": False, "wandb": False})

    # Шляхи
    dataset_yaml = PROJECT_ROOT / "data" / "split" / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"[ci_train] dataset.yaml не знайдено: {dataset_yaml}")
        print("[ci_train] Запустіть спочатку: python src/prepare.py")
        sys.exit(1)

    # CI параметри — мінімальні для CPU runner
    epochs = int(os.getenv("CI_EPOCHS", "1"))
    imgsz = int(os.getenv("CI_IMGSZ", "160"))
    batch = int(os.getenv("CI_BATCH", "4"))

    # Модель — намагаємось знайти локально, інакше YOLO завантажить
    model_name = os.getenv("CI_MODEL", "yolo11n-pose.pt")
    model_path = PROJECT_ROOT / "models" / "pretrained" / model_name
    model_str = str(model_path) if model_path.exists() else model_name

    print(f"[ci_train] epochs={epochs}, imgsz={imgsz}, batch={batch}")
    print(f"[ci_train] model={model_str}")
    print(f"[ci_train] data={dataset_yaml}")

    model = YOLO(model_str)

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=1,
        optimizer="auto",
        device="cpu",
        project=str(PROJECT_ROOT / "runs" / "ci"),
        name="ci_train",
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # Зберегти metrics.json для Quality Gate
    metrics = {}
    for key, value in results.results_dict.items():
        # Нормалізація ключів
        clean_key = key.strip()
        try:
            metrics[clean_key] = float(value)
        except (TypeError, ValueError):
            pass

    # Виокремити ключові метрики зі стандартних ключів YOLO
    summary = {}
    for k, v in metrics.items():
        if "mAP50-95" in k:
            summary["mAP50-95"] = v
        elif "mAP50" in k and "95" not in k:
            summary["mAP50"] = v
        elif "precision" in k.lower() and "mAP" not in k:
            summary["precision"] = v
        elif "recall" in k.lower() and "mAP" not in k:
            summary["recall"] = v

    # Додати всі raw метрики теж
    summary["_raw"] = metrics

    metrics_path = PROJECT_ROOT / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[ci_train] Метрики збережено: {metrics_path}")

    # Копіювати confusion_matrix якщо є
    run_dir = Path(results.save_dir)
    for img_name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        cm_src = run_dir / img_name
        if cm_src.exists():
            import shutil
            cm_dst = PROJECT_ROOT / img_name
            shutil.copy2(cm_src, cm_dst)
            print(f"[ci_train] Скопійовано: {cm_dst}")

    print("[ci_train] ✓ Завершено")


if __name__ == "__main__":
    main()
