"""
Тренування YOLO Pose на датасеті бджіл з MLflow трекінгом.

Запуск:
    dvc repro train                  # через DVC пайплайн
    python src/train.py              # читає параметри з params.yaml
"""

from math import degrees
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.entities import Metric
from ultralytics import YOLO, settings

from config import PROJECT_ROOT, dagshub_config

EXPERIMENT_NAME = "bee-pose-estimation"
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"


def setup_mlflow(experiment_name: str) -> None:
    """Налаштовує MLflow: завжди локально, додатково DagsHub якщо є credentials."""
    # Локальний tracking — завжди
    local_uri = str(PROJECT_ROOT / "mlruns")
    mlflow.set_tracking_uri(local_uri)
    print(f"[mlflow] Локальний: {local_uri}")

    # DagsHub — додатково, якщо задані credentials
    token, user, repo = dagshub_config()
    if token and user and repo:
        uri = f"https://dagshub.com/{user}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(uri)
        print(f"[mlflow] DagsHub: {uri}")

    mlflow.set_experiment(experiment_name)


def log_results(results, run_dir: Path) -> None:
    """Логує метрики та артефакти в MLflow."""
    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id

    # Фінальні метрики
    final = {}
    for key, value in results.results_dict.items():
        safe = key.replace("(", "").replace(")", "").replace("/", "_")
        final[safe] = value
    mlflow.log_metrics(final)

    # Поепохові метрики
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        metrics_batch = []
        ts = int(time.time() * 1000)
        for col in df.columns:
            if col == "epoch":
                continue
            safe = col.replace("(", "").replace(")", "").replace("/", "_")
            for step, val in enumerate(df[col]):
                metrics_batch.append(Metric(safe, float(val), ts, step))

        # log_batch приймає до 1000 метрик за раз
        for i in range(0, len(metrics_batch), 1000):
            client.log_batch(run_id, metrics=metrics_batch[i:i + 1000])

    # Всі артефакти з run_dir (графіки, матриці, ваги)
    mlflow.log_artifacts(str(run_dir))


def main() -> None:
    with open(PROJECT_ROOT / "params.yaml") as f:
        all_params = yaml.safe_load(f)
        
    train_params = all_params.get("train", {})
    epochs = train_params.get("epochs", 50)
    batch = train_params.get("batch", 16)
    imgsz = train_params.get("imgsz", 1280)
    lr0 = train_params.get("lr0", 0.01)
    patience = train_params.get("patience", 20)
    optimizer = train_params.get("optimizer", "auto")
    model_name = train_params.get("model", "yolo11n-pose.pt")
    
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    settings.update({"mlflow": False, "wandb": False, "weights_dir": str(PRETRAINED_DIR)})
    setup_mlflow(EXPERIMENT_NAME)

    # Split має бути готовий (stage prepare)
    dataset = all_params.get("prepare", {}).get("dataset", "pose")

    split_dir = PROJECT_ROOT / "data" / "split" / dataset
    dataset_yaml = split_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml} не знайдено — спочатку запусти prepare stage "
            "(dvc repro prepare або python src/prepare.py)"
        )

    counts = {
        s: len(list((split_dir / s / "images").glob("*.jpg")))
        for s in ("train", "val")
    }

    # Прочитати val_hives з params.yaml для логування
    val_hives = all_params["prepare"]["val_hives"]

    # Хеш датасету з .dvc файлу — зв'язок MLflow run ↔ версія даних
    dvc_file = PROJECT_ROOT / "data" / "raw" / f"{dataset}.dvc"
    if dvc_file.exists():
        with open(dvc_file) as f:
            dvc_meta = yaml.safe_load(f)
        dataset_hash = dvc_meta["outs"][0]["md5"][:8]  # перші 8 символів
    else:
        dataset_hash = "unknown"

    run_name = f"e{epochs}-b{batch}-lr{lr0}-{optimizer}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch,
            "imgsz": imgsz,
            "lr0": lr0,
            "patience": patience,
            "optimizer": optimizer,
            "model": model_name,
            "dataset": dataset,
            "dataset_hash": dataset_hash,
            "val_hives": ", ".join(val_hives),
            "train_images": counts["train"],
            "val_images": counts["val"],
            # degrees: 180.0
        })

        env = "colab" if os.environ.get("COLAB_RELEASE_TAG") else "local"
        mlflow.set_tag("environment", env)

        print(f"\n{'='*50}")
        print(f"  {run_name}")
        print(f"  epochs={epochs}  batch={batch}  imgsz={imgsz}")
        print(f"  lr0={lr0}  patience={patience}  optimizer={optimizer}")
        print(f"{'='*50}\n")
        # Якщо задано просто ім'я файлу (не шлях) — шукаємо / завантажуємо в models/pretrained/
        model_path = Path(model_name)
        if not model_path.is_absolute() and not model_path.parent.parts:
            # Просто ім'я файлу (напр. yolo11n-pose.pt)
            PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
            pretrained_path = PRETRAINED_DIR / model_name
            # Передаємо повний шлях — YOLO завантажить в pretrained/ якщо файл не існує
            model_path = pretrained_path
        else:
            model_path = PROJECT_ROOT / model_path if not model_path.is_absolute() else model_path

        model = YOLO(str(model_path))
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            patience=patience,
            optimizer=optimizer,
            project=str(PROJECT_ROOT / "runs" / EXPERIMENT_NAME),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        run_dir = Path(results.save_dir)
        log_results(results, run_dir)
        print(f"\n[done] {run_name} — результати: {run_dir}")


if __name__ == "__main__":
    main()
