"""
Тренування YOLO Pose на датасеті бджіл з MLflow трекінгом.

DVC stage або ручний запуск:
    dvc repro                                    # через пайплайн
    python src/train.py                          # з params.yaml
    python src/train.py --epochs 100 --lr0 0.005 # з CLI override
"""

import argparse
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

    # Поепохові метрики — batch замість 1000+ окремих запитів
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

    # Артефакти
    artifact_names = [
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "results.png", "results.csv",
        "R_curve.png", "P_curve.png", "PR_curve.png", "F1_curve.png",
        "val_batch0_pred.jpg", "val_batch1_pred.jpg",
    ]
    for name in artifact_names:
        path = run_dir / name
        if path.exists():
            mlflow.log_artifact(str(path))

    # Модель
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        mlflow.log_artifact(str(best_pt))


def load_train_params() -> dict:
    """Завантажує параметри тренування з params.yaml."""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)["train"]


def parse_args() -> argparse.Namespace:
    defaults = load_train_params()
    p = argparse.ArgumentParser(
        description="YOLO Pose — bee pose estimation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--batch", type=int, default=defaults["batch"])
    p.add_argument("--imgsz", type=int, default=defaults["imgsz"])
    p.add_argument("--lr0", type=float, default=defaults["lr0"],
                   help="Initial learning rate")
    p.add_argument("--patience", type=int, default=defaults["patience"],
                   help="Early stopping: зупинити якщо немає покращення N епох")
    p.add_argument("--optimizer", default=defaults["optimizer"],
                   choices=["auto", "SGD", "Adam", "AdamW"],
                   help="auto = YOLO обере оптимальний варіант")
    p.add_argument("--model", default=defaults["model"],
                   help="Модель (YOLO завантажить автоматично якщо не знайде)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    settings.update({"mlflow": False, "wandb": False})
    setup_mlflow(EXPERIMENT_NAME)

    # Split має бути готовий (stage prepare)
    split_dir = PROJECT_ROOT / "data" / "split"
    dataset_yaml = split_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml} не знайдено — спочатку запусти prepare stage "
            "(dvc repro або python src/prepare.py)"
        )

    counts = {
        s: len(list((split_dir / s / "images").glob("*.jpg")))
        for s in ("train", "val")
    }

    # Прочитати val_hives з params.yaml для логування
    with open(PROJECT_ROOT / "params.yaml") as f:
        val_hives = yaml.safe_load(f)["prepare"]["val_hives"]

    run_name = f"e{args.epochs}-b{args.batch}-lr{args.lr0}-{args.optimizer}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "lr0": args.lr0,
            "patience": args.patience,
            "optimizer": args.optimizer,
            "model": args.model,
            "dataset": "bee-pose-2kpt",
            "val_hives": ", ".join(val_hives),
            "train_images": counts["train"],
            "val_images": counts["val"],
        })

        env = "colab" if os.environ.get("COLAB_RELEASE_TAG") else "local"
        mlflow.set_tag("environment", env)

        print(f"\n{'='*50}")
        print(f"  {run_name}")
        print(f"  epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}")
        print(f"  lr0={args.lr0}  patience={args.patience}  optimizer={args.optimizer}")
        print(f"{'='*50}\n")

        model = YOLO(args.model)
        results = model.train(
            data=str(dataset_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            patience=args.patience,
            optimizer=args.optimizer,
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
