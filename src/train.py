"""
Тренування YOLO Pose на датасеті бджіл з MLflow трекінгом.

Приклади:
    python src/train.py --epochs 50 --batch 16
    python src/train.py --epochs 100 --optimizer SGD --lr0 0.005
"""

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from ultralytics import YOLO, settings

from config import PROJECT_ROOT, dagshub_config
from data import DEFAULT_VAL_HIVES, prepare_data

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
    # Фінальні метрики
    for key, value in results.results_dict.items():
        safe = key.replace("(", "").replace(")", "").replace("/", "_")
        mlflow.log_metric(safe, value)

    # Поепохові метрики
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if col == "epoch":
                continue
            safe = col.replace("(", "").replace(")", "").replace("/", "_")
            for step, val in enumerate(df[col]):
                mlflow.log_metric(safe, float(val), step=step)

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO Pose — bee pose estimation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping: зупинити якщо немає покращення N епох")
    p.add_argument("--optimizer", default="auto",
                   choices=["auto", "SGD", "Adam", "AdamW"],
                   help="auto = YOLO обере оптимальний варіант")
    p.add_argument("--model", default="yolo11n-pose.pt",
                   help="Модель (YOLO завантажить автоматично якщо не знайде)")
    p.add_argument("--force-split", action="store_true",
                   help="Перестворити train/val split")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    settings.update({"mlflow": False, "wandb": False})
    setup_mlflow(EXPERIMENT_NAME)

    dataset_yaml, counts = prepare_data(force=args.force_split)

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
            "val_hives": ", ".join(DEFAULT_VAL_HIVES),
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
