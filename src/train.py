"""
Тренування YOLO Pose на датасеті бджіл з MLflow трекінгом (DagsHub).

Приклади:
    python src/train.py --epochs 50 --batch 16
    python src/train.py --epochs 100 --optimizer SGD --lr0 0.005
    python src/train.py --epochs 5 --imgsz 320
"""

import argparse
from pathlib import Path

import mlflow
from ultralytics import YOLO, settings

from config import PROJECT_ROOT
from data import prepare_data
from tracking import setup_mlflow, log_yolo_results

EXPERIMENT_NAME = "bee-pose-estimation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO Pose — bee pose estimation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--optimizer", default="AdamW", choices=["SGD", "Adam", "AdamW"])
    p.add_argument("--model", default="yolo11n-pose.pt",
                   help="Модель (YOLO завантажить автоматично якщо не знайде)")
    p.add_argument("--force-split", action="store_true",
                   help="Перестворити train/val split")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Вимикаємо вбудовані callback-и ultralytics (логуємо самі)
    settings.update({"mlflow": False, "wandb": False})

    # MLflow (DagsHub або локальний)
    setup_mlflow(EXPERIMENT_NAME)

    # Підготовка даних (split по вуликах)
    dataset_yaml = prepare_data(force=args.force_split)

    run_name = f"yolo-e{args.epochs}-b{args.batch}-lr{args.lr0}-{args.optimizer}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "YOLO-Pose")
        mlflow.set_tag("optimizer", args.optimizer)

        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "lr0": args.lr0,
            "optimizer": args.optimizer,
            "model": Path(args.model).name,
        })

        print(f"\n{'='*50}")
        print(f"  {run_name}")
        print(f"  epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}")
        print(f"  lr0={args.lr0}  optimizer={args.optimizer}")
        print(f"{'='*50}\n")

        model = YOLO(args.model)
        results = model.train(
            data=str(dataset_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            optimizer=args.optimizer,
            project=str(PROJECT_ROOT / "runs" / "pose"),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        run_dir = Path(results.save_dir)
        log_yolo_results(results, run_dir)
        print(f"\n[done] {run_name} — результати: {run_dir}")


if __name__ == "__main__":
    main()
