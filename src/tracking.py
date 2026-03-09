"""MLflow інтеграція (DagsHub) для логування результатів YOLO тренування."""

import os
import shutil
from pathlib import Path

import mlflow
import pandas as pd

from config import PROJECT_ROOT, dagshub_config


def setup_mlflow(experiment_name: str) -> None:
    """
    Налаштовує MLflow tracking.
    Якщо задані DAGSHUB_TOKEN + DAGSHUB_USER + DAGSHUB_REPO — використовує DagsHub.
    Інакше — локальний mlruns/.
    """
    token, user, repo = dagshub_config()

    if token and user and repo:
        uri = f"https://dagshub.com/{user}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(uri)
        print(f"[mlflow] DagsHub: {uri}")
    else:
        local_uri = str(PROJECT_ROOT / "mlruns")
        mlflow.set_tracking_uri(local_uri)
        print(f"[mlflow] Локальний: {local_uri}")

    mlflow.set_experiment(experiment_name)


def log_yolo_results(results, run_dir: Path) -> None:
    """Логує фінальні та поепохові метрики + артефакти в MLflow."""

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

    # Артефакти: графіки, CSV
    artifacts = [
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "results.png", "results.csv",
        "R_curve.png", "P_curve.png", "PR_curve.png", "F1_curve.png",
        "val_batch0_pred.jpg", "val_batch1_pred.jpg",
    ]
    for name in artifacts:
        path = run_dir / name
        if path.exists():
            mlflow.log_artifact(str(path))

    # Найкраща модель
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        mlflow.log_artifact(str(best_pt), artifact_path="model")
        dst = PROJECT_ROOT / "models" / "best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dst)
        print(f"[mlflow] best.pt → {dst}")
