"""
Гіперпараметрична оптимізація YOLO11 Pose з Optuna + MLflow + Hydra.

Запуск:
    python src/optimize.py                          # TPE sampler, 20 trials (з config.yaml)
    python src/optimize.py hpo.sampler=random       # Random sampler
    python src/optimize.py hpo.n_trials=30          # більше trials
    python src/optimize.py hpo.trial_epochs=10      # довше тренування per trial

    # Мінімальний тест (швидка перевірка що все працює):
    python src/optimize.py --config-name config_test

Логування → DagsHub MLflow (якщо задані DAGSHUB_* в .env).
Структура MLflow runs:
    parent run "hpo_study_tpe_<timestamp>"
      ├── child run "trial_000"
      ├── child run "trial_001"
      └── ...
"""

import json
import os
import shutil
import time
from pathlib import Path

import hydra
import mlflow
import optuna
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from config import PROJECT_ROOT, dagshub_config

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# MLflow setup (same logic as train.py)
# ---------------------------------------------------------------------------

def setup_mlflow(experiment_name: str) -> None:
    """Налаштовує MLflow: DagsHub якщо є credentials, інакше — локально."""
    local_uri = str(PROJECT_ROOT / "mlruns")
    mlflow.set_tracking_uri(local_uri)

    token, user, repo = dagshub_config()
    if token and user and repo:
        uri = f"https://dagshub.com/{user}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(uri)
        print(f"[mlflow] DagsHub: {uri}")
    else:
        print(f"[mlflow] Локальний: {local_uri}")

    mlflow.set_experiment(experiment_name)


# ---------------------------------------------------------------------------
# Resolve model weights — auto-download if not present locally
# ---------------------------------------------------------------------------

def resolve_model_weights(weights: str) -> str:
    """
    Повертає шлях до ваг моделі.
    Порядок пошуку:
      1. Абсолютний шлях як є
      2. Відносно кореня проєкту (models/pretrained/ тощо)
      3. Назва файлу (YOLO автоматично завантажить з ultralytics hub)
    """
    # Якщо це абсолютний або вже з PROJECT_ROOT
    abs_path = PROJECT_ROOT / weights
    if abs_path.exists():
        return str(abs_path)

    # Якщо YOLO може завантажити за назвою (yolo11n-pose.pt тощо)
    return weights


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------

def make_sampler(sampler_name: str, seed: int) -> optuna.samplers.BaseSampler:
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Невідомий sampler: '{sampler_name}'. Доступні: tpe, random")


# ---------------------------------------------------------------------------
# Extract metric from YOLO results
# ---------------------------------------------------------------------------

def extract_metric(results, metric_key: str) -> float:
    """Витягує числову метрику з results_dict YOLO. Шукає по підрядку."""
    rd = results.results_dict
    needle = metric_key.replace(" ", "")
    for k, v in rd.items():
        if needle in k.replace(" ", ""):
            return float(v)
    # Fallback: перший ключ що містить mAP50-95
    for k, v in rd.items():
        if "mAP50-95" in k or "mAP50_95" in k:
            return float(v)
    raise KeyError(
        f"Метрика '{metric_key}' не знайдена в results_dict.\n"
        f"Доступні ключі: {list(rd.keys())}"
    )


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def objective_factory(cfg: DictConfig, dataset_yaml: str, model_weights: str):
    """Повертає objective function для Optuna."""
    from ultralytics import YOLO, settings
    settings.update({"mlflow": False, "wandb": False})

    hpo = cfg.hpo

    def objective(trial: optuna.Trial) -> float:
        # --- Suggest hyperparameters з search space в config ---
        params: dict = {}

        if "lr0" in hpo:
            params["lr0"] = trial.suggest_float(
                "lr0", hpo.lr0.low, hpo.lr0.high, log=True
            )
        if "lrf" in hpo:
            params["lrf"] = trial.suggest_float(
                "lrf", hpo.lrf.low, hpo.lrf.high, log=True
            )
        if "batch" in hpo:
            params["batch"] = trial.suggest_categorical(
                "batch", list(hpo.batch.choices)
            )
        if "optimizer" in hpo:
            params["optimizer"] = trial.suggest_categorical(
                "optimizer", list(hpo.optimizer.choices)
            )
        if "mosaic" in hpo:
            params["mosaic"] = trial.suggest_float(
                "mosaic", hpo.mosaic.low, hpo.mosaic.high
            )
        if "degrees" in hpo:
            params["degrees"] = trial.suggest_float(
                "degrees", hpo.degrees.low, hpo.degrees.high
            )
        if "imgsz" in hpo:
            params["imgsz"] = trial.suggest_categorical(
                "imgsz", list(hpo.imgsz.choices)
            )

        run_name = f"trial_{trial.number:03d}"
        print(f"\n[trial {trial.number:02d}] {params}")

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("sampler", hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)

            try:
                model = YOLO(model_weights)
                train_kwargs = dict(
                    data=dataset_yaml,
                    epochs=int(hpo.trial_epochs),
                    imgsz=int(params.get("imgsz", cfg.model.imgsz)),
                    batch=int(params.get("batch", 16)),
                    patience=int(cfg.model.patience),
                    optimizer=params.get("optimizer", "auto"),
                    project=str(PROJECT_ROOT / "runs" / "hpo_trials"),
                    name=run_name,
                    exist_ok=True,
                    verbose=False,
                    plots=False,
                )
                if "lr0" in params:
                    train_kwargs["lr0"] = params["lr0"]
                if "lrf" in params:
                    train_kwargs["lrf"] = params["lrf"]
                if "mosaic" in params:
                    train_kwargs["mosaic"] = params["mosaic"]
                if "degrees" in params:
                    train_kwargs["degrees"] = params["degrees"]

                results = model.train(**train_kwargs)

                score = extract_metric(results, hpo.metric)
                mlflow.log_metric("trial_score", score)

                # Логуємо всі фінальні метрики YOLO
                for k, v in results.results_dict.items():
                    try:
                        safe_k = (
                            k.replace("(", "").replace(")", "")
                             .replace("/", "_").strip()
                        )
                        mlflow.log_metric(safe_k, float(v))
                    except Exception:
                        pass

                run_dir = Path(results.save_dir)

                # Зберегти всі артефакти trial (графіки, матриці, CSV)
                if hpo.get("log_artifacts", False):
                    for artifact_file in run_dir.glob("*"):
                        if artifact_file.is_file() and artifact_file.suffix in (
                            ".png", ".jpg", ".csv", ".json", ".yaml",
                        ):
                            mlflow.log_artifact(
                                str(artifact_file), artifact_path="artifacts"
                            )
                    print(f"[trial {trial.number:02d}] artifacts → DagsHub ✓")

                # Зберегти ваги trial у DagsHub (якщо log_trial_weights=true)
                if hpo.get("log_trial_weights", False):
                    best_pt = run_dir / "weights" / "best.pt"
                    if best_pt.exists():
                        mlflow.log_artifact(str(best_pt), artifact_path="weights")
                        print(f"[trial {trial.number:02d}] weights → DagsHub ✓")

                print(f"[trial {trial.number:02d}] score={score:.4f}")
                return score

            except Exception as e:
                print(f"[trial {trial.number:02d}] FAILED: {e}")
                mlflow.set_tag("failed", str(e))
                return 0.0

    return objective


# ---------------------------------------------------------------------------
# Retrain best model at full epochs (from params.yaml)
# ---------------------------------------------------------------------------

def retrain_best(cfg: DictConfig, best_params: dict, dataset_yaml: str,
                 model_weights: str) -> Path:
    """Перетренувати з найкращими параметрами на повну кількість епох."""
    import yaml
    from ultralytics import YOLO, settings
    settings.update({"mlflow": False, "wandb": False})

    with open(PROJECT_ROOT / "params.yaml") as f:
        train_params = yaml.safe_load(f)["train"]

    model = YOLO(model_weights)

    print(f"\n[retrain] Параметри: {best_params}")
    print(f"[retrain] Epochs: {train_params['epochs']}")

    train_kwargs = dict(
        data=dataset_yaml,
        epochs=int(train_params["epochs"]),
        imgsz=int(best_params.get("imgsz", cfg.model.imgsz)),
        batch=int(best_params.get("batch", train_params["batch"])),
        patience=int(train_params["patience"]),
        optimizer=best_params.get("optimizer", train_params["optimizer"]),
        project=str(PROJECT_ROOT / "runs" / "hpo_best"),
        name="retrain_best",
        exist_ok=True,
        verbose=True,
    )
    if "lr0" in best_params:
        train_kwargs["lr0"] = best_params["lr0"]
    if "lrf" in best_params:
        train_kwargs["lrf"] = best_params["lrf"]
    if "mosaic" in best_params:
        train_kwargs["mosaic"] = best_params["mosaic"]
    if "degrees" in best_params:
        train_kwargs["degrees"] = best_params["degrees"]

    results = model.train(**train_kwargs)

    # Copy best.pt to models/
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "best_model_hpo.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, out_path)
        print(f"[retrain] Збережено: {out_path}")
    else:
        print(f"[retrain] Увага: best.pt не знайдено в {run_dir}")

    # Логуємо всі артефакти retrain у MLflow (parent run)
    if mlflow.active_run():
        # Артефакти (графіки, confusion matrix, results.csv, etc.)
        for artifact_file in run_dir.glob("*"):
            if artifact_file.is_file() and artifact_file.suffix in (
                ".png", ".jpg", ".csv", ".json", ".yaml",
            ):
                mlflow.log_artifact(
                    str(artifact_file), artifact_path="retrain_artifacts"
                )
        # Ваги (best + last)
        weights_dir = run_dir / "weights"
        if weights_dir.exists():
            mlflow.log_artifacts(str(weights_dir), artifact_path="retrain_weights")
        print("[retrain] Артефакти → MLflow ✓")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Paths
    dataset_yaml = to_absolute_path(cfg.data.dataset_yaml)
    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(
            f"Dataset YAML не знайдено: {dataset_yaml}\n"
            "Спочатку запусти: python src/prepare.py"
        )

    model_weights = resolve_model_weights(cfg.model.weights)
    print(f"[optimize] Базова модель: {model_weights}")

    # MLflow
    setup_mlflow(cfg.mlflow.experiment_name)

    # Sampler
    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed)

    # Parent run
    run_name = f"hpo_study_{cfg.hpo.sampler}_{int(time.time())}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", cfg.seed)
        mlflow.set_tag("model_weights", cfg.model.weights)
        mlflow.set_tag("trial_epochs", cfg.hpo.trial_epochs)
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True),
            "config_resolved.json",
        )

        print(f"\n{'='*60}")
        print(f"  HPO Study: sampler={cfg.hpo.sampler}, n_trials={cfg.hpo.n_trials}")
        print(f"  Trial epochs: {cfg.hpo.trial_epochs}  |  Metric: {cfg.hpo.metric}")
        print(f"{'='*60}\n")

        # Create and run study
        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler,
            study_name=run_name,
        )

        objective = objective_factory(cfg, dataset_yaml, model_weights)
        study.optimize(objective, n_trials=int(cfg.hpo.n_trials))

        # Best trial results
        best_trial = study.best_trial
        print(f"\n[study] ✓ Найкращий trial #{best_trial.number}")
        print(f"[study]   Score     : {best_trial.value:.4f}")
        print(f"[study]   Параметри : {best_trial.params}")

        mlflow.log_metric("best_score", float(best_trial.value))
        mlflow.log_metric("best_trial_number", float(best_trial.number))
        mlflow.log_dict(best_trial.params, "best_params.json")

        # Retrain with best params
        print("\n[main] Retrain з найкращими параметрами на повних epochs...")
        best_model_path = retrain_best(
            cfg, best_trial.params, dataset_yaml, model_weights
        )

        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="model")

        # Save best params summary
        best_params_path = PROJECT_ROOT / "models" / "best_params_hpo.json"
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "trial": best_trial.number,
                    "score": best_trial.value,
                    "params": best_trial.params,
                    "sampler": cfg.hpo.sampler,
                    "n_trials": cfg.hpo.n_trials,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(best_params_path))

        print(f"\n[done] Parent run ID: {parent_run.info.run_id}")
        print(f"[done] Best model  : {best_model_path}")
        print(f"[done] Best params : {best_params_path}")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
