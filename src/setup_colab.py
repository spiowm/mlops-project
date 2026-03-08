"""
Підготовка середовища для тренування в Google Colab.

Використання в Colab notebook:
    import os
    os.environ["DAGSHUB_TOKEN"] = "..."
    os.environ["DAGSHUB_USER"] = "..."
    os.environ["DAGSHUB_REPO"] = "..."

    !git clone https://github.com/<user>/mlops_project.git
    %cd mlops_project
    !python src/setup_colab.py
    !python src/train.py --epochs 50 --batch 16
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "pose"


def install_deps():
    """Встановлює залежності проєкту."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "mlflow", "dvc", "python-dotenv", "pyyaml", "pandas"],
        check=True,
    )
    print("[setup] Залежності встановлено")


def setup_dvc_credentials():
    """Налаштовує DVC remote credentials з env vars."""
    user = os.environ.get("DAGSHUB_USER")
    token = os.environ.get("DAGSHUB_TOKEN")

    if not (user and token):
        print("[setup] DAGSHUB_USER / DAGSHUB_TOKEN не задані — dvc pull не працюватиме")
        return

    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"], check=True)
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "user", user], check=True)
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "password", token], check=True)
    print("[setup] DVC credentials налаштовано")


def pull_dataset():
    """Завантажує датасет з DagsHub через DVC."""
    images_dir = DATA_DIR / "images"
    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        print(f"[setup] Датасет вже є ({len(list(images_dir.glob('*.jpg')))} зобр.)")
        return

    print("[setup] Завантаження датасету (dvc pull)...")
    subprocess.run(["dvc", "pull"], check=True)
    print(f"[setup] Завантажено: {len(list(images_dir.glob('*.jpg')))} зображень")


def download_pretrained():
    """Завантажує pretrained ваги якщо відсутні."""
    weights_dir = PROJECT_ROOT / "models" / "pretrained"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_path = weights_dir / "yolo11n-pose.pt"
    if model_path.exists():
        print(f"[setup] Pretrained ваги є: {model_path}")
        return

    print("[setup] Завантаження yolo11n-pose.pt...")
    from ultralytics import YOLO
    YOLO("yolo11n-pose.pt")

    downloaded = Path("yolo11n-pose.pt")
    if downloaded.exists():
        downloaded.rename(model_path)
    print(f"[setup] Ваги: {model_path}")


def main():
    print("=" * 50)
    print("  Setup: Bee Pose Estimation")
    print("=" * 50)

    install_deps()
    setup_dvc_credentials()
    pull_dataset()
    download_pretrained()

    print("\n[setup] Готово! Запускайте:")
    print("  python src/train.py --epochs 50 --batch 16")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
