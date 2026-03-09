"""
Конфігурація проєкту + налаштування середовища.

При імпорті автоматично завантажує credentials (.env або Colab Secrets).
Як скрипт — додатково налаштовує DVC remote auth:
    python src/config.py   # потім: dvc pull
"""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Завантаження credentials ---
# 1. .env файл (локально)
load_dotenv(PROJECT_ROOT / ".env")

# 2. Colab Secrets (якщо в Colab)
try:
    from google.colab import userdata  # type: ignore[import-not-found]
    for _key in ("DAGSHUB_TOKEN", "DAGSHUB_USER", "DAGSHUB_REPO"):
        _val = userdata.get(_key)
        if _val:
            os.environ[_key] = _val
except ImportError:
    pass


def dagshub_config() -> tuple[str | None, str | None, str | None]:
    """Повертає (token, user, repo) з env vars. None якщо не задано."""
    return (
        os.environ.get("DAGSHUB_TOKEN"),
        os.environ.get("DAGSHUB_USER"),
        os.environ.get("DAGSHUB_REPO"),
    )


def setup_dvc_credentials():
    """Налаштовує DVC remote auth з env vars."""
    token, user, _ = dagshub_config()
    if not (user and token):
        print("[setup] DAGSHUB_USER/DAGSHUB_TOKEN не задані — DVC credentials не налаштовано")
        return

    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"], check=True)
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "user", user], check=True)
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "password", token], check=True)
    print("[setup] DVC credentials налаштовано")


if __name__ == "__main__":
    setup_dvc_credentials()
    print("\nДалі:")
    print("  dvc pull                        # завантажити датасет")
    print("  python src/train.py --epochs 50 # тренування")
