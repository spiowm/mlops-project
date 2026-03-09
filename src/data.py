"""Розбиття сирих даних на train/val за ідентифікатором вулика."""

import re
import shutil
from pathlib import Path

import yaml

from config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "pose"
SPLIT_DIR = PROJECT_ROOT / "data" / "split"

# Валідаційні вулики обрані так, щоб уникнути data leakage між train/val
DEFAULT_VAL_HIVES = ["20230711b", "20230609e"]


def extract_hive_id(filename: str) -> str:
    """Витягує ID вулика (напр. '20230609a') з імені файлу."""
    match = re.match(r"(\d{8}[a-z])", filename)
    return match.group(1) if match else "unknown"


def prepare_data(force: bool = False) -> tuple[Path, dict[str, int]]:
    """
    Розподіляє зображення та мітки по train/val на основі hive ID.
    Повертає (шлях до dataset.yaml, {"train": N, "val": M}).
    """
    dataset_yaml = SPLIT_DIR / "dataset.yaml"

    if dataset_yaml.exists() and not force:
        # Порахувати існуючі
        counts = {
            s: len(list((SPLIT_DIR / s / "images").glob("*.jpg")))
            for s in ("train", "val")
        }
        print("[data] Split вже існує (--force-split для перестворення)")
        return dataset_yaml, counts

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    for subset in ("train", "val"):
        (SPLIT_DIR / subset / "images").mkdir(parents=True)
        (SPLIT_DIR / subset / "labels").mkdir(parents=True)

    images_dir = RAW_DIR / "images"
    labels_dir = RAW_DIR / "labels"
    counts = {"train": 0, "val": 0}

    for img_file in sorted(images_dir.glob("*.jpg")):
        hive_id = extract_hive_id(img_file.name)
        subset = "val" if hive_id in DEFAULT_VAL_HIVES else "train"

        shutil.copy2(img_file, SPLIT_DIR / subset / "images" / img_file.name)

        label_file = labels_dir / img_file.with_suffix(".txt").name
        if label_file.exists():
            shutil.copy2(label_file, SPLIT_DIR / subset / "labels" / label_file.name)

        counts[subset] += 1

    config = {
        "path": str(SPLIT_DIR),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": [2, 2],
        "names": {0: "bee"},
    }
    with open(dataset_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[data] Split: train={counts['train']}, val={counts['val']}")
    return dataset_yaml, counts
