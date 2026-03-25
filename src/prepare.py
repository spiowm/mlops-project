"""
Підготовка даних: split raw → train/val за ID вулика.

Запуск:
    dvc repro prepare                  # через пайплайн
    python src/prepare.py              # читає params.yaml
"""

import re
import shutil
from pathlib import Path

import yaml

from config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET = "pose"
DEFAULT_VAL_HIVES = ["20230711b", "20230609e"]


def extract_hive_id(filename: str) -> str:
    """Витягує ID вулика (напр. '20230609a') з імені файлу."""
    match = re.match(r"(\d{8}[a-z])", filename)
    return match.group(1) if match else "unknown"


def prepare_data(
    dataset: str = DEFAULT_DATASET,
    val_hives: list[str] | None = None,
    force: bool = False,
) -> tuple[Path, dict[str, int]]:
    """
    Розподіляє зображення та мітки по train/val на основі hive ID.

    Args:
        dataset: назва датасету (підпапка в data/raw/, напр. 'pose')
        val_hives: список hive ID для валідації
        force: перестворити split навіть якщо він вже існує

    Returns:
        (шлях до dataset.yaml, {"train": N, "val": M})
    """
    if val_hives is None:
        val_hives = DEFAULT_VAL_HIVES

    raw_dir = DATA_DIR / "raw" / dataset
    split_dir = DATA_DIR / "split" / dataset
    dataset_yaml = split_dir / "dataset.yaml"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"[prepare] Датасет не знайдено: {raw_dir}\n"
            f"  Перевір params.yaml → prepare.dataset = '{dataset}'\n"
            f"  та що data/raw/{dataset}/ існує (можливо треба dvc pull)"
        )

    if dataset_yaml.exists() and not force:
        counts = {
            s: len(list((split_dir / s / "images").glob("*.jpg")))
            for s in ("train", "val")
        }
        print(f"[prepare] Split вже існує: {split_dir} (force=True для перестворення)")
        return dataset_yaml, counts

    if split_dir.exists():
        shutil.rmtree(split_dir)

    for subset in ("train", "val"):
        (split_dir / subset / "images").mkdir(parents=True)
        (split_dir / subset / "labels").mkdir(parents=True)

    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    counts = {"train": 0, "val": 0}

    for img_file in sorted(images_dir.glob("*.jpg")):
        hive_id = extract_hive_id(img_file.name)
        subset = "val" if hive_id in val_hives else "train"

        shutil.copy2(img_file, split_dir / subset / "images" / img_file.name)

        label_file = labels_dir / img_file.with_suffix(".txt").name
        if label_file.exists():
            shutil.copy2(label_file, split_dir / subset / "labels" / label_file.name)

        counts[subset] += 1

    # kpt_shape визначається автоматично з першого label файлу
    first_label = next((split_dir / "train" / "labels").glob("*.txt"), None)
    kpt_shape = [2, 2]  # default для pose датасету (2 точки, xy без confidence)
    if first_label:
        with open(first_label) as f:
            parts = f.readline().strip().split()
        if len(parts) >= 9:
            n_kpts = (len(parts) - 5) // 2
            kpt_shape = [n_kpts, 2]  # без confidence flag
        elif len(parts) >= 11:
            n_kpts = (len(parts) - 5) // 3
            kpt_shape = [n_kpts, 3]  # з confidence flag

    config = {
        "path": str(split_dir),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": kpt_shape,
        "names": {0: "bee"},
    }
    with open(dataset_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[prepare] dataset={dataset}  train={counts['train']}, val={counts['val']}")
    print(f"[prepare] kpt_shape={kpt_shape}")
    return dataset_yaml, counts


if __name__ == "__main__":
    with open(PROJECT_ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)

    dataset = params["prepare"].get("dataset", DEFAULT_DATASET)
    val_hives = params["prepare"]["val_hives"]

    dataset_yaml, counts = prepare_data(dataset=dataset, val_hives=val_hives, force=True)
    print(f"[prepare] → {dataset_yaml}")
