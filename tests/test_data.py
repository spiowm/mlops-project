"""
Pre-train тести: валідація даних для YOLO Pose.

Перевіряє структуру data/split/, формат YOLO-міток,
коректність координат та мінімальну кількість зразків.
"""

import yaml

# ── Структура data/split ───────────────────────────────────────────────────


def test_split_dir_exists(split_dir):
    """data/split/ має існувати (після prepare stage)."""
    assert split_dir.exists(), (
        f"{split_dir} не знайдено — спочатку запустіть: dvc repro prepare"
    )


def test_dataset_yaml_exists(split_dir):
    """dataset.yaml має існувати у data/split/."""
    ds_yaml = split_dir / "dataset.yaml"
    assert ds_yaml.exists(), f"{ds_yaml} не знайдено"


def test_dataset_yaml_keys(split_dir):
    """dataset.yaml має містити необхідні ключі для YOLO Pose."""
    ds_yaml = split_dir / "dataset.yaml"
    if not ds_yaml.exists():
        return  # пропускаємо якщо файл ще не створений

    with open(ds_yaml) as f:
        config = yaml.safe_load(f)

    required_keys = {"path", "train", "val", "kpt_shape", "names"}
    missing = required_keys - set(config.keys())
    assert not missing, f"dataset.yaml: відсутні ключі: {sorted(missing)}"

    # kpt_shape для bee pose: [2, 2] (2 keypoints, 2D)
    assert config["kpt_shape"] == [2, 2], (
        f"Очікується kpt_shape=[2, 2], отримано: {config['kpt_shape']}"
    )


def test_train_val_dirs_exist(split_dir):
    """Підкаталоги train/val з images/ та labels/ мають існувати."""
    for subset in ("train", "val"):
        for subdir in ("images", "labels"):
            d = split_dir / subset / subdir
            assert d.exists(), f"Директорія {d} не знайдена"


def test_min_images(split_dir):
    """Мінімальна кількість зображень: ≥20 train, ≥5 val."""
    for subset, min_count in [("train", 20), ("val", 5)]:
        images_dir = split_dir / subset / "images"
        if not images_dir.exists():
            continue
        count = len(list(images_dir.glob("*.jpg")))
        assert count >= min_count, (
            f"{subset}: {count} зображень, потрібно ≥{min_count}"
        )


def test_images_have_labels(split_dir):
    """Кожне зображення має мати відповідний .txt label-файл."""
    for subset in ("train", "val"):
        images_dir = split_dir / subset / "images"
        labels_dir = split_dir / subset / "labels"
        if not images_dir.exists():
            continue

        images = list(images_dir.glob("*.jpg"))
        missing = []
        for img in images[:50]:  # перевіряємо перші 50 для швидкості
            label = labels_dir / img.with_suffix(".txt").name
            if not label.exists():
                missing.append(img.name)

        assert not missing, (
            f"{subset}: {len(missing)} зображень без label-файлів: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )


def test_label_format(split_dir):
    """Label-файли мають бути у форматі YOLO Pose: class cx cy w h kp1x kp1y kp2x kp2y."""
    labels_dir = split_dir / "train" / "labels"
    if not labels_dir.exists():
        return

    label_files = list(labels_dir.glob("*.txt"))
    assert label_files, "Немає label-файлів у train/labels/"

    for label_file in label_files[:20]:  # перевіряємо перші 20
        with open(label_file) as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            values = line.split()
            assert len(values) == 9, (
                f"{label_file.name}:{i+1}: очікується 9 значень "
                f"(class cx cy w h kp1x kp1y kp2x kp2y), отримано {len(values)}"
            )

            # Перевіряємо що всі значення — числа
            try:
                nums = [float(v) for v in values]
            except ValueError:
                raise AssertionError(
                    f"{label_file.name}:{i+1}: не всі значення є числами: {line}"
                )

            # class має бути 0 (bee)
            assert nums[0] == 0, (
                f"{label_file.name}:{i+1}: class={nums[0]}, очікується 0"
            )

            # bbox координати мають бути в [0, 1]
            for j, name in [(1, "cx"), (2, "cy"), (3, "w"), (4, "h")]:
                assert 0 <= nums[j] <= 1, (
                    f"{label_file.name}:{i+1}: {name}={nums[j]} поза діапазоном [0,1]"
                )
