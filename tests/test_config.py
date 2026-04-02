"""
Pre-train тести: валідація конфігурації проєкту.

Перевіряє params.yaml, config/config.yaml, наявність моделі.
"""


import yaml


def test_params_yaml_exists(params_path):
    """params.yaml має існувати."""
    assert params_path.exists(), f"{params_path} не знайдено"


def test_params_yaml_sections(params_path):
    """params.yaml має містити секції train та prepare."""
    with open(params_path) as f:
        data = yaml.safe_load(f)
    assert "train" in data, "Секція 'train' не знайдена в params.yaml"
    assert "prepare" in data, "Секція 'prepare' не знайдена в params.yaml"


def test_train_params_valid(params_path):
    """Параметри тренування мають бути валідними."""
    if not params_path.exists():
        return

    with open(params_path) as f:
        train = yaml.safe_load(f).get("train", {})

    assert train.get("epochs", 0) > 0, "epochs має бути > 0"
    assert train.get("batch", 0) > 0, "batch має бути > 0"
    assert train.get("imgsz", 0) > 0, "imgsz має бути > 0"
    assert 0 < train.get("lr0", 0) < 1, "lr0 має бути в (0, 1)"
    assert train.get("patience", 0) > 0, "patience має бути > 0"


def test_prepare_params_valid(params_path):
    """Параметри підготовки даних мають бути валідними."""
    if not params_path.exists():
        return

    with open(params_path) as f:
        prepare = yaml.safe_load(f).get("prepare", {})

    val_hives = prepare.get("val_hives", [])
    assert isinstance(val_hives, list), "val_hives має бути списком"
    assert len(val_hives) >= 1, "val_hives має містити хоча б 1 вулик"


def test_hydra_config_exists(config_path):
    """config/config.yaml (Hydra) має існувати."""
    assert config_path.exists(), f"{config_path} не знайдено"


def test_hydra_config_parsable(config_path):
    """config/config.yaml має парситись без помилок."""
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "config.yaml має бути словником"
    assert "model" in config, "config.yaml: відсутня секція 'model'"
    assert "hpo" in config, "config.yaml: відсутня секція 'hpo'"


def test_model_weights_available(params_path, project_root):
    """Файл моделі має бути доступний (локально або як YOLO model ID)."""
    if not params_path.exists():
        return

    with open(params_path) as f:
        model = yaml.safe_load(f)["train"].get("model", "")

    model_path = project_root / model
    # Допустимо: або файл існує, або це назва YOLO моделі (yolo*.pt)
    is_local = model_path.exists()
    is_yolo_id = model.endswith(".pt") and "yolo" in model.lower()

    assert is_local or is_yolo_id, (
        f"Модель '{model}' не знайдена локально ({model_path}) "
        f"і не виглядає як YOLO model ID"
    )


def test_dvc_yaml_exists(project_root):
    """dvc.yaml має існувати для відтворюваного пайплайну."""
    dvc_path = project_root / "dvc.yaml"
    assert dvc_path.exists(), f"{dvc_path} не знайдено"


def test_dvc_yaml_stages(project_root):
    """dvc.yaml має описувати стадії prepare та train."""
    dvc_path = project_root / "dvc.yaml"
    if not dvc_path.exists():
        return

    with open(dvc_path) as f:
        dvc = yaml.safe_load(f)

    stages = dvc.get("stages", {})
    for stage in ("prepare", "train"):
        assert stage in stages, f"dvc.yaml: відсутня стадія '{stage}'"
