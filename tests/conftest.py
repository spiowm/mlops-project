"""
Shared fixtures для тестів CI/CD пайплайну bee pose estimation.
"""

import os
from pathlib import Path

import pytest

# ── Шляхи ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def split_dir(project_root):
    return project_root / "data" / "split"


@pytest.fixture
def params_path(project_root):
    return project_root / "params.yaml"


@pytest.fixture
def config_path(project_root):
    return project_root / "config" / "config.yaml"


@pytest.fixture
def metrics_path(project_root):
    return project_root / "metrics.json"


@pytest.fixture
def baseline_path(project_root):
    return project_root / "baseline" / "metrics.json"


# ── Пороги Quality Gate (з env vars для гнучкості в CI) ────────────────────

@pytest.fixture
def map50_threshold():
    """Мінімальний mAP50 для Quality Gate. Для CI (1 epoch CPU) — дуже низький."""
    return float(os.getenv("MAP50_THRESHOLD", "0.01"))


@pytest.fixture
def map50_95_threshold():
    """Мінімальний mAP50-95 для Quality Gate."""
    return float(os.getenv("MAP50_95_THRESHOLD", "0.005"))
