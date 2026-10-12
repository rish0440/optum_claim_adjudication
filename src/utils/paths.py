"""Repository-relative paths shared by CPU code and portable training bundles."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("OPTUM_PROJECT_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def project_path(value: str | Path) -> Path:
    """Resolve a configured relative path against the project, not the shell CWD."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def configured_path(variable: str, default: str | Path) -> Path:
    return project_path(os.environ.get(variable, default))


def assignment_annotation_path(value: str | Path) -> Path:
    """Resolve recorded assignment paths while keeping original label bytes intact."""
    path = Path(value)
    for original, current in (
        (Path("data/invoices"), Path("data/assignment_dataset/invoices")),
        (Path("data/evaluation/ground_truth"), Path("data/assignment_dataset/ground_truth")),
    ):
        if path.is_relative_to(original):
            return current / path.relative_to(original)
    return path
