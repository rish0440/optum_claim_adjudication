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
