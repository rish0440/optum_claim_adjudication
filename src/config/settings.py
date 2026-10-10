"""Load local project settings once; resolve data paths from repo root."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.utils.paths import ROOT, project_path


class PathSettings(BaseModel):
    policy: Path
    optum_invoices: Path
    evaluation_ground_truth: Path
    synthetic_dataset: Path
    cord: Path

    @model_validator(mode="after")
    def make_absolute(self) -> "PathSettings":
        for key in type(self).model_fields:
            value = getattr(self, key)
            if not value.is_absolute():
                setattr(self, key, ROOT / value)
        return self


class SyntheticSettings(BaseModel):
    initial_count: int = Field(default=10, ge=1)
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    seed: int = 42
    min_rows: int = Field(default=3, ge=1)
    max_rows: int = Field(default=9, ge=1)
    missing_rate_fraction: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def check_rows(self) -> "SyntheticSettings":
        if self.min_rows > self.max_rows:
            raise ValueError("min_rows cannot exceed max_rows")
        return self


class RagSettings(BaseModel):
    """Local Milvus Lite RAG configuration; no database server/API key."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection: str = "policy_rules"
    db_path: Path = Path("data/policy_index/policy.db")
    manifest_path: Path = Path("data/policy_index/manifest.json")
    top_k: int = Field(default=5, ge=1, le=45)
    min_score: float | None = Field(default=None, ge=-1, le=1)
    query_timeout: float = Field(default=10, gt=0)

    @model_validator(mode="after")
    def paths_from_repo(self) -> "RagSettings":
        for name in ("db_path", "manifest_path"):
            value = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, ROOT / value)
        return self


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    app: dict[str, Any]
    paths: PathSettings
    synthetic: SyntheticSettings
    rag: RagSettings = Field(default_factory=RagSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load configurable YAML; resolve relative paths against the project root."""
    config_path = project_path(os.environ.get("OPTUM_CONFIG_FILE", "src/config/default.yaml"))
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    # Share the dataset override with portable training CLIs without requiring
    # application dependencies inside the synthetic-only training bundle.
    if "OPTUM_SYNTHETIC_DATASET" in os.environ:
        raw["paths"]["synthetic_dataset"] = os.environ["OPTUM_SYNTHETIC_DATASET"]
    return AppSettings.model_validate(raw)
