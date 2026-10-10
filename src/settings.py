"""Load local project settings once; resolve data paths from repo root."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent


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
    """Load src/config.yaml into validated Pydantic configuration."""
    with (ROOT / "src" / "config.yaml").open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return AppSettings.model_validate(raw)
