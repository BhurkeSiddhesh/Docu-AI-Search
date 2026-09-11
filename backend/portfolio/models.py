"""Validation for explicitly reviewed portfolio imports."""
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class TableMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    table_id: str
    mapping: Dict[str, str] = Field(default_factory=dict)
    ignored: bool = False


class MappingConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tables: List[TableMapping] = Field(min_length=1, max_length=100)
    defaults: Dict[str, str] = Field(default_factory=dict)

