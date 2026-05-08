"""
Shim for analysis.output_paths.
The microservices don't use the filesystem layout from the original
batch pipeline, so this provides no-op stubs for all symbols that
chemical_modeling.py, species.py, and features.py import at module
level.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

SCOPES: List[str] = ["meta", "air", "diameter"]

def metadata_csv_path(scope: str, section: str, filename: str) -> Path:
    return Path("/tmp") / scope / section / filename

def metadata_section_dir(scope: str, section: str) -> Path:
    p = Path("/tmp") / scope / section
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_all_scope_layouts() -> None:
    pass

def chemical_modeling_dir(scope: str) -> Path:
    p = Path("/tmp") / scope / "chemical_modeling"
    p.mkdir(parents=True, exist_ok=True)
    return p

def chemspecies_figures_dir(scope: str) -> Path:
    p = Path("/tmp") / scope / "chemspecies"
    p.mkdir(parents=True, exist_ok=True)
    return p

def pca_dir(scope: str) -> Path:
    p = Path("/tmp") / scope / "pca"
    p.mkdir(parents=True, exist_ok=True)
    return p
