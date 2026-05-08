"""
Shim for data_ingestion.scoped_writes.
The write helpers are only used in main() of species.py / features.py,
which the services never call. This stub prevents ImportError at module load.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd

def write_scoped_csv(df: pd.DataFrame, section: str, filename: str,
                     allow_global: bool = False, scopes=None) -> List[Path]:
    return []
