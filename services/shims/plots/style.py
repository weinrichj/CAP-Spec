"""
Shim for plots.style.
chemical_modeling.py imports apply_publication_style, get_palette,
style_axes, to_species_label at module level. These are only used in
the plot-to-file functions which the services never call directly —
the reporting service generates its own charts. Stubs prevent ImportError.
"""
from __future__ import annotations
from typing import List
import matplotlib.pyplot as plt

def apply_publication_style() -> None:
    pass

def get_palette(n: int, name: str = "tab10") -> List[str]:
    cmap = plt.get_cmap(name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]

def style_axes(ax, grid_axis: str = "both") -> None:
    if grid_axis in ("both", "x"):
        ax.xaxis.grid(True, alpha=0.3)
    if grid_axis in ("both", "y"):
        ax.yaxis.grid(True, alpha=0.3)

def to_species_label(species: str) -> str:
    return str(species)
