#!/usr/bin/env python3
"""
Attention mechanism family tree: Bahdanau (2014) → DeepSeek-V4 (2025).

Renders attention_lineage.png and attention_lineage.svg using pure matplotlib
(no graphviz binary required).

Edge legend:
  solid  = direct ancestry (one work directly builds on another)
  dashed = conceptual influence (borrows ideas without being a direct descendant)
  dotted = systems enabler (IO-aware implementation that enables the architecture)

Usage:
    uv run attention_lineage.py
    python attention_lineage.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

OUT_DIR = Path(__file__).parent

# ── colour palette ─────────────────────────────────────────────────────────────
C_FOUND  = "#AED6F1"   # light blue  — foundational attention
C_MEM    = "#D7BDE2"   # light purple — memory / recurrence
C_SPARSE = "#FAD7A0"   # light orange — sparse attention family
C_EFF    = "#D5DBDB"   # light grey  — efficient attention (Performer/Reformer)
C_KV     = "#A9DFBF"   # light green — KV-head compression
C_SYS    = "#F0F0F0"   # very light grey — systems / IO paper
C_DS4P   = "#FADBD8"   # very light salmon — DeepSeek-V3.2 precursor (DSA)
C_DS4    = "#F1948A"   # salmon — DeepSeek-V4 cluster

# ── nodes: id → (two-line label, fill colour) ─────────────────────────────────
NODES: dict[str, tuple[str, str]] = {
    "N0":  ("Bahdanau\nattention",                           C_FOUND),
    "N1":  ("Luong\nglobal/local attention",                 C_FOUND),
    "N2":  ("Transformer\nMHA",                              C_FOUND),
    "N3":  ("Transformer-XL\nrecurrence / memory",           C_MEM),
    "N4":  ("Sparse Transformer\nfixed sparse masks",        C_SPARSE),
    "N5":  ("Compressive Transformer\ncompressed memory",    C_MEM),
    "N6":  ("Longformer\nlocal-window + global",             C_SPARSE),
    "N7":  ("BigBird\nlocal/random/global sparse",           C_SPARSE),
    "N8":  ("Routing Transformer\ncontent-based routing",    C_SPARSE),
    "N9":  ("Performer / Reformer\nefficient attn cousins",  C_EFF),
    "N10": ("MQA\nshared KV heads",                          C_KV),
    "N11": ("GQA\ngrouped KV heads",                         C_KV),
    "N12": ("DeepSeek-V2 MLA\nlow-rank latent KV",           C_KV),
    "N13": ("FlashAttention\nIO-aware kernels",               C_SYS),
    "N14": ("DeepSeek-V2 stack\nMLA + DeepSeekMoE",          C_KV),
    "N15": ("Native Sparse Attn\ncompression + selection",   C_SPARSE),
    "N16": ("DeepSeek-V3.2 DSA\ntop-k sparse attn + MLA",   C_DS4P),
    "N17": ("DeepSeek-V4\nCSA",                              C_DS4),
    "N18": ("DeepSeek-V4\nHCA",                              C_DS4),
    "N19": ("DeepSeek-V4\nhybrid CSA/HCA",                   C_DS4),
    "N20": ("DeepSeek-V4\nheterogeneous KV-cache",           C_DS4),
    "N21": ("DeepSeek-V4\nlow-prec indexer / serving",       C_DS4),
}

# ── node positions: id → (x-centre, y-centre) in data coordinates ─────────────
POS: dict[str, tuple[float, float]] = {
    # core spine
    "N0":  (10.0, 26.0),
    "N1":  (10.0, 24.3),
    "N2":  (10.0, 22.6),

    # generation 3 — seven children of N2, spread across full width
    "N3":  ( 1.5, 20.2),
    "N4":  ( 4.33, 20.2),
    "N6":  ( 7.17, 20.2),
    "N7":  (10.0,  20.2),
    "N8":  (12.83, 20.2),
    "N9":  (15.67, 20.2),
    "N10": (18.0,  20.2),

    # generation 4
    "N5":  ( 1.5, 17.8),
    "N11": (18.0, 17.8),

    # generation 5
    "N15": ( 7.0, 15.5),
    "N12": (15.5, 15.5),
    "N13": (19.0, 10.5),   # systems enabler — floated right beside DS4 generation

    # generation 6
    "N16": (11.5, 13.0),
    "N14": (16.5, 13.0),

    # generation 7
    "N17": (10.0, 10.5),
    "N18": ( 2.0, 10.5),

    # generation 8-10
    "N19": (10.0,  8.0),
    "N20": (10.0,  5.5),
    "N21": (10.0,  3.0),
}

# ── node geometry ──────────────────────────────────────────────────────────────
NODE_W  = 2.5    # full width
NODE_H  = 1.0    # full height
HW      = NODE_W / 2
HH      = NODE_H / 2

# ── edges: (src, dst, style, rad) ─────────────────────────────────────────────
# rad: curvature (arc3); 0 = straight; + curves right when going down
EDGES: list[tuple[str, str, str, float]] = [
    # core spine
    ("N0",  "N1",  "solid",  0.0),
    ("N1",  "N2",  "solid",  0.0),

    # N2 fan-out
    ("N2",  "N3",  "solid",  0.0),
    ("N2",  "N4",  "solid",  0.0),
    ("N2",  "N6",  "solid",  0.0),
    ("N2",  "N7",  "solid",  0.0),
    ("N2",  "N8",  "solid",  0.0),
    ("N2",  "N9",  "dashed", 0.0),
    ("N2",  "N10", "solid",  0.0),

    # memory branch
    ("N3",  "N5",  "solid",  0.0),
    ("N5",  "N18", "solid",  0.0),

    # sparse branch
    ("N4",  "N16", "dashed", 0.25),   # curve right to avoid N15
    ("N6",  "N17", "solid",  0.0),
    ("N7",  "N17", "dashed", 0.0),
    ("N8",  "N16", "solid",  0.0),
    ("N15", "N16", "solid",  0.0),
    ("N16", "N17", "solid",  0.0),

    # KV compression branch
    ("N2",  "N10", "solid",  0.0),   # duplicate removed at render time
    ("N10", "N11", "solid",  0.0),
    ("N11", "N12", "solid",  0.0),
    ("N12", "N14", "solid",  0.0),
    ("N12", "N16", "solid",  0.2),   # curve slightly to distinguish from N8→N16
    ("N12", "N19", "solid",  0.3),   # long-range; curve right to stay clear

    # systems enabler (FlashAttention)
    ("N13", "N20", "dotted", 0.0),
    ("N13", "N21", "dotted", 0.1),

    # DeepSeek-V4 cluster convergence
    ("N17", "N19", "solid",  0.0),
    ("N18", "N19", "solid",  0.0),
    ("N19", "N20", "solid",  0.0),
    ("N20", "N21", "solid",  0.0),
]

# deduplicate (N2→N10 appears twice in the spec)
seen: set[tuple[str, str]] = set()
EDGES_DEDUP: list[tuple[str, str, str, float]] = []
for e in EDGES:
    key = (e[0], e[1])
    if key not in seen:
        seen.add(key)
        EDGES_DEDUP.append(e)
EDGES = EDGES_DEDUP

# ── edge style table ───────────────────────────────────────────────────────────
EDGE_CFG = {
    "solid":  {"color": "#1a1a1a", "lw": 2.0,  "ls": "-",        "alpha": 1.0},
    "dashed": {"color": "#555555", "lw": 1.6,  "ls": "--",       "alpha": 0.85},
    "dotted": {"color": "#888888", "lw": 1.4,  "ls": (0,(2,3)),  "alpha": 0.75},
}


# ── drawing helpers ────────────────────────────────────────────────────────────

def _arrow(ax: plt.Axes, src: str, dst: str, style: str, rad: float) -> None:
    """Draw one directed edge."""
    sx, sy = POS[src]
    dx, dy = POS[dst]
    cfg    = EDGE_CFG[style]

    arr = FancyArrowPatch(
        (sx, sy - HH),           # depart from bottom centre of source
        (dx, dy + HH),           # arrive at top centre of destination
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=11,
        color=cfg["color"],
        lw=cfg["lw"],
        linestyle=cfg["ls"],
        alpha=cfg["alpha"],
        zorder=2,
    )
    ax.add_patch(arr)


def _node(ax: plt.Axes, nid: str) -> None:
    """Draw a rounded rectangle with label."""
    label, color = NODES[nid]
    cx, cy = POS[nid]

    box = FancyBboxPatch(
        (cx - HW, cy - HH), NODE_W, NODE_H,
        boxstyle="round,pad=0.12",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        cx, cy, label,
        ha="center", va="center",
        fontsize=8.5, zorder=4,
        multialignment="center",
    )


def _legend(ax: plt.Axes) -> None:
    lx, ly = 0.3, 5.5
    ax.text(lx, ly + 1.35, "Legend", fontsize=9, fontweight="bold", zorder=5)

    for i, (style, label) in enumerate([
        ("solid",  "Direct ancestry"),
        ("dashed", "Conceptual influence"),
        ("dotted", "Systems enabler"),
    ]):
        y = ly + 0.75 - i * 0.62
        cfg = EDGE_CFG[style]
        arr = FancyArrowPatch(
            (lx, y), (lx + 1.1, y),
            connectionstyle="arc3,rad=0",
            arrowstyle="-|>",
            mutation_scale=9,
            color=cfg["color"],
            lw=cfg["lw"],
            linestyle=cfg["ls"],
            alpha=cfg["alpha"],
            zorder=5,
        )
        ax.add_patch(arr)
        ax.text(lx + 1.3, y, label, va="center", fontsize=8.0, zorder=5)

    # colour key for node families
    families = [
        (C_FOUND,  "Foundational attention"),
        (C_MEM,    "Memory / recurrence"),
        (C_SPARSE, "Sparse attention"),
        (C_EFF,    "Efficient attention"),
        (C_KV,     "KV-head compression"),
        (C_SYS,    "IO-aware systems paper"),
        (C_DS4P,   "DeepSeek-V3.2 (precursor)"),
        (C_DS4,    "DeepSeek-V4"),
    ]
    ax.text(lx, ly - 0.8, "Node families", fontsize=9, fontweight="bold", zorder=5)
    for i, (color, label) in enumerate(families):
        y = ly - 1.35 - i * 0.54
        r = FancyBboxPatch(
            (lx, y - 0.17), 0.7, 0.34,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="#555",
            linewidth=0.9,
            zorder=5,
        )
        ax.add_patch(r)
        ax.text(lx + 0.88, y, label, va="center", fontsize=7.8, zorder=5)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    fig, ax = plt.subplots(figsize=(20, 27))
    ax.set_xlim(0, 20)
    ax.set_ylim(-1.0, 28.0)
    ax.axis("off")

    # Title
    ax.text(
        10.0, 27.7,
        "Attention Mechanism Family Tree",
        ha="center", va="center", fontsize=14, fontweight="bold", zorder=6,
    )
    ax.text(
        10.0, 27.2,
        "Bahdanau (2014) → DeepSeek-V4 (2025)   ·   N0–N21",
        ha="center", va="center", fontsize=9, color="#555555", zorder=6,
    )

    # Edges drawn before nodes so arrowheads appear at node boundaries
    for src, dst, style, rad in EDGES:
        _arrow(ax, src, dst, style, rad)

    # Nodes (on top of edges except arrowhead tips)
    for nid in NODES:
        _node(ax, nid)

    # Node ID annotations (small, top-left corner of each box)
    for nid in NODES:
        cx, cy = POS[nid]
        ax.text(
            cx - HW + 0.08, cy + HH - 0.12,
            nid,
            fontsize=6.5, color="#666666",
            va="top", ha="left", zorder=5,
        )

    _legend(ax)

    plt.tight_layout(pad=0.5)

    png = OUT_DIR / "attention_lineage.png"
    svg = OUT_DIR / "attention_lineage.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"Saved  {png}")
    print(f"Saved  {svg}")


if __name__ == "__main__":
    main()
