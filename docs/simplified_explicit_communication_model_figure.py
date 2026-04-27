"""Upper-half Manhattan figure for the simplified explicit communication model.

A single arena above the core: every byte (inputs, intermediates, outputs)
lives in the upper half-plane and is read at cost = Manhattan distance from
the core. Reads are priced; writes and arithmetic are free.

Renders ``simplified_explicit_communication_model.svg/png`` alongside this
script. Adapted from ``manhattan_function_figure.py`` (which renders the
two-arena diamond).
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def isqrt_ceil(x: int) -> int:
    if x <= 0:
        return 0
    return math.isqrt(x - 1) + 1


def upper_half_spiral(n: int):
    """Spiral packing depths 1..n into the upper half-plane (y > 0)."""
    for i in range(1, n + 1):
        k = isqrt_ceil(i)
        start_i = (k - 1) ** 2 + 1
        idx = i - start_i
        if k % 2 == 1:
            x = -(k - 1) + idx
        else:
            x = (k - 1) - idx
        y = k - abs(x)
        yield x, y


N_PTS = 400
PTS = np.array(list(upper_half_spiral(N_PTS)))


RING_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # rings 1..4
MAX_RING = len(RING_COLORS)


def render_frame(n_short: int = 16, table_size: int = 16) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(10, 7))

    short = PTS[:n_short]
    dists = np.array([abs(x) + abs(y) for x, y in short])
    colors = [RING_COLORS[min(int(v), MAX_RING) - 1] for v in dists]
    ax1.scatter(short[:, 0], short[:, 1], c=colors, s=55,
                zorder=2, edgecolors='black', linewidths=0.4)

    # Cell labels.
    for i in range(min(table_size, n_short)):
        on_spine = short[i, 0] == 0
        x_off = 6 if on_spine else 0
        y_off = 7
        ha = 'left' if on_spine else 'center'
        ax1.annotate(
            f"{i + 1}", (short[i, 0], short[i, 1]),
            textcoords="offset points", xytext=(x_off, y_off),
            ha=ha, va='bottom', fontsize=9,
            color='black', alpha=0.7, zorder=4,
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='none', alpha=0.5),
        )

    # H-tree wires: vertical spine + per-row horizontal branches.
    LW = 2.0
    WIRE_COLOR = '#bdbdbd'
    y_max = int(short[:, 1].max())
    ax1.plot([0, 0], [0, y_max], color=WIRE_COLOR,
             lw=LW, zorder=0.3, solid_capstyle='round')

    for k in range(1, MAX_RING + 1):
        xs = [x for x, y in short if y == k]
        if xs:
            ax1.plot([min(xs), max(xs)], [k, k], color=WIRE_COLOR,
                     lw=LW, zorder=0.3, solid_capstyle='round')

    # Core (ALU) at the origin.
    ax1.plot(0, 0, marker='o', color='red', ms=14, mec='black', zorder=5)
    ax1.annotate("core", (0, 0), textcoords="offset points",
                 xytext=(14, 6), ha='left', va='bottom', fontsize=10,
                 fontweight='bold', color='red')

    # Baseline.
    ax1.axhline(0, color='black', alpha=0.25, lw=1, zorder=0)

    ax1.set(aspect='equal')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.tick_params(bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    ax1.grid(True, linestyle=':', alpha=0.5)

    y_top = short[:, 1].max() + 1
    ax1.set_ylim(-1, y_top)
    x_ext = abs(short[:, 0]).max() + 1
    ax1.set_xlim(-x_ext, x_ext)

    # Right-side address table.
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", size="35%", pad=0.6)
    ax2.axis('off')

    table_data = []
    wire_lengths = []
    for t in range(1, table_size + 1):
        x, y = PTS[t - 1]
        wl = int(abs(x) + abs(y))
        wire_lengths.append(wl)
        table_data.append(['', t, wl])

    table = ax2.table(
        cellText=table_data,
        colLabels=['', 'd', 'Wire length'],
        cellLoc='center',
        bbox=[0, 0, 1, 1],
        colWidths=[0.08, 0.4, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='medium')
            cell.set_facecolor('#f2f2f2')
        elif col == 0:
            wl = wire_lengths[row - 1]
            cell.set_facecolor(RING_COLORS[min(wl, MAX_RING) - 1])
            cell.set_edgecolor('white')

    return fig


if __name__ == "__main__":
    fig = render_frame()
    fig.savefig('simplified_explicit_communication_model.svg',
                bbox_inches='tight')
    fig.savefig('simplified_explicit_communication_model.png',
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved simplified_explicit_communication_model.svg and .png")
