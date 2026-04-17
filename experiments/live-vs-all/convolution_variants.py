#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Spatial convolution vs FFT convolution — ByteDMD comparison.

Two algorithms computing the same circular convolution y = x (*) h:
  1. spatial_conv  — direct O(N^2) sliding-window inner product
  2. fft_conv      — FFT-based O(N log N) via DFT, pointwise multiply, IDFT

Mirrors the naive-attention vs flash-attention comparison:
  - Spatial conv is the "naive" approach with good locality but O(N^2) work.
  - FFT conv is the "clever" approach with fewer ops but must materialize
    full frequency-domain arrays, analogous to materializing the attention matrix.

Usage:
    uv run --script convolution_variants.py [N_values]
    uv run --script convolution_variants.py 4,8,16,32
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import bytedmd_ir as b2


# ============================================================================
# Convolution implementations (pure Python, traceable)
# ============================================================================

def spatial_conv(x, h):
    """Circular convolution y[n] = sum_k x[(n-k) % N] * h[k], direct O(N^2)."""
    N = len(x)
    y = [None] * N
    for n in range(N):
        acc = x[n % N] * h[0]
        for k in range(1, N):
            acc = acc + x[(n - k) % N] * h[k]
        y[n] = acc
    return y


def _fft_recursive(x_re, x_im, inverse=False):
    """Radix-2 DIT FFT on paired real/imag lists. Returns (re, im) lists.

    Twiddle factors are plain floats (not tracked) — only signal values
    generate LOAD/STORE events in the tracer.
    """
    N = len(x_re)
    if N == 1:
        return x_re[:], x_im[:]

    # Split into even/odd
    even_re = [x_re[i] for i in range(0, N, 2)]
    even_im = [x_im[i] for i in range(0, N, 2)]
    odd_re  = [x_re[i] for i in range(1, N, 2)]
    odd_im  = [x_im[i] for i in range(1, N, 2)]

    # Recurse
    E_re, E_im = _fft_recursive(even_re, even_im, inverse)
    O_re, O_im = _fft_recursive(odd_re, odd_im, inverse)

    # Combine with butterfly
    out_re = [None] * N
    out_im = [None] * N
    sign = 1.0 if inverse else -1.0
    for k in range(N // 2):
        angle = sign * 2.0 * math.pi * k / N
        w_re = math.cos(angle)
        w_im = math.sin(angle)
        # Complex multiply: (O_re + i*O_im) * (w_re + i*w_im)
        # = (O_re*w_re - O_im*w_im) + i*(O_re*w_im + O_im*w_re)
        t_re = O_re[k] * w_re - O_im[k] * w_im
        t_im = O_re[k] * w_im + O_im[k] * w_re
        out_re[k]         = E_re[k] + t_re
        out_im[k]         = E_im[k] + t_im
        out_re[k + N//2]  = E_re[k] - t_re
        out_im[k + N//2]  = E_im[k] - t_im

    return out_re, out_im


def fft_conv(x, h):
    """Circular convolution via FFT: IFFT(FFT(x) * FFT(h)), O(N log N).

    Materializes full frequency-domain arrays X and H, analogous to
    materializing the N×N attention matrix in naive attention.
    """
    N = len(x)
    zero = [x[0] - x[0]] * N  # tracked zeros

    # Forward FFTs
    X_re, X_im = _fft_recursive(list(x), list(zero))
    H_re, H_im = _fft_recursive(list(h), list(zero))

    # Pointwise complex multiply: Y = X * H
    Y_re = [None] * N
    Y_im = [None] * N
    for k in range(N):
        Y_re[k] = X_re[k] * H_re[k] - X_im[k] * H_im[k]
        Y_im[k] = X_re[k] * H_im[k] + X_im[k] * H_re[k]

    # Inverse FFT
    y_re, y_im = _fft_recursive(Y_re, Y_im, inverse=True)

    # Scale by 1/N (constant, not tracked)
    inv_N = 1.0 / N
    y = [y_re[n] * inv_N for n in range(N)]
    return y


# ============================================================================
# Experiment runner
# ============================================================================

ALGORITHMS = [
    ('Spatial convolution (direct)', spatial_conv),
    ('FFT convolution',              fft_conv),
]

MEASURES = [
    ('lp_lb',           'LP lower',     'black',      'v', ':'),
    ('bytedmd_live',    'DMD-live',     'tab:green',  '^', '-'),
    ('ripple',          'Ripple Shift', 'tab:purple', 'D', '-'),
    ('tombstone',       'Tombstone',    'tab:blue',   's', '--'),
    ('bytedmd_classic', 'Classic DMD',  'tab:red',    'o', '-'),
]


def make_inputs(N):
    x = [1.0] * N
    h = [1.0] * N
    return x, h


def run_one(func, N):
    x, h = make_inputs(N)
    l2, _ = b2.trace(func, (x, h))
    results = {'N': N}
    results['bytedmd_classic'] = b2.bytedmd_classic(l2)
    results['bytedmd_live']    = b2.bytedmd_live(l2)
    results['lp_lb']           = b2.lp_lower_bound(l2)
    for key in ('ripple', 'tombstone'):
        l3 = b2.ALLOCATORS[key](l2)
        results[key] = b2.cost(l3)
    return results


def collect(Ns):
    table = {}
    for label, func in ALGORITHMS:
        rows = []
        for N in Ns:
            print(f'  {label}  N={N}', end='', flush=True)
            row = run_one(func, N)
            rows.append(row)
            print(f"  lp_lb={row['lp_lb']:,.0f}  live={row['bytedmd_live']:,}"
                  f"  classic={row['bytedmd_classic']:,}")
        table[label] = rows
    return table


def plot(table, Ns, out_path):
    fig, axes = plt.subplots(len(ALGORITHMS), 1,
                              figsize=(10, 6.5 * len(ALGORITHMS)), sharex=True)
    if len(ALGORITHMS) == 1:
        axes = [axes]

    for ax, (label, _) in zip(axes, ALGORITHMS):
        rows = table[label]
        Ns_arr = np.array([r['N'] for r in rows])

        for key, legend, color, marker, linestyle in MEASURES:
            ys = np.array([r[key] for r in rows], dtype=float)
            ax.loglog(Ns_arr, ys, color=color, marker=marker, linestyle=linestyle,
                      linewidth=2.2, markersize=9, label=legend, zorder=3)

        ax.set_xlabel('Signal length N', fontsize=12)
        ax.set_ylabel('Total cost', fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.95)

    fig.suptitle('Circular Convolution: Spatial (direct) vs FFT\n'
                 'ByteDMD cost under various allocator models',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def plot_comparison(table, Ns, out_path):
    """Side-by-side comparison of spatial vs FFT across measures."""
    fig, ax = plt.subplots(figsize=(10, 6))
    Ns_arr = np.array(Ns)

    for key, legend, color, marker, linestyle in MEASURES:
        for label, _ in ALGORITHMS:
            rows = table[label]
            ys = np.array([r[key] for r in rows], dtype=float)
            short = 'Spatial' if 'Spatial' in label else 'FFT'
            ax.loglog(Ns_arr, ys, color=color, marker=marker, linestyle=linestyle,
                      linewidth=2 if short == 'Spatial' else 1.5,
                      alpha=1.0 if short == 'Spatial' else 0.6,
                      markersize=8, label=f'{legend} ({short})', zorder=3)

    ax.set_xlabel('Signal length N', fontsize=12)
    ax.set_ylabel('Total cost', fontsize=12)
    ax.set_title('Spatial vs FFT Convolution — All Measures', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95, ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def main():
    Ns_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if Ns_arg:
        Ns = [int(x) for x in Ns_arg.split(',')]
    else:
        Ns = [4, 8, 16, 32]

    table = collect(Ns)

    out_dir = os.path.dirname(__file__)
    plot(table, Ns, os.path.join(out_dir, 'convolution_envelope.png'))
    plot_comparison(table, Ns, os.path.join(out_dir, 'convolution_comparison.png'))

    # Summary table
    print()
    for label, _ in ALGORITHMS:
        print(f'\n{label}')
        rows = table[label]
        display_names = [legend for _, legend, _, _, _ in MEASURES]
        col_w = max(14, max(len(n) for n in display_names))
        header = f"  {'N':>3} | " + " | ".join(n.rjust(col_w) for n in display_names)
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for r in rows:
            cells = " | ".join(
                f"{r[m[0]]:>{col_w},.0f}" if isinstance(r[m[0]], float) else f"{r[m[0]]:>{col_w},}"
                for m in MEASURES)
            print(f"  {r['N']:>3} | {cells}")


if __name__ == '__main__':
    main()
