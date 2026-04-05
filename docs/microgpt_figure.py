import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Colors
EMB_COLOR = '#a8d8ea'
NORM_COLOR = '#e8e8e8'
LINEAR_COLOR = '#ffd3b6'
ATTN_COLOR = '#ffaaa5'
MLP_COLOR = '#98d1a0'
HEAD_COLOR = '#d5aaff'
RESIDUAL_COLOR = '#cccccc'

def box(y, label, color, width=4, height=0.7, x=3):
    rect = mpatches.FancyBboxPatch((x, y), width, height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', lw=1.2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=9, fontweight='bold')

def arrow(y_from, y_to, x=5):
    ax.annotate('', xy=(x, y_to + 0.7), xytext=(x, y_from),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

def side_label(y, label, x=7.5):
    ax.text(x, y + 0.35, label, ha='left', va='center', fontsize=8, color='#555555')

# Title
ax.text(5, 19.3, 'microGPT Architecture (test config)', ha='center', va='center',
        fontsize=13, fontweight='bold')
ax.text(5, 18.7, 'vocab=4, embd=4, heads=2, head_dim=2, 1 layer', ha='center', va='center',
        fontsize=9, color='#666666')

# Input
y = 17.5
box(y, 'Token Embedding', EMB_COLOR)
side_label(y, 'wte[token_id]  4x4 -> [4]')
arrow(y, y - 0.9)

y = 16.3
box(y, 'Position Embedding', EMB_COLOR)
side_label(y, 'wpe[pos_id]  4x4 -> [4]')
arrow(y, y - 0.9)

y = 15.1
box(y, 'Add + RMSNorm', NORM_COLOR)
side_label(y, 'x = rmsnorm(tok + pos)  [4]')
arrow(y, y - 0.9)

# Attention block
y = 13.9
box(y, 'RMSNorm', NORM_COLOR)
side_label(y, '[4] -> [4]')
arrow(y, y - 0.9)

y = 12.7
box(y, 'Q = Linear(x, Wq)', LINEAR_COLOR)
side_label(y, 'Wq: 4x4  [4] -> [4]')

y = 11.8
box(y, 'K = Linear(x, Wk)', LINEAR_COLOR)
side_label(y, 'Wk: 4x4  [4] -> [4]')

y = 10.9
box(y, 'V = Linear(x, Wv)', LINEAR_COLOR)
side_label(y, 'Wv: 4x4  [4] -> [4]')
arrow(10.9, 10.0)

y = 9.6
box(y, 'Multi-Head Attention', ATTN_COLOR)
side_label(y, '2 heads x dim 2, Q*K/sqrt(2)')
arrow(y, y - 0.9)

y = 8.4
box(y, 'Linear(attn, Wo)', LINEAR_COLOR)
side_label(y, 'Wo: 4x4  [4] -> [4]')
arrow(y, y - 0.9)

y = 7.2
box(y, 'Residual Add', RESIDUAL_COLOR)
side_label(y, 'x = attn_out + x_residual  [4]')
arrow(y, y - 0.9)

# MLP block
y = 6.0
box(y, 'RMSNorm', NORM_COLOR)
side_label(y, '[4] -> [4]')
arrow(y, y - 0.9)

y = 4.8
box(y, 'Linear(x, fc1)', MLP_COLOR)
side_label(y, 'fc1: 16x4  [4] -> [16]')
arrow(y, y - 0.9)

y = 3.6
box(y, 'ReLU', MLP_COLOR)
side_label(y, '[16] -> [16]')
arrow(y, y - 0.9)

y = 2.4
box(y, 'Linear(x, fc2)', MLP_COLOR)
side_label(y, 'fc2: 4x16  [16] -> [4]')
arrow(y, y - 0.9)

y = 1.2
box(y, 'Residual Add', RESIDUAL_COLOR)
side_label(y, 'x = mlp_out + x_residual  [4]')
arrow(y, y - 0.9)

y = 0.0
box(y, 'Linear(x, lm_head)', HEAD_COLOR)
side_label(y, 'lm_head: 4x4  [4] -> [4] logits')

# Cost annotation
ax.text(5, -1.0, 'ByteDMD cost = 7047', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#cc0000')

plt.savefig('microgpt_figure.svg', bbox_inches='tight')
plt.savefig('microgpt_figure.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved microgpt_figure.svg and microgpt_figure.png")
