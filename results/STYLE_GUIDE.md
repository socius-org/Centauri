# Figure Style Guide

Reference for all publication figures in this project. All scripts under `results/` should follow these conventions.

## Model Naming

Always use full model names in labels:
- Finetuned Llama: `Llama-Centaur-1B`, `Llama-Centaur-3B`, `Llama-Centaur-8B`
- Finetuned Qwen: `Qwentaur-0.6B`, `Qwentaur-1.7B`, etc.
- Base Llama: `Llama-3.2-1B`, `Llama-3.2-3B`, `Llama-3.1-8B`
- Base Qwen: `Qwen3-0.6B`, `Qwen3-1.7B`, etc.
- 70B: `Centaur-70B` (unique model, no prefix needed)

For multi-line x-axis labels where space is tight, break at hyphens:
`'Llama\nCentaur\n1B'`, `'Qwentaur\n0.6B'`, `'Llama-3.2\n1B'`

## Style Context

```python
import scienceplots  # noqa: F401

with plt.style.context(['nature']):
    apply_style()
    # ... figure code ...
```

## Core Parameters (`apply_style`)

```python
def apply_style(fs=8, fl=5.5):
    plt.rcParams.update({
        'font.size': fs,           # 8
        'axes.labelsize': fs,      # 8
        'xtick.labelsize': fs - 1, # 7
        'ytick.labelsize': fs - 1, # 7
        'legend.fontsize': fl,     # 5.5
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
    })
```

## Colour Palette

| Role | Hex | Usage |
|---|---|---|
| Llama / Llama-Centaur | `#0082fb` | Primary blue |
| Centaur-70B | `#005bb5` | Dark blue |
| Llama-Centaur 4-bit | `#66b3fd` | Light blue |
| Qwen / Qwentaur | `#7F6DEF` | Primary purple |
| Qwentaur 4-bit | `#b3a8f5` | Light purple |
| Cognitive model ref | `#888888` | Gray |

Gradient shading (lighter = smaller model):
```python
def tint(hex_color, amount=0.0):
    """Lighten by mixing with white. amount=0 is original, 0.55 is lightest."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])
```

Typical gradient: `amount = 0.55 * (1 - idx / max(n-1, 1))` where idx=0 is smallest.

## Markers

| Model family | Marker | Usage |
|---|---|---|
| Llama-Centaur | `'o'` (circle) | Finetuned: solid fill. Base: hollow (`markerfacecolor='none'`) |
| Qwentaur | `'s'` (square) | Finetuned: solid fill. Base: hollow |
| Centaur-70B | `'D'` (diamond) | Always solid, dark blue |

Standard sizes:
- `markersize=6`, `markeredgewidth=0.8` (normal)
- `markersize=8`, `markeredgewidth=1.0` (70B emphasis)
- `markeredgecolor='white'` for filled markers

## Bars

- `edgecolor='white'`, `linewidth=0.5`
- Error bars: `capsize=2`, `error_kw={'linewidth': 0.5}`
- 4-bit / base distinction: `hatch='///'`, lighter tint, or hollow
- Typical bar width: `0.7` (single), `0.35` (paired)

## Reference Lines

| Line | Style |
|---|---|
| Cognitive model baseline | `linestyle=':'`, `linewidth=0.8`, `alpha=0.7`, `color='gray'` |
| Zero / chance | `linestyle='--'`, `linewidth=0.3-0.5`, `color='gray'` |
| Group separator | `linestyle='--'`, `linewidth=0.6`, `alpha=0.5`, `color='gray'` |

## Legends

- Default fontsize: `5.5` (from `apply_style`)
- Compact layout: `borderpad=0.3`, `handlelength=1.5`, `handletextpad=0.4`, `labelspacing=0.3`
- Multi-column: `ncol=2` or `ncol=3` as needed
- `frameon=False` for overlaid legends; default frame for corner placement

## Panel Labels

- Bold, lowercase: `'a'`, `'b'`, etc.
- Placed via `set_panel_title()` or in `ax.set_title('a  Title', fontweight='bold', fontsize=9)` for simple cases
- Panel title fontsize: 7-9 (7 for per-panel, 8-9 for prominent titles)

## Figure Sizing

- Single panel: `(3.5, 2.6)` to `(4.5, 3.5)`
- Two-panel side-by-side: `(7, 3)` to `(7, 3.2)`
- Wide multi-panel: `(10, 4)` to `(14, 4.5)`
- Grid layouts: use `fig.add_axes()` for precise control

## Saving

```python
def save_figure(fig, outdir, basename, dpi=600):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, f'{basename}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
```

- Formats: **PNG + PDF** (not JPG)
- DPI: **600**
- Always: `bbox_inches='tight'`, `facecolor='white'`

## Axis Labels

- Use `'Mean negative log-likelihood'` (not `'Mean NLL'` with arrows)
- `'Parameters (billions)'` for x-axis on scaling plots
- Log-scale x-axis ticks: `['$10^0$', '$10^1$', '$10^2$']`

## Spine Handling

With `['nature']` style context, top and right spines are hidden by default. Do not manually set `ax.spines[...].set_visible(False)`.

## Text Encoding

Avoid Unicode characters (checkmarks, box-drawing) in `print()` statements -- they fail on Windows cp1252. Use ASCII alternatives.
