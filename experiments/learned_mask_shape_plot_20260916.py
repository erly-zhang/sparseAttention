"""CPU-only visual summary of sampled final kernel indices."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/project/yl258/jz648/sparseAttention/experiments/outputs/learned_mask_shape_20260916')


def main():
    for variant in ('per_cluster', 'full_mask'):
        folder = ROOT / variant
        summary = json.loads((folder / 'shape_summary.json').read_text())
        records = [row for file in sorted((folder / 'masks').glob('layer*.json'))
                   for row in json.loads(file.read_text()) if not row['is_representative']]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        layer = summary['per_layer']
        axes[0, 0].plot(range(32), [x['overlap'] * 100 for x in layer], label='Final vs Shared')
        axes[0, 0].plot(range(32), [x['member_overlap'] * 100 for x in layer], label='Within-group members')
        axes[0, 0].set(xlabel='Layer (0-based)', ylabel='Intersection / 10240 (%)', ylim=(0, 102))
        axes[0, 0].legend()
        for method, color in [('shared', '#666666'), ('final', '#008c95')]:
            hist = np.mean([x[method]['position_hist'] for x in records], axis=0) / 10240 * 100
            axes[0, 1].plot((np.arange(32) + .5) / 32, hist, label=method, color=color)
        axes[0, 1].set(xlabel='Key position / visible prefix length', ylabel='Selected keys in bin (%)')
        axes[0, 1].legend()
        data = np.load(folder / 'masks/layer16.npz')
        target = next(x for x in records if x['layer'] == 16)
        head, group = target['head'], target['group']
        bins = np.linspace(0, int(data['n']), 513)
        for ax, name, keys in [(axes[1, 0], 'Shared', data['shared'][group]),
                               (axes[1, 1], 'NN final', data['final'][head])]:
            density = np.stack([np.histogram(k, bins=bins)[0] / np.diff(bins) for k in keys])
            im = ax.imshow(density, aspect='auto', origin='lower', vmin=0, vmax=1,
                           extent=(0, int(data['n']), -.5, len(keys)-.5), cmap='viridis')
            ax.set(title=f'{name}: layer 16, member head {head}', xlabel='Original key position',
                   ylabel='Sampled query-tile index')
        fig.colorbar(im, ax=list(axes[1]), label='Selected fraction per key-position bin')
        fig.suptitle(f'{variant}: one aligned input, 32 layers, 8 sparse tiles/layer\n'
                     'Shared comparator uses identical Q/K; figures exclude representative heads')
        fig.savefig(folder / 'mask_shape.png', dpi=160)
        plt.close(fig)
        print('PLOT', folder / 'mask_shape.png')


if __name__ == '__main__':
    main()
