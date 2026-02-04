"""Optimization diagnostics for PCAPG."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_pcapg_convergence(history: Dict[str, Sequence[float]],
                           filename: str) -> str:
    """Plot objective function and reconstruction trajectories."""
    objective = np.asarray(history.get('objective', []), dtype=float)
    reconstruction = np.asarray(history.get('reconstruction', []), dtype=float)
    if objective.size == 0:
        raise ValueError("PCAPG convergence plot requires recorded objective values.")
    iterations = np.arange(1, objective.size + 1)

    fig, ax1 = plt.subplots(figsize=(10.8, 6.24), dpi=340, facecolor='#f4f5fb')
    ax1.set_facecolor('#ffffff')
    ax1.plot(iterations,
             objective,
             marker='o',
             markersize=4.5,
             linewidth=2.4,
             color='#0d47a1',
             label='Objective value')
    ax1.set_xlabel("Iteration", fontsize=13)
    ax1.set_ylabel("Objective value", color='#0d47a1', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='#0d47a1')
    ax1.grid(True, linestyle=(0, (2, 4)), linewidth=0.7, alpha=0.4)

    ax2 = ax1.twinx()
    if reconstruction.size:
        ax2.plot(iterations[:reconstruction.size],
                 reconstruction,
                 marker='s',
                 markersize=4.0,
                 linewidth=2.0,
                 color='#b71c1c',
                 label='Reconstruction error')
        ax2.set_ylabel("Normalized reconstruction error", color='#b71c1c', fontsize=13)
        ax2.tick_params(axis='y', labelcolor='#b71c1c')

    handles, labels = ax1.get_legend_handles_labels()
    if reconstruction.size:
        rec_handles, rec_labels = ax2.get_legend_handles_labels()
        handles += rec_handles
        labels += rec_labels
    ax1.legend(handles,
               labels,
               loc='upper right',
               frameon=False,
               fontsize=10,
               ncol=1)

    fig.suptitle("PCAPG convergence diagnostics", fontsize=20.4, fontweight='bold', y=0.97)
    fig.text(0.5,
             0.91,
             "Alternating minimization stabilizes once the objective plateaus",
             ha='center',
             fontsize=12.6,
             color='#4a4a4a')
    fig.tight_layout(rect=[0.04, 0.05, 0.96, 0.9])
    fig.savefig(filename, dpi=340, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("PCAPG convergence plot saved to:", filename)
    return filename
