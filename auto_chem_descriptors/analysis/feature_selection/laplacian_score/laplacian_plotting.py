#!/usr/bin/python3
"""
Central plotting orchestrator for Laplacian/MLS rankings.

Delegates each figure to specialized modules so every output honors the
requested SRP-style modularization.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .plot_ls_ranking import plot_ls_ranking
from .plot_mls_ranking import plot_mls_ranking


def generate_laplacian_plots(feature_names: Sequence[str],
                             payload: Dict[str, np.ndarray],
                             config: Dict[str, object]) -> Dict[str, str | None]:
    ls_filename = plot_ls_ranking(feature_names,
                                  payload['ls_scores'],
                                  str(config.get('ls_plot_filename', 'plot_laplacian_ls.png')),
                                  int(config.get('top_descriptors', 20)))

    mls_filename = None
    if np.isfinite(payload['mls_scores']).any():
        mls_filename = plot_mls_ranking(feature_names,
                                        payload['mls_scores'],
                                        payload['margin_coverage'],
                                        str(config.get('mls_plot_filename', 'plot_laplacian_mls.png')),
                                        int(config.get('top_descriptors', 20)))
    else:
        print("Marginal Laplacian Score plot skipped (MLS disabled or undefined).")

    return {
        'ls': ls_filename,
        'mls': mls_filename,
    }
