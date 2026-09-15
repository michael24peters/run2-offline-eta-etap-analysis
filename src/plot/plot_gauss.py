"""
Plot Gaussian fit results from JSON file. Uses plot_mass_fit.py as a generic
plotting utilty for any fit result from a JSON file. Here we provide custom text
and formatting for the Gaussian + Chebyshev fit parameters.

Usage:
    python plot_gauss.py fit_result.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_utils
import plot_style

from plot_mass_fit import plot_fit, plot_fit_data

def build_text(params, chi2_per_ndof):
    mean,  mean_err  = plot_utils.val_err(params['mean'])
    sigma, sigma_err = plot_utils.val_err(params['sigma'])

    stats = [
        [(r'\mu', 'μ', mean, mean_err, '.2f', 'MeV')],
        [(r'\sigma', 'σ', sigma, sigma_err, '.2f', 'MeV')],
    ]
    extra = (
        fr'$\chi^2/ndof$ = {chi2_per_ndof:.2f}',
        f'χ²/ndof = {chi2_per_ndof:.2f}',
    )
    return plot_style.build_stats_text(stats, extra=extra)

DEFAULT_CONFIG = {
    'description'    : 'Plot Gaussian fit results from JSON file',
    'output'         : 'gauss_mass_fit.png',  # output filename
    # "xlim"           : (420, 680),
    # "pull_ylim"      : (-3, 3),
    'font'           : 'sans',   # "serif" (default, STIXGeneral) or "sans" (Inter)
    'show_stats'     : True,   # set False to omit the stats box entirely
    'build_text'     : build_text,
}

# This executes if this script is run from somewhere else.
def plot_gauss(data, config=None):
    """Render a Gaussian fit plot: DEFAULT_CONFIG with any caller overrides layered on top."""
    plot_fit_data(data, {**DEFAULT_CONFIG, **(config or {})})

# This executes if this script is run directly.
if __name__ == '__main__':
    plot_fit(DEFAULT_CONFIG)
