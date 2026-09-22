"""
Plot Gaussian fit results from JSON file. Uses plot_mass_fit.py as a generic
plotting utilty for any fit result from a JSON file. Here we provide custom text
and formatting for the Gaussian + Chebyshev fit parameters.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_utils
import plot_style

from plot_mass_fit import plot_fit_data

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

def plot_gauss(data, output, x_lim=None, title=None):
    """Render a Gaussian fit plot."""
    plot_fit_data(data, output, build_text, x_lim=x_lim, title=title)
