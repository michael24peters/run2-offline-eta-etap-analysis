"""
Plot DCB fit results from JSON file. Uses plot_mass_fit.py as a generic plotting
utilty for any fit result from a JSON file. Here we provide custom text and
formatting for the DCB + Chebyshev fit parameters.

Usage:
    python plot_dcb.py fit_result.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_utils
import plot_style

from plot_mass_fit import plot_fit, plot_fit_data

def build_text(params, chi2_per_ndof):
    mean, mean_err   = plot_utils.val_err(params['mean'])
    sigma, sigma_err = plot_utils.val_err(params['sigma'])
    n_sig, n_sig_err = plot_utils.val_err(params['n_sig'])
    n_bkg, n_bkg_err = plot_utils.val_err(params['n_bkg'])

    # DCB tail parameters (asymmetric)
    alphaL, alphaL_err = plot_utils.val_err(params.get('alphaL', {}))
    nL, nL_err         = plot_utils.val_err(params.get('nL', {}))
    alphaR, alphaR_err = plot_utils.val_err(params.get('alphaR', {}))
    nR, nR_err         = plot_utils.val_err(params.get('nR', {}))

    # Background shape parameters vary by model -- c0/c1 for the Chebyshev,
    # k0/k1 for the exponentiated Chebyshev, and so on for any future entry
    # in fit_utils.BACKGROUND_MODELS. Whatever key in
    # `params` isn't one of the known signal/yield names above IS a
    # background parameter, so its row is built from whatever is actually
    # there instead of a hardcoded pair.
    SIGNAL_KEYS = {'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'n_sig', 'n_bkg'}
    bkg_row = [(name, name, *plot_utils.val_err(params[name]), '.3f', '')
               for name in params if name not in SIGNAL_KEYS]

    # Build summary text
    # Expected format:
    #   (latex_symbol, plain_symbol, value, error, format string, unit)
    stats = [
        [(r'\mu', 'μ', mean, mean_err, '.2f', 'MeV')],
        [(r'\sigma', 'σ', sigma, sigma_err, '.2f', 'MeV')],
        [(r'\alpha_L', 'α_L', alphaL, alphaL_err, '.2f', ''),
         (r'n_L', 'n_L', nL, nL_err, '.1f', '')],
        [(r'\alpha_R', 'α_R', alphaR, alphaR_err, '.2f', ''),
         (r'n_R', 'n_R', nR, nR_err, '.2f', '')],
        [(r'n_{sig}', 'n_sig', n_sig, n_sig_err, '.0f', '')],
        [(r'n_{bkg}', 'n_bkg', n_bkg, n_bkg_err, '.0f', '')],
        bkg_row,
    ]
    # Values which are not a value-error pair (e.g., chi2/ndof). Must provide
    # exact strings for desired LaTeX and terminal output.
    extra = (
        fr'$\chi^2/ndof$ = {chi2_per_ndof:.2f}',
        f'χ²/ndof = {chi2_per_ndof:.2f}',
    )
    return plot_style.build_stats_text(stats, extra=extra)

# Default configuration for plot_dcb.py, which can be overridden by the caller
# (i.e., the driver script)
DEFAULT_CONFIG = {
    'description'    : 'Plot DCB fit results from JSON file',
    'output'         : 'dcb_mass_fit.png',  # output filename
    'title'          : 'Eta Mass',  # plot title
    "xlim"           : (400, 700),
    # "pull_ylim"      : (-3, 3),
    'font'           : 'sans',   # "serif" (default, STIXGeneral) or "sans" (Inter)
    # Pinned upper left: the stats box is tall enough that "auto" placement can
    # drop it over the peak. The legend defaults to the upper right, so the two
    # do not collide.
    'text_loc'       : 'upper left',   # "auto" or a plot_style.POSITIONS key
    'show_stats'     : True,   # set False to omit the stats box entirely
    # "legend_loc"     : "upper right",  # any matplotlib legend loc string, default "best"
    'build_text'     : build_text,
}

# This executes if this script is run from somewhere else.
def plot_dcb(data, config=None):
    """
    Render a DCB fit plot: DEFAULT_CONFIG with any caller overrides layered on
    top, which can override any of DEFAULT_CONFIG's keys.
    """
    plot_fit_data(data, {**DEFAULT_CONFIG, **(config or {})})

# This executes if this script is run directly.
if __name__ == '__main__':
    plot_fit(DEFAULT_CONFIG)
