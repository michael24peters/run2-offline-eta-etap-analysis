"""
plot_utils.py

Plotting helpers: the custom font (apply_font) and fit-parameter unpacking (val_err). The custom
style of the figures (stats box, labels, ticks, saving) is in plot_style.py.
"""

import matplotlib.pyplot as plt


def apply_font():
    """Set up the Inter font, falling back to the default sans-serif if it's not installed."""
    family = 'Inter'
    try:
        import matplotlib.font_manager as fm
        fm.findfont('Inter', fallback_to_default=False)
    except Exception:
        print('[INFO] Inter font not found, falling back to default sans-serif.')
        family = 'sans-serif'
    plt.rcParams['font.family'] = family
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

def val_err(param):
    """(value, error) of a fit-parameter dict {'value': ..., 'error': ...}, nan where missing."""
    return param.get('value', float('nan')), param.get('error', float('nan'))
