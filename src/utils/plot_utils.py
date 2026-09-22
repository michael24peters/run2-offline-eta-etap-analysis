"""
plot_utils.py

Generic plotting helpers: font setup, JSON I/O, and param-dict unpacking.

For "house style" presentation logic (stats-box text, text placement,
save/report boilerplate), see plot_style.py.
"""

import json

import matplotlib.pyplot as plt

def apply_font():
    """Set up the Inter font, falling back to the default sans-serif if it's
    not installed."""
    family = 'Inter'
    try:
        import matplotlib.font_manager as fm
        fm.findfont('Inter', fallback_to_default=False)
    except Exception:
        print('[INFO] Inter font not found, falling back to default sans-serif.')
        family = 'sans-serif'
    plt.rcParams['font.family'] = family
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

def load_data(infile):
    with open(infile) as f: return json.load(f)

def val_err(param):
    return param.get('value', float('nan')), param.get('error', float('nan'))
