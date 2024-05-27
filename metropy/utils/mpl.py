import matplotlib as mpl
import matplotlib.pyplot as plt

CMP = mpl.colormaps["Set2"]
COLOR_LIST = mpl.colormaps["Set3"].colors

PARAMETERS = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "figure.dpi": 600,
    "font.size": 12,
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.6,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
}
plt.rcParams.update(PARAMETERS)

def set_size(width=470, ratio="golden", fraction=1.0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    if ratio == "golden":
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * float(ratio)
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def get_figure(width=470, ratio="golden", fraction=0.8):
    fig, ax = plt.subplots(figsize=set_size(width, ratio, fraction))
    return fig, ax
