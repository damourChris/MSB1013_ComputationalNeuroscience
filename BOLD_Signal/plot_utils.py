def get_plt_settings():
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.serif": "Times",
        "font.family": "serif",
        'text.latex.preamble': r'\usepackage{mathptmx} \usepackage{amsmath}',
        "figure.titlesize": 11,
        'axes.titlesize': 10,
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.loc": 'upper right'
    }
    return tex_fonts


def get_plt_size(percentage_textwidth=0.8):
    """Convert textwidth of latex document to inches"""
    width = (455 * percentage_textwidth) / 72.27
    height = width / 1.618
    return width, height
