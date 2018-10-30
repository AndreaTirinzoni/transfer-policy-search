import numpy as np
import matplotlib.pyplot as plt

MARKERS = ["o", "D", "s", "^", "v", "p", "*"]
COLORS = ["#0e5ad3", "#bc2d14", "#22aa16", "#a011a3", "#d1ba0e", "#14ccc2", "#d67413"]
LINES = ["solid", "dashed", "dashdot", "dotted", "solid", "dashed", "dashdot", "dotted"]


def plot_curves(x_data, y_mean_data, y_std_data=None, title="", x_label="Episodes", y_label="Performance", names=None,
                file_name=None):
    """
    Plots a list of N curves.
    :param x_data: a list of N elements containing the x-values for each curve
    :param y_mean_data: a list of N elements containing the mean of the y-values for each curve
    :param y_std_data: a list of N elements containing the width of the confidence interval for each curve
    :param title: the plot title
    :param x_label: the label of the x-axis
    :param y_label: the label of the y-axis
    :param names: a list of N elements containing the name of each curve (draw a legend if given)
    :param file_name: name of the file where to save the image (saves only if given)
    """
    assert len(x_data) < 8

    plt.style.use('ggplot')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    # plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['figure.titlesize'] = 20

    fig, ax = plt.subplots()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    X = np.array(x_data)
    plt.xlim([X.min(), X.max()])

    for i in range(len(x_data)):

        ax.plot(x_data[i], y_mean_data[i], linewidth=3, color=COLORS[i], marker=None, markersize=8.0,
                linestyle="solid", label=names[i] if names is not None else None)
        if y_std_data is not None:
            ax.fill_between(x_data[i], y_mean_data[i] - y_std_data[i], y_mean_data[i] + y_std_data[i],
                            facecolor=COLORS[i], edgecolor=COLORS[i], alpha=0.3)

    if names is not None:
        ax.legend(loc='best', numpoints=1, fancybox=True, frameon=False)

    if file_name is not None:
        plt.savefig(file_name + ".pdf", format='pdf')

    plt.show()