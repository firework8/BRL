from os.path import abspath, dirname, join
 
import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt


FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")
 
 
MOUSE_10X_COLORS = {
    0: "#FF4A46",
    1: "#FF4A46",
    2: "#FF4A46",
    3: "#FF4A46",
    4: "#FF4A46",
    5: "#FFFF00",
    6: "#1CE6FF",
    7: "#FF34FF",
    8: "#008941",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
}
 
def plot(
        x,
        y,
        ax=None,
        title=None,
        draw_legend=False,
        draw_centers=False,
        draw_cluster_labels=False,
        colors=None,
        legend_kwargs=None,
        label_order=None,
        save_name = None,
        **kwargs
):
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
 
    if title is not None:
        ax.set_title(title)
 
    # plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}
    plot_params = {"alpha": kwargs.get("alpha", 1), "s": kwargs.get("s", 1)}
    # 1
    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    
    point_colors = list(map(colors.get, y))
    
    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)
 
        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=60, alpha=1, edgecolor="k"
        )
        # 48
        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label + 1,
                    fontsize=kwargs.get("fontsize", 14),
                    horizontalalignment="center",
                )
 
    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")
 
    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)
        plt.show()
    plt.savefig(save_name)
