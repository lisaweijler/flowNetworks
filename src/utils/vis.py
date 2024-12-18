import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import abc
from pathlib import Path
from typing import List


from ..utils.loggingmanager import LoggingManager


def mrd_plot(mrd_list_gt, mrd_list_pred, f1_score, filenames=None):
    """
    Creates a plot for true and predicted mrd values.
    Interactive, for wandb
    """

    if filenames is None:
        filenames = [str(n) for n in range(len(mrd_list_gt))]

    min_val = 1.0e-5

    # Set min mrd to min_val:
    mrd_list_gt = [mrd if mrd >= min_val else min_val for mrd in mrd_list_gt]
    mrd_list_pred = [mrd if mrd >= min_val else min_val for mrd in mrd_list_pred]

    # Create figure
    data = pd.DataFrame(
        list(zip(mrd_list_gt, mrd_list_pred, f1_score, filenames)),
        columns=["gt", "pred", "f1_score", "names"],
    )
    fig = px.scatter(
        data,
        x="gt",
        y="pred",
        log_x=True,
        log_y=True,
        range_x=[min_val, 1],
        range_y=[min_val, 1],
        color="f1_score",
        template="simple_white",
        hover_name="names",
    )

    # Add diagonal line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color="Gray",
            width=2,
            dash="dashdot",
        ),
    )

    # Add vertical line
    fig.add_shape(
        type="line",
        x0=5.0e-4,
        y0=0,
        x1=5.0e-4,
        y1=1,
        line=dict(
            color="Gray",
            width=2,
            dash="dot",
        ),
    )

    # Add horizontal line
    fig.add_shape(
        type="line",
        x0=0,
        y0=5.0e-4,
        x1=1,
        y1=5.0e-4,
        line=dict(
            color="Gray",
            width=2,
            dash="dot",
        ),
    )
    return fig


class BasePlot(abc.ABC):

    SAVE_AS_SVG_IN_ADDITION = False

    def __init__(self, filepath: Path, caption: str, ax: matplotlib.axes = plt.gca()):
        self.filepath = filepath
        self.caption = caption
        self.ax = ax
        matplotlib.rcParams["axes.linewidth"] = 2

    def showPlot(self):
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def savePlot(self):
        plt.savefig(self.filepath, dpi=80)
        if BasePlot.SAVE_AS_SVG_IN_ADDITION and self.filepath.name.endswith(".png"):
            plt.savefig(
                self.filepath.parent / Path(self.filepath.name.replace(".png", ".svg")),
                dpi=300,
            )
        plt.cla()
        plt.clf()
        plt.close()
        LoggingManager.get_default_logger().info(
            f"successfully saved plot with caption '{self.caption}' to path: '{self.filepath}'"
        )

    @abc.abstractmethod
    def createPlot(self):
        pass

    def generatePlotFile(self):
        self.createPlot()
        self.savePlot()


class PanelPlot(BasePlot):
    """
    TODO: update for multiclass prediction, now only for binary classification
    """

    def __init__(
        self,
        filepath: Path,
        caption: str,
        events: np.ndarray,
        target: np.ndarray,
        marker_list: List[str],
        panels: List[List[str]],
        min_fig_size: int,
        cluster_idx_orig: np.ndarray = None,
        n_points: int = 10000,
        ax: matplotlib.axes = plt.gca(),
        use_new_ax=True,
    ):

        self.data = events
        self.target = target
        self.panels = panels
        self.marker_list = marker_list
        self.min_fig_size = min_fig_size
        self.use_new_ax = use_new_ax
        self.n_points = n_points
        self.cluster_idx_orig = cluster_idx_orig

        super(PanelPlot, self).__init__(filepath, caption, ax)

    def createPlot(self):
        """
        Creates a scatter plot for the marker combinations given in the config panel.
        """
        num_plots = len(self.panels)

        fig, ax = plt.subplots(
            1, num_plots, figsize=(self.min_fig_size * num_plots, self.min_fig_size * 1)
        )

        plt.suptitle(self.caption)

        for p in range(num_plots):
            m0_str = self.panels[p][0]
            m0 = self.marker_list.index(m0_str)
            m1_str = self.panels[p][1]
            m1 = self.marker_list.index(m1_str)

            x_data = self.data[: self.n_points, m0]
            y_data = self.data[: self.n_points, m1]
            blast_mask = self.target[: self.n_points] > 0.5
            colors_orig = [
                "red" if x > 0.5 else "grey" for x in self.target[: self.n_points]
            ]

            marker_size_orig = [
                1.5 if colors_orig[n] == "red" else 0.75
                for n, m in enumerate(colors_orig)
            ]

            if self.cluster_idx_orig is not None:
                ax[p].scatter(
                    x_data, y_data, s=0.75, c=self.cluster_idx_orig[: self.n_points]
                )
            else:
                ax[p].scatter(
                    x_data[~blast_mask], y_data[~blast_mask], s=1.5, c="grey", alpha=0.7
                )
                ax[p].scatter(x_data[blast_mask], y_data[blast_mask], s=1.5, c="red")

                if p == num_plots - 1:
                    ax[p].legend(["cancerous", "healthy"])
            ax[p].set_xlabel(m0_str)
            ax[p].set_ylabel(m1_str)


class PanelPlotTargetVSPrediction(BasePlot):
    """
    TODO: update for multiclass prediction, now only for binary classification
    """

    def __init__(
        self,
        filepath: Path,
        caption: str,
        events: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        marker_list: List[str],
        panels: List[List[str]],
        min_fig_size: int,
        n_points: int = 10000,
        ax: matplotlib.axes = plt.gca(),
        use_new_ax=True,
    ):

        self.data = events
        self.target = target
        self.prediction = prediction
        self.panels = panels
        self.marker_list = marker_list
        self.min_fig_size = min_fig_size
        self.use_new_ax = use_new_ax
        self.n_points = n_points

        super(PanelPlotTargetVSPrediction, self).__init__(filepath, caption, ax)

    def createPlot(self):
        """
        Creates a scatter plot for the marker combinations given in theconfig panel.
        """
        num_plots = len(self.panels)

        fig, ax = plt.subplots(
            2, num_plots, figsize=(self.min_fig_size * num_plots, self.min_fig_size * 2)
        )

        plt.suptitle(self.caption)

        for p in range(num_plots):
            m0_str = self.panels[p][0]
            m0 = self.marker_list.index(m0_str)
            m1_str = self.panels[p][1]
            m1 = self.marker_list.index(m1_str)

            x_data = self.data[: self.n_points, m0]
            y_data = self.data[: self.n_points, m1]
            colors_GT = [
                "red" if x > 0.5 else "blue" for x in self.target[: self.n_points]
            ]
            colors_pred = [
                "red" if x > 0.0 else "blue" for x in self.prediction[: self.n_points]
            ]

            marker_size_GT = [
                1.5 if colors_GT[n] == "red" else 0.75 for n, m in enumerate(colors_GT)
            ]
            marker_size_pred = [
                1.5 if colors_GT[n] == "red" else 0.75
                for n, m in enumerate(colors_pred)
            ]

            ax[0, p].scatter(x_data, y_data, s=marker_size_pred, c=colors_pred)
            ax[0, p].set_xlabel(m0_str)
            ax[0, p].set_ylabel(m1_str)

            ax[1, p].scatter(x_data, y_data, s=marker_size_GT, c=colors_GT)
            ax[1, p].set_xlabel(m0_str)
            ax[1, p].set_ylabel(m1_str)

            if p == 0:
                ax[0, p].title.set_text("Prediction")
                ax[1, p].title.set_text("Target")
