"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from time import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
from magicgui import magic_factory
from magicgui.tqdm import tqdm
from matplotlib.backends.backend_qt5agg import FigureCanvas
from napari.utils import notifications
from superqt.utils import thread_worker

if TYPE_CHECKING:
    import napari

fig_added = False
fig, ax = None, None


@magic_factory(
    min_mass={"widget_type": "SpinBox", "max": int(1e8)},
)
def xyz_particle_tracking_settings_widget(
    viewer: "napari.viewer.Viewer",
    img_layer: "napari.layers.Image",
    feature_size_xy_µm: float = 0.3,
    feature_size_z_µm: float = 0.3,
    min_separation_xy_µm: float = 0.3,
    min_separation_z_µm: float = 0.3,
    min_mass=int(1e5),
    show_plots: bool = False,
):
    if img_layer is None:
        notifications.show_error("No image selected")
        return

    if "aicsimage" not in img_layer.metadata:
        notifications.show_error(
            "Data not loaded via aicsimageio plugin, cannot extract metadata"
        )
        return

    global fig_added, fig, ax
    if not fig_added:
        fig, ax = plt.subplots(1, 1)
        xyz_particle_tracking_settings_widget.native.layout().addWidget(
            FigureCanvas(fig)
        )
        fig_added = True

    with tqdm() as pbar:
        results = do_particle_tracking(
            img_layer,
            feature_size_xy_μm,  # noqa F821
            feature_size_z_μm,  # noqa F821
            min_separation_xy_μm,  # noqa F821
            min_separation_z_μm,  # noqa F821
            min_mass,
            show_plots,
        )
        results.returned.connect(
            lambda x: add_points_to_viewer(viewer, img_layer, x)
        )
        results.returned.connect(lambda x: show_mass_histogram(ax, x))
        results.finished.connect(lambda: pbar.progressbar.hide())
        results.start()


def add_points_to_viewer(viewer, img_layer, output):
    coords, pixel_sizes = output
    # @todo: fix size of points
    viewer.add_points(
        np.array(coords[["z", "y", "x"]]),
        properties={"mass": coords["mass"]},
        scale=pixel_sizes,
        edge_color="red",
        face_color="transparent",
        name=f"{img_layer.name}_coords",
        out_of_slice_display=True,
    )


def show_mass_histogram(axis, output):
    coords, pixel_sizes = output
    axis.cla()
    axis.hist(coords["mass"], "auto")
    axis.set_xlabel("mass (a.u.)")
    axis.set_ylabel("occurence")
    axis.figure.tight_layout()
    axis.figure.canvas.draw()


@thread_worker
def do_particle_tracking(
    img_layer: "napari.layers.Image",
    feature_size_xy_µm: float,
    feature_size_z_µm: float,
    min_separation_xy_µm: float,
    min_separation_z_µm: float,
    min_mass,
    show_plots: bool,
):
    img = img_layer.metadata["aicsimage"]

    # tracking code implementation based on `sp8_xyz_tracking_lif.py` by Maarten Bransen
    stack = np.squeeze(
        img_layer.data_raw
    )  # squeeze out dimensions with length 1
    nz, ny, nx = stack.shape
    pixel_sizes = np.array(
        [getattr(img.physical_pixel_sizes, dim) for dim in ["Z", "Y", "X"]]
    )

    # convert feature size and min_separation to pixel units

    feature_sizes = np.array(
        [
            np.ceil(feature_size_z_µm / np.abs(pixel_sizes[0])) // 2 * 2 + 1,
            np.ceil(feature_size_xy_µm / pixel_sizes[1]) // 2 * 2 + 1,
            np.ceil(feature_size_xy_µm / pixel_sizes[2]) // 2 * 2 + 1,
        ]
    )

    min_separations = np.array(
        [
            np.ceil(min_separation_z_µm / np.abs(pixel_sizes[0])) // 2 * 2 + 1,
            np.ceil(min_separation_xy_µm / pixel_sizes[1]) // 2 * 2 + 1,
            np.ceil(min_separation_xy_µm / pixel_sizes[2]) // 2 * 2 + 1,
        ]
    )

    # disallow equal sizes in all dimensions
    if (
        feature_sizes[2] == feature_sizes[1]
        and feature_sizes[2] == feature_sizes[0]
    ):
        feature_sizes[0] += 2
        notifications.show_warning(
            f"Increasing z-size to {feature_sizes[0] * np.abs(pixel_sizes[0])}"
        )

    t = time()

    # trackpy particle tracking with set parameters
    coords = tp.locate(
        stack,
        diameter=feature_sizes,
        minmass=min_mass,
        separation=min_separations,
        characterize=False,
    )
    coords = coords.dropna(subset=["x", "y", "z", "mass"])
    # coords = coords.loc[(coords['mass']<max_mass)]
    coords = coords.loc[
        (
            (coords["x"] >= feature_sizes[2] / 2)
            & (coords["x"] <= nx - feature_sizes[2] / 2)
            & (coords["y"] >= feature_sizes[1] / 2)
            & (coords["y"] <= ny - feature_sizes[1] / 2)
            & (coords["z"] >= feature_sizes[0] / 2)
            & (coords["z"] <= nz - feature_sizes[0] / 2)
        )
    ]

    if show_plots:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1, 1)
        ax.hist(coords["mass"], bins="auto", fc="blue", ec="k")
        ax.set_title("Histogram of particle mass")
        ax.set_xlabel("Mass")
        ax.set_ylabel("Occurence")
        plt.tight_layout()
        plt.show(block=False)

    notifications.show_info(
        f"{np.shape(coords)[0]} features found, took {time()-t:.2f} s"
    )

    return (coords, pixel_sizes)
