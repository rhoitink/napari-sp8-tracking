"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from time import time
from typing import TYPE_CHECKING

import numpy as np
import trackpy as tp
from magicgui import magic_factory
from napari.utils.notifications import Notification

if TYPE_CHECKING:
    import napari


@magic_factory
def particle_tracking_settings_widget(
    viewer: "napari.viewer.Viewer",
    img_layer: "napari.layers.Image",
    feature_size_xy_µm: float = 0.3,
    feature_size_z_µm: float = 0.3,
    min_separation_xy_µm: float = 0.3,
    min_separation_z_µm: float = 0.3,
    min_mass: float = 1e4,
):
    if "aicsimage" not in img_layer.metadata:
        raise ValueError(
            "Data not loaded via aicsimageio plugin, cannot extract metadata"
        )

    img = img_layer.metadata["aicsimage"]
    stack = np.squeeze(img.data)
    nz, ny, nx = stack.shape
    pixel_sizes = np.array(
        [getattr(img.physical_pixel_sizes, dim) for dim in ["Z", "Y", "X"]]
    )

    feature_sizes = np.array(
        [
            np.ceil(feature_size_z_µm / pixel_sizes[0]) // 2 * 2 + 1,
            np.ceil(feature_size_xy_µm / pixel_sizes[1]) // 2 * 2 + 1,
            np.ceil(feature_size_xy_µm / pixel_sizes[2]) // 2 * 2 + 1,
        ]
    )

    min_separations = np.array(
        [
            np.ceil(min_separation_z_µm / pixel_sizes[0]) // 2 * 2 + 1,
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
        Notification(
            "Increasing z-size to {:}".format(
                feature_sizes[0] * pixel_sizes[0]
            )
        )

    t = time()

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

    # @todo: fix notifications, not showing up currently
    Notification(
        f"{np.shape(coords)[0]} features found, took {time()-t:.2f} s"
    )

    # @todo: fix size of points
    viewer.add_points(
        np.array(coords[["z", "y", "x"]]),
        scale=pixel_sizes,
    )
