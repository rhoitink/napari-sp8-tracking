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
from napari.utils import notifications
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class ParticleTrackingWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self.setup_gui()

    def setup_gui(self) -> None:
        self.imagelayer_label = QLabel("Image")
        self.imagelayer_value = QComboBox()
        self.imagelayer_value.setEditable(False)
        self.imagelayer_value.addItems(
            [layer.name for layer in self._viewer.layers]
        )

        self.featuresize_xy_label = QLabel("Feature size xy (µm)")

        self.featuresize_xy_value = QDoubleSpinBox()
        self.featuresize_xy_value.setMinimum(0.0)
        self.featuresize_xy_value.setSingleStep(0.1)
        self.featuresize_xy_value.setValue(0.5)

        self.featuresize_z_label = QLabel("Feature size xy (µm)")

        self.featuresize_z_value = QDoubleSpinBox()
        self.featuresize_z_value.setMinimum(0.0)
        self.featuresize_z_value.setSingleStep(0.1)
        self.featuresize_z_value.setValue(0.5)

        self.minsep_xy_label = QLabel("Min. separation xy (µm)")

        self.minsep_xy_value = QDoubleSpinBox()
        self.minsep_xy_value.setMinimum(0.0)
        self.minsep_xy_value.setSingleStep(0.1)
        self.minsep_xy_value.setValue(0.5)

        self.minsep_z_label = QLabel("Min. separation z (µm)")

        self.minsep_z_value = QDoubleSpinBox()
        self.minsep_z_value.setMinimum(0.0)
        self.minsep_z_value.setSingleStep(0.1)
        self.minsep_z_value.setValue(0.5)

        self.minmass_label = QLabel("Min. mass")

        self.minmass_value = QSpinBox()
        self.minmass_value.setMinimum(0)
        self.minmass_value.setSingleStep(1000)
        self.minmass_value.setMaximum(2**31 - 1)
        self.minmass_value.setValue(1000)

        self.showplots_label = QLabel("Show plots?")
        self.showplots_value = QCheckBox()

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._run)

        self.setLayout(QGridLayout())

        self.layout().addWidget(self.featuresize_xy_label, 0, 0)
        self.layout().addWidget(self.featuresize_xy_value, 0, 1)
        self.layout().addWidget(self.featuresize_z_label, 1, 0)
        self.layout().addWidget(self.featuresize_z_value, 1, 1)

        self.layout().addWidget(self.minsep_xy_label, 2, 0)
        self.layout().addWidget(self.minsep_xy_value, 2, 1)
        self.layout().addWidget(self.minsep_z_label, 3, 0)
        self.layout().addWidget(self.minsep_z_value, 3, 1)

        self.layout().addWidget(self.minmass_label, 4, 0)
        self.layout().addWidget(self.minmass_value, 4, 1)

        self.layout().addWidget(self.showplots_label, 5, 0)
        self.layout().addWidget(self.showplots_value, 5, 1)

        self.layout().addWidget(self.imagelayer_label, 6, 0)
        self.layout().addWidget(self.imagelayer_value, 6, 1)

        self.layout().addWidget(self.run_button, 8, 0)

    def _run(self):
        notifications.show_info("Button was clicked!")


@magic_factory
def particle_tracking_settings_widget(
    viewer: "napari.viewer.Viewer",
    img_layer: "napari.layers.Image",
    feature_size_xy_µm: float = 0.3,
    feature_size_z_µm: float = 0.3,
    min_separation_xy_µm: float = 0.3,
    min_separation_z_µm: float = 0.3,
    min_mass=1e5,
    show_plots: bool = False,
):
    if "aicsimage" not in img_layer.metadata:
        raise ValueError(
            "Data not loaded via aicsimageio plugin, cannot extract metadata"
        )

    img = img_layer.metadata["aicsimage"]

    # tracking code implementation based on `sp8_xyz_tracking_lif.py` by Maarten Bransen
    stack = np.squeeze(img.data)
    nz, ny, nx = stack.shape
    pixel_sizes = np.array(
        [getattr(img.physical_pixel_sizes, dim) for dim in ["Z", "Y", "X"]]
    )

    # convert feature size and min_separation to pixel units

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
        notifications.show_warning(
            "Increasing z-size to {:}".format(
                feature_sizes[0] * pixel_sizes[0]
            )
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

    print(coords.head())

    if show_plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.hist(coords["mass"], bins="auto", fc="blue", ec="k")
        ax.set_title("Histogram of particle mass")
        ax.set_xlabel("Mass")
        ax.set_ylabel("Occurence")
        plt.tight_layout()
        plt.show(block=False)

    notifications.show_info(
        f"{np.shape(coords)[0]} features found, took {time()-t:.2f} s"
    )

    # @todo: fix size of points
    viewer.add_points(
        np.array(coords[["z", "y", "x"]]),
        properties={"mass": coords["mass"]},
        scale=pixel_sizes,
        edge_color="red",
        face_color="transparent",
        name=f"{img_layer.name}_coordinates",
    )
