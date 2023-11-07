from time import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
from matplotlib.backends.backend_qt5agg import FigureCanvas
from napari.utils import notifications
from superqt.utils import thread_worker

if TYPE_CHECKING:
    import napari
    import napari.layers
    import napari.viewer

from typing import cast

from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    FloatSpinBox,
    PushButton,
    create_widget,
)


class XYZWidget(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self.viewer = viewer
        self.img_layer = None
        self.img_layer_name = cast(
            ComboBox,
            create_widget(annotation=napari.layers.Image, label="Image"),
        )
        self.feature_size_xy_µm = FloatSpinBox(
            value=0.3,
            min=0.0,
            max=1e3,
            step=0.05,
            label="Feature size xy (µm)",
        )
        self.feature_size_z_µm = FloatSpinBox(
            value=0.5, min=0.0, max=1e3, step=0.05, label="Feature size z (µm)"
        )
        self.min_separation_xy_µm = FloatSpinBox(
            value=0.3,
            min=0.0,
            max=1e3,
            step=0.05,
            label="Min. separation xy(µm)",
        )
        self.min_separation_z_µm = FloatSpinBox(
            value=0.3,
            min=0.0,
            max=1e3,
            step=0.05,
            label="Max. separation z(µm)",
        )
        self.min_mass = FloatSlider(
            value=1e2, min=0, max=1e9, label="Min. mass"
        )
        self.max_mass = FloatSlider(
            value=1e8, min=1, max=1e9, label="Max. mass"
        )

        self.run_btn = PushButton(label="Run")
        self.reset_btn = PushButton(enabled=False, label="Reset")
        self.save_params_btn = PushButton(
            enabled=False, label="Save tracking parameters"
        )
        self.save_tracking_btn = PushButton(
            enabled=False, label="Save coordinates"
        )
        self.run_btn = PushButton(label="Run")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.reset_btn.clicked.connect(self.reset)
        self.img_layer_name.changed.connect(self._on_label_layer_changed)

        self.last_added_points_layer = None
        self.fig, self.ax = None, None

        super().__init__(
            widgets=[
                self.img_layer_name,
                self.feature_size_xy_µm,
                self.feature_size_z_μm,
                self.min_separation_xy_μm,
                self.min_separation_z_μm,
                self.min_mass,
                self.max_mass,
                self.run_btn,
                self.reset_btn,
                self.save_params_btn,
                self.save_tracking_btn,
            ]
        )

    @thread_worker
    def do_particle_tracking(self):
        img = self.img_layer.metadata["aicsimage"]

        # tracking code implementation based on `sp8_xyz_tracking_lif.py` by Maarten Bransen
        stack = np.squeeze(
            self.img_layer.data_raw
        )  # squeeze out dimensions with length 1
        nz, ny, nx = stack.shape
        self.pixel_sizes = np.array(
            [getattr(img.physical_pixel_sizes, dim) for dim in ["Z", "Y", "X"]]
        )

        # convert feature size and min_separation to pixel units

        feature_sizes = np.array(
            [
                np.ceil(
                    self.feature_size_z_µm.value / np.abs(self.pixel_sizes[0])
                )
                // 2
                * 2
                + 1,
                np.ceil(self.feature_size_xy_µm.value / self.pixel_sizes[1])
                // 2
                * 2
                + 1,
                np.ceil(self.feature_size_xy_µm.value / self.pixel_sizes[2])
                // 2
                * 2
                + 1,
            ]
        )

        min_separations = np.array(
            [
                np.ceil(
                    self.min_separation_z_µm.value
                    / np.abs(self.pixel_sizes[0])
                )
                // 2
                * 2
                + 1,
                np.ceil(self.min_separation_xy_µm.value / self.pixel_sizes[1])
                // 2
                * 2
                + 1,
                np.ceil(self.min_separation_xy_µm.value / self.pixel_sizes[2])
                // 2
                * 2
                + 1,
            ]
        )

        # disallow equal sizes in all dimensions
        if (
            feature_sizes[2] == feature_sizes[1]
            and feature_sizes[2] == feature_sizes[0]
        ):
            feature_sizes[0] += 2
            notifications.show_warning(
                f"Increasing z-size to {feature_sizes[0] * np.abs(self.pixel_sizes[0])}"
            )

        t = time()

        # trackpy particle tracking with set parameters
        coords = tp.locate(
            stack,
            diameter=feature_sizes,
            minmass=self.min_mass.value,
            separation=min_separations,
            characterize=False,
        )
        coords = coords.dropna(subset=["x", "y", "z", "mass"])
        coords = coords.loc[(coords["mass"] < self.max_mass.value)]
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

        self.coords = coords.copy()
        del coords

        notifications.show_info(
            f"{np.shape(self.coords)[0]} features found, took {time()-t:.2f} s"
        )

        return (self.coords, self.pixel_sizes)

    def _on_label_layer_changed(self, new_value: napari.layers.Image):
        # print(self.img_layer_name, type(self.img_layer_name), type(new_value))
        self.img_layer = new_value
        # set your internal annotation layer here.

    def _on_run_clicked(self):
        if self.img_layer is None:
            notifications.show_error("No image selected")
            return

        if "aicsimage" not in self.img_layer.metadata:
            notifications.show_error(
                "Data not loaded via aicsimageio plugin, cannot extract metadata"
            )
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.native.layout().addWidget(FigureCanvas(self.fig))

        self.run_btn.enabled = False
        tracking_thread = self.do_particle_tracking()
        tracking_thread.finished.connect(lambda: self.process_tracking())
        # tracking_thread.finished.connect(lambda: )
        tracking_thread.start()

    def process_tracking(self):
        self.add_points_to_viewer()
        self.show_mass_histogram()
        self.run_btn.enabled = True
        self.reset_btn.enabled = True
        # self.save_params_btn.enabled = True
        # self.save_tracking_btn.enabled = True

    def add_points_to_viewer(self):
        # @todo: fix size of points
        self.last_added_points_layer = self.viewer.add_points(
            np.array(self.coords[["z", "y", "x"]]),
            properties={"mass": self.coords["mass"]},
            scale=self.pixel_sizes,
            edge_color="red",
            face_color="transparent",
            name=f"{self.img_layer.name}_coords",
            out_of_slice_display=True,
        )

    def show_mass_histogram(self):
        self.reset_histogram()
        self.ax.hist(self.coords["mass"], "auto")
        self.ax.figure.tight_layout()
        self.ax.figure.canvas.draw()

    def reset_histogram(self):
        self.ax.cla()
        self.ax.set_xlabel("mass (a.u.)")
        self.ax.set_ylabel("occurence")
        self.ax.figure.tight_layout()
        self.ax.figure.canvas.draw()

    def reset(self):
        self.reset_histogram()
        self.coords = None
        self.pixel_sizes = None
        if self.last_added_points_layer is not None:
            self.viewer.layers.remove(self.last_added_points_layer.name)
