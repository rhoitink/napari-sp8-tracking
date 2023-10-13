try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._widget import xyz_particle_tracking_settings_widget

__all__ = [
    "xyz_particle_tracking_settings_widget",
]
