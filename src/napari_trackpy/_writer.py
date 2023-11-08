from typing import Any, Dict, List

import numpy as np


def xyz_file_writer(
    path: str, layer_data: Any, attributes: Dict[str, Any]
) -> List[str]:
    coordinates = np.array(layer_data)
    header = f"{len(coordinates)}\nProperties=Pos:R:3"

    if "particle_tracking_pixel_sizes" in attributes["metadata"]:
        # scale coordinates with pixel size
        coordinates *= np.array(
            attributes["metadata"]["particle_tracking_pixel_sizes"]
        )
        header += ' Unit="micrometer"'

    coordinates = coordinates[:, [2, 1, 0]]  # change from ZYX to XYZ

    if "particle_tracking_settings" in attributes["metadata"]:
        # save particle tracking settings to txt file
        with open(
            path.replace(".xyz", "_params.txt"), "w", encoding="utf-8"
        ) as f:
            for key, val in attributes["metadata"][
                "particle_tracking_settings"
            ].items():
                f.write(f"{key} = {val}\n")

    np.savetxt(path, coordinates, comments="", header=header)  # save .xyz file

    return [path]
