# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the 3D bounding box and the diameter of 3D object models."""
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
import numpy as np

# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "lnd1",
    # Type of input object models.
    "model_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": "/home/utsav/IProject/data/dataset",
}
################################################################################

# Load dataset parameters.
dp_model = dataset_params.get_model_params(
    p["datasets_path"], p["dataset"], p["model_type"]
)

models_info = {}
for obj_id in dp_model["obj_ids"]:
    misc.log("Processing model of object {}...".format(obj_id))

    model = inout.load_ply(dp_model["model_tpath"].format(obj_id=obj_id))

    # Calculate 3D bounding box.
    ref_pt = np.array(list(map(float, model["pts"].min(axis=0).flatten())))
    size = np.array(list(map(float, (model["pts"].max(axis=0) - ref_pt).flatten())))

    # Calculate diameter.
    diameter = misc.calc_pts_diameter(model["pts"])

    models_info[obj_id] = {
        "min_x": ref_pt[0].item(),
        "min_y": ref_pt[1].item(),
        "min_z": ref_pt[2].item(),
        "size_x": size[0].item(),
        "size_y": size[1].item(),
        "size_z": size[2].item(),
        "diameter": diameter,
    }

# Save the calculated info about the object models.
inout.save_json(dp_model["models_info_path"], models_info)
